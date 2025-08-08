use rayon::prelude::*;
use std::cmp::Ordering;
use wide::f32x8;
use crate::engine::Engine;
use std::convert::TryInto;
use pdqselect::select_by;

const LANES: usize = 8;

fn compute_score(query: &[f32], vec: &[f32]) -> f32 {
    let chunks = query.len() / LANES;
    let mut simd_sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let q = f32x8::new(query[i * LANES..i * LANES + 8].try_into().unwrap());
        let v = f32x8::new(vec[i * LANES..i * LANES + 8].try_into().unwrap());
        simd_sum += q * v;
    }

    let mut total = simd_sum.reduce_add();
    for i in (chunks * LANES)..query.len() {
        total += query[i] * vec[i];
    }

    total
}

impl Engine {
    pub fn top_k_query_scalar(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        let query_ref = query;

        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(query_ref),
                None => (0..self.rows).collect(),
            }
        } else {
            (0..self.rows).collect()
        };

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .map(|i| {
                let offset = i * self.dims;
                let vec_i = &self.vectors[offset..offset + self.dims];
                let score = query_ref.iter().zip(vec_i).map(|(a, b)| a * b).sum::<f32>();
                (i, score)
            })
            .collect();

        if results.len() > k {
            select_by(&mut results, k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        Ok(results)
    }

    pub fn top_k_query_simd(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        let query_ref = query;

        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(query_ref),
                None => (0..self.rows).collect(),
            }
        } else {
            (0..self.rows).collect()
        };

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .map(|i| {
                let offset = i * self.dims;
                let vec_i = &self.vectors[offset..offset + self.dims];
                let score = compute_score(query_ref, vec_i);
                (i, score)
            })
            .collect();

        if results.len() > k {
            select_by(&mut results, k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        Ok(results)
    }

    pub fn top_k_query(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        if self.dims >= 64 && self.rows >= 1000 {
            self.top_k_query_simd(query, k)
        } else {
            self.top_k_query_scalar(query, k)
        }
    }
}
