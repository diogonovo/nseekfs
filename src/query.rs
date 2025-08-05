// src/query.rs

use rayon::prelude::*;
use std::cmp::Ordering;
use std::simd::{Simd, SimdFloat};
use crate::engine::Engine;

const LANES: usize = 16;

impl Engine {
    /// Versão escalar simples
    pub fn top_k_query_scalar(
        &self,
        query: &[f32],
        k: usize,
        normalize: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        let mut query = query.to_vec();
        if normalize {
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm <= std::f32::EPSILON {
                return Err("Query vector has zero norm and cannot be normalized.".to_string());
            }
            query.iter_mut().for_each(|x| *x /= norm);
        }

        let candidates: Vec<usize> = if self.use_ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(&query),
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
                let score = query.iter().zip(vec_i).map(|(a, b)| a * b).sum::<f32>();
                (i, score)
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        if results.len() > k {
            results.truncate(k);
        }

        Ok(results)
    }

    /// Versão otimizada com SIMD + paralelismo
    pub fn top_k_query_simd(
        &self,
        query: &[f32],
        k: usize,
        normalize: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        let mut query_vec = query.to_vec();
        if normalize {
            let norm = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm <= std::f32::EPSILON {
                return Err("Query vector has zero norm and cannot be normalized.".to_string());
            }
            query_vec.iter_mut().for_each(|x| *x /= norm);
        }

        let candidates: Vec<usize> = if self.use_ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(&query_vec),
                None => (0..self.rows).collect(),
            }
        } else {
            (0..self.rows).collect()
        };

        let vector_data = &self.vectors;
        let dims = self.dims;
        let query_ref = &query_vec;

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .map(|i| {
                let start = i * dims;
                let vec_i = &vector_data[start..start + dims];

                let mut simd_sum = Simd::<f32, LANES>::splat(0.0);
                let chunks = dims / LANES;

                for j in 0..chunks {
                    let q_chunk = Simd::from_slice(&query_ref[j * LANES..(j + 1) * LANES]);
                    let v_chunk = Simd::from_slice(&vec_i[j * LANES..(j + 1) * LANES]);
                    simd_sum += q_chunk * v_chunk;
                }

                let mut total = simd_sum.reduce_sum();
                for j in (chunks * LANES)..dims {
                    total += query_ref[j] * vec_i[j];
                }

                (i, total)
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        if results.len() > k {
            results.truncate(k);
        }

        Ok(results)
    }

    /// Versão auto que escolhe scalar ou simd
    pub fn top_k_query(
        &self,
        query: &[f32],
        k: usize,
        normalize: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        if self.dims >= 64 && self.rows >= 1000 {
            self.top_k_query_simd(query, k, normalize)
        } else {
            self.top_k_query_scalar(query, k, normalize)
        }
    }
} 
