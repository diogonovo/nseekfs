use log::{debug, info};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct AnnIndex {
    pub bits: usize,
    pub dims: usize,
    pub projections: Vec<Vec<f32>>,
    pub buckets: HashMap<u64, Vec<usize>>,
}

impl AnnIndex {
    pub fn build(
        vectors: &[f32],
        dims: usize,
        rows: usize,
        bits: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let bits = bits.unwrap_or(16);
        let seed = seed.unwrap_or(42);

        info!(
            "Building ANN index → bits={} seed={} dims={} rows={}",
            bits, seed, dims, rows
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let projections: Vec<Vec<f32>> = (0..bits)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        debug!("Generated {} random hyperplanes", bits);

        let buckets: HashMap<u64, Vec<usize>> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let vec_i = &vectors[i * dims..(i + 1) * dims];
                let hash = Self::hash_vector(vec_i, &projections);
                (hash, i)
            })
            .fold(HashMap::new, |mut local_map, (hash, idx)| {
                local_map.entry(hash).or_insert_with(Vec::new).push(idx);
                local_map
            })
            .reduce(HashMap::new, |mut acc, partial| {
                for (hash, mut indices) in partial {
                    acc.entry(hash)
                        .or_insert_with(Vec::new)
                        .append(&mut indices);
                }
                acc
            });

        info!(
            "Completed ANN index build → {} unique buckets",
            buckets.len()
        );

        Self {
            bits,
            dims,
            projections,
            buckets,
        }
    }

    fn hash_vector(vector: &[f32], projections: &[Vec<f32>]) -> u64 {
        let mut hash: u64 = 0;
        for (i, proj) in projections.iter().enumerate() {
            let dot = vector.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
            if dot > 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }

    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        let hash = Self::hash_vector(query, &self.projections);

        let candidates = self.buckets.get(&hash).cloned().unwrap_or_default();
        debug!(
            "ANN query → hash={} → {} candidates",
            hash,
            candidates.len()
        );

        candidates
    }
}
