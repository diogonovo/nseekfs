// ann_opt.rs – High-Performance ANN Index (LSH + SIMD-safe + Parallel)

use std::collections::HashMap;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::{File};
use std::io::{BufWriter, BufReader, Write, Read};
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone)]
pub struct AnnIndex {
    pub dims: usize,
    pub bits: usize,
    pub projections: Vec<Vec<f32>>,
    pub buckets: HashMap<u64, Vec<usize>>,
    pub checksum: [u8; 32],
}

impl AnnIndex {
    pub fn build(vectors: &[f32], dims: usize, rows: usize, bits: usize, seed: u64) -> Self {
        assert!(vectors.len() >= dims * rows);
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let projections: Vec<Vec<f32>> = (0..bits)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        use std::sync::Mutex;
        let buckets = Mutex::new(HashMap::new());

        (0..rows).into_par_iter().for_each(|i| {
            let offset = i * dims;
            let vec_i = &vectors[offset..offset + dims];
            let mut hash = 0u64;
            for (j, proj) in projections.iter().enumerate() {
                let dot = vec_i.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
                if dot >= 0.0 {
                    hash |= 1 << j;
                }
            }
            let mut guard = buckets.lock().unwrap();
            guard.entry(hash).or_insert_with(Vec::new).push(i);

        });

        let checksum = Self::compute_checksum(vectors);

        Self {
            dims,
            bits,
            projections,
            buckets: buckets.into_inner().unwrap(),
            checksum,
        }
    }

    pub fn query(&self, query: &[f32], top_k: usize, vectors: &[f32]) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dims);

        let mut hash = 0u64;
        for (j, proj) in self.projections.iter().enumerate() {
            let dot = query.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
            if dot >= 0.0 {
                hash |= 1 << j;
            }
        }

        let candidates = match self.buckets.get(&hash) {
            Some(c) => c,
            None => return vec![],
        };

        let results: Vec<(usize, f32)> = candidates
            .par_iter()
            .map(|&i| {
                let offset = i * self.dims;
                let vec_i = &vectors[offset..offset + self.dims];
                let score = query.iter().zip(vec_i).map(|(a, b)| a * b).sum();
                (i, score)
            })
            .collect();

        let mut sorted = results;
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.into_iter().take(top_k).collect()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&(self.dims as u32).to_le_bytes())?;
        file.write_all(&(self.bits as u32).to_le_bytes())?;

        for proj in &self.projections {
            for &f in proj {
                file.write_all(&f.to_le_bytes())?;
            }
        }

        file.write_all(&self.checksum)?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, vectors: &[f32]) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut u32_buf = [0u8; 4];

        file.read_exact(&mut u32_buf)?;
        let dims = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let bits = u32::from_le_bytes(u32_buf) as usize;

        let mut projections = vec![vec![0f32; dims]; bits];
        for proj in &mut projections {
            for f in proj.iter_mut() {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *f = f32::from_le_bytes(buf);
            }
        }

        let mut checksum = [0u8; 32];
        file.read_exact(&mut checksum)?;

        let actual = Self::compute_checksum(vectors);
        if checksum != actual {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Checksum mismatch"));
        }

        let rows = vectors.len() / dims;
        use std::sync::Mutex;
        let buckets = Mutex::new(HashMap::new());

        (0..rows).into_par_iter().for_each(|i| {
            let offset = i * dims;
            let vec_i = &vectors[offset..offset + dims];
            let mut hash = 0u64;
            for (j, proj) in projections.iter().enumerate() {
                let dot = vec_i.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
                if dot >= 0.0 {
                    hash |= 1 << j;
                }
            }
            let mut guard = buckets.lock().unwrap();
            guard.entry(hash).or_insert_with(Vec::new).push(i);

        });

        Ok(Self {
            dims,
            bits,
            projections,
            buckets: buckets.into_inner().unwrap(),
            checksum,
        })
    }

    fn compute_checksum(vectors: &[f32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vectors.as_ptr() as *const u8,
                vectors.len() * std::mem::size_of::<f32>(),
            )
        };
        hasher.update(bytes);
        hasher.finalize().into()
    }

    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        assert_eq!(query.len(), self.dims);

        let mut hash = 0u64;
        for (j, proj) in self.projections.iter().enumerate() {
            let dot = query.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
            if dot >= 0.0 {
                hash |= 1 << j;
            }
        }

        self.buckets.get(&hash).cloned().unwrap_or_else(Vec::new)
    }
}
