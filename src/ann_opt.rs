use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, BufReader, Write, Read};
use std::path::Path;
use std::collections::HashMap;
use smallvec::SmallVec;
use dashmap::DashMap;

#[derive(Clone)]
pub struct AnnIndex {
    pub dims: usize,
    pub bits: usize,
    pub projections: Vec<Vec<f32>>,
    pub buckets: HashMap<u16, SmallVec<[usize; 16]>>,
}

impl AnnIndex {
    pub fn build(vectors: &[f32], dims: usize, rows: usize, bits: usize, seed: u64) -> Self {
        assert_eq!(vectors.len(), dims * rows, "Invalid vector length");

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let projections: Vec<Vec<f32>> = (0..bits)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let buckets = DashMap::<u16, SmallVec<[usize; 16]>>::new();

        (0..rows).into_par_iter().for_each(|i| {
            let vec_i = &vectors[i * dims..(i + 1) * dims];
            let mut hash = 0u16;

            for (j, proj) in projections.iter().enumerate().take(bits.min(16)) {
                let dot = vec_i.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
                if dot >= 0.0 {
                    hash |= 1 << j;
                }
            }

            buckets.entry(hash).or_default().push(i);
        });

        let buckets_final = buckets.into_iter().collect::<HashMap<_, _>>();

        Self {
            dims,
            bits,
            projections,
            buckets: buckets_final,
        }
    }

    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        assert_eq!(query.len(), self.dims);

        let mut hash = 0u16;
        for (j, proj) in self.projections.iter().enumerate().take(self.bits.min(16)) {
            let dot = query.iter().zip(proj).map(|(a, b)| a * b).sum::<f32>();
            if dot >= 0.0 {
                hash |= 1 << j;
            }
        }

        self.buckets
            .get(&hash)
            .map(|v| v.to_vec())
            .unwrap_or_else(Vec::new)
    }

    pub fn query(&self, query: &[f32], top_k: usize, vectors: &[f32]) -> Vec<(usize, f32)> {
        let candidates = self.query_candidates(query);

        let mut results: Vec<(usize, f32)> = candidates
            .par_iter()
            .map(|&i| {
                let offset = i * self.dims;
                let vec_i = &vectors[offset..offset + self.dims];
                let score = query.iter().zip(vec_i).map(|(a, b)| a * b).sum();
                (i, score)
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);
        results
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

        let num_buckets = self.buckets.len() as u32;
        file.write_all(&num_buckets.to_le_bytes())?;

        for (&hash, ids) in &self.buckets {
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids {
                file.write_all(&(id as u32).to_le_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, _vectors: &[f32]) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);

        let mut u32_buf = [0u8; 4];

        file.read_exact(&mut u32_buf)?;
        let dims = u32::from_le_bytes(u32_buf) as usize;

        file.read_exact(&mut u32_buf)?;
        let bits = u32::from_le_bytes(u32_buf) as usize;

        let mut projections = vec![vec![0f32; dims]; bits];
        for proj in &mut projections {
            for val in proj.iter_mut() {
                let mut f32_buf = [0u8; 4];
                file.read_exact(&mut f32_buf)?;
                *val = f32::from_le_bytes(f32_buf);
            }
        }

        file.read_exact(&mut u32_buf)?;
        let num_buckets = u32::from_le_bytes(u32_buf);

        let mut buckets = HashMap::new();

        for _ in 0..num_buckets {
            let mut hash_buf = [0u8; 2];
            file.read_exact(&mut hash_buf)?;
            let hash = u16::from_le_bytes(hash_buf);

            file.read_exact(&mut u32_buf)?;
            let len = u32::from_le_bytes(u32_buf);

            let mut ids = SmallVec::with_capacity(len as usize);
            for _ in 0..len {
                file.read_exact(&mut u32_buf)?;
                ids.push(u32::from_le_bytes(u32_buf) as usize);
            }

            buckets.insert(hash, ids);
        }

        Ok(Self {
            dims,
            bits,
            projections,
            buckets,
        })
    }
}
