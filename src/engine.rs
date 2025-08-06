use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::io::Result;

use fast_float::parse as fast_parse;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use bytemuck;
use crate::ann_opt::AnnIndex;
use crate::utils::vector::{cosine_similarity, normalize_vector_inplace, validate_dimensions};
use log::{debug, error, info, warn};


#[derive(Clone)]
pub struct Engine {
    pub vectors: Arc<[f32]>,
    pub dims: usize,
    pub rows: usize,
    pub use_ann: bool,
    pub ann_index: Option<AnnIndex>,
}

impl Engine {
    pub fn from_bin<P: AsRef<Path>>(path: P, use_ann: bool) -> std::io::Result<Self> {
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            log::error!("Bin file too small: {:?}", path.as_ref());
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary file too small",
            ));
        }

        let dims = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let rows = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let vector_bytes_len = 4 * dims * rows;
        let expected_len = 8 + vector_bytes_len + 32;

        if mmap.len() != expected_len {
            error!(
                "❌ Unexpected binary file size: expected {}, found {}",
                expected_len,
                mmap.len()
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unexpected binary file size",
            ));
        }

        let data_bytes = &mmap[8..8 + vector_bytes_len];

        info!(
            "✅ Loaded binary index with integrity: dims={} rows={} ANN={} path={:?}",
            dims,
            rows,
            use_ann,
            path.as_ref()
        );

        let data: &[f32] = bytemuck::cast_slice(data_bytes);
        let vectors = Arc::from(data);

        let ann_index = if use_ann {
            Some(AnnIndex::build(&vectors, dims, rows, 32, 42))
        } else {
            None
        };

        Ok(Self {
            vectors,
            dims,
            rows,
            use_ann,
            ann_index,
        })
    }

    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();

        let vector_bytes_len = 4 * self.dims * self.rows;
        let total_len = 8 + vector_bytes_len + 32;

        let file = File::create(path)?;
        file.set_len(total_len as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        mmap[0..4].copy_from_slice(&(self.dims as u32).to_le_bytes());
        mmap[4..8].copy_from_slice(&(self.rows as u32).to_le_bytes());

        let data_bytes: &[u8] = bytemuck::cast_slice(&self.vectors);
        mmap[8..8 + data_bytes.len()].copy_from_slice(data_bytes);

        mmap.flush()?;

        log::info!(
            "✅ Saved engine to binary: {:?} dims={} rows={} size={} bytes",
            path,
            self.dims,
            self.rows,
            total_len
        );

        Ok(())
    }

    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows {
            warn!("Index out of bounds: {} (rows={})", idx, self.rows);
            return None;
        }
        let start = idx * self.dims;
        let end = start + self.dims;
        Some(&self.vectors[start..end])
    }

    pub fn top_k_index(&self, idx: usize, k: usize) -> std::io::Result<Vec<(usize, f32)>> {
        debug!("Top-k index search → idx={} k={}", idx, k);

        let query = self.get_vector(idx).ok_or_else(|| {
            warn!("Invalid index in top_k_index: {}", idx);
            std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Invalid index: {}", idx))
        })?;

        self.top_k_query(query, k, false)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn top_k_subset(
        &self,
        query: &[f32],
        subset: &[usize],
        k: usize,
        normalize: bool,
    ) -> std::io::Result<Vec<(usize, f32)>> {
        if query.len() != self.dims {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Query vector has wrong dimension: expected {}, got {}",
                    self.dims,
                    query.len()
                ),
            ));
        }

        debug!(
            "Top-k subset search → k={} normalize={} subset_len={}",
            k,
            normalize,
            subset.len()
        );

        let mut query = query.to_vec();
        if normalize {
            normalize_vector_inplace(&mut query);
        }

        let mut results: Vec<(usize, f32)> = subset
            .par_iter()
            .filter_map(|&i| match self.get_vector(i) {
                Some(vec_i) => {
                    let score = cosine_similarity(&query, vec_i);
                    Some((i, score))
                }
                None => {
                    warn!("Index {} out of bounds in subset", i);
                    None
                }
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        debug!(
            "Subset search complete → returning top {} results",
            results.len()
        );

        Ok(results)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}
