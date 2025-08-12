use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use crate::ann_opt::AnnIndex;
use crate::utils::vector::{compute_similarity, SimilarityMetric};
use log::{debug, error, info, warn};

#[derive(Clone)]
pub struct Engine {
    pub vectors: Arc<[f32]>,
    pub dims: usize,
    pub rows: usize,
    pub ann: bool,
    pub ann_index: Option<Arc<AnnIndex>>,
}

impl Engine {
    pub fn from_bin<P: AsRef<Path>>(path: P, ann: bool) -> std::io::Result<Self> {
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
        let expected_len = 8 + vector_bytes_len;
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

        let data_bytes = &mmap[8..];

        info!(
            "✅ Loaded binary index: dims={} rows={} ANN={} path={:?}",
            dims, rows, ann, path.as_ref()
        );

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, dims * rows)
        };
        let vectors = Arc::from(data);

        let ann_index = if ann {
            let ann_path = path.as_ref().with_extension("ann");
            if ann_path.exists() {
                match AnnIndex::load(&ann_path, &vectors) {
                    Ok(index) => Some(Arc::new(index)),
                    Err(e) => {
                        warn!("⚠️ Failed to load ANN index from {:?}: {}", ann_path, e);
                        None
                    }
                }
            } else {
                warn!("⚠️ ANN requested but file {:?} not found", ann_path);
                None
            }
        } else {
            None
        };

        Ok(Self {
            vectors,
            dims,
            rows,
            ann,
            ann_index,
        })
    }

    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();

        let vector_bytes_len = 4 * self.dims * self.rows;
        let total_len = 8 + vector_bytes_len;

        let file = File::create(path)?;
        file.set_len(total_len as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        mmap[0..4].copy_from_slice(&(self.dims as u32).to_le_bytes());
        mmap[4..8].copy_from_slice(&(self.rows as u32).to_le_bytes());

        let src = self.vectors.as_ref();
        let dst = &mut mmap[8..];

        unsafe {
            let src_bytes = std::slice::from_raw_parts(
                src.as_ptr() as *const u8,
                vector_bytes_len,
            );
            dst.copy_from_slice(src_bytes);
        }

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

        self.top_k_query(query, k)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    // Main query methods with similarity metric support
    pub fn top_k_query(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    pub fn top_k_query_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        // Use auto-selection logic 
        if self.dims >= 64 && self.rows >= 1000 && matches!(similarity, SimilarityMetric::Cosine | SimilarityMetric::DotProduct) {
            self.top_k_query_simd_impl_with_similarity(query, k, similarity)
        } else {
            self.top_k_query_scalar_with_similarity(query, k, similarity)
        }
    }

    // Scalar implementation
    pub fn top_k_query_scalar(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_scalar_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    pub fn top_k_query_scalar_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }

        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(query),
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
                let score = compute_similarity(query, vec_i, similarity);
                (i, score)
            })
            .collect();

        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        Ok(results)
    }

    // SIMD implementation
    pub fn top_k_query_simd(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_simd_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    pub fn top_k_query_simd_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims,
                query.len()
            ));
        }
        
        self.top_k_query_simd_impl_with_similarity(query, k, similarity)
    }

    // Internal SIMD implementation
    fn top_k_query_simd_impl_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => index.query_candidates(query),
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
                
                let score = match similarity {
                    SimilarityMetric::DotProduct => {
                        // Use SIMD for dot product
                        crate::query::compute_score_simd(query, vec_i)
                    },
                    SimilarityMetric::Cosine => {
                        // Use proper cosine similarity calculation
                        compute_similarity(query, vec_i, similarity)
                    },
                    SimilarityMetric::Euclidean => {
                        // Fall back to scalar for euclidean
                        compute_similarity(query, vec_i, similarity)
                    }
                };
                
                (i, score)
            })
            .collect();

        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        Ok(results)
    }

    pub fn top_k_subset(
        &self,
        query: &[f32],
        subset: &[usize],
        k: usize,
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
            "Top-k subset search → k={} subset_len={}",
            k,
            subset.len()
        );

        let mut results: Vec<(usize, f32)> = subset
            .par_iter()
            .filter_map(|&i| match self.get_vector(i) {
                Some(vec_i) => {
                    let score = compute_similarity(query, vec_i, &SimilarityMetric::Cosine);
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