use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use sha2::{Sha256, Digest};  
use std::io::Result;

use fast_float::parse as fast_parse;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use bytemuck;   
use crate::ann_opt::AnnIndex;
use crate::utils::vector::{cosine_similarity, normalize_vector_inplace, validate_dimensions};
use log::{debug, error, info, warn};

//use crate::query::EngineTopKExt;


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
        let hash_stored = &mmap[8 + vector_bytes_len..];

        let mut hasher = sha2::Sha256::new();
        hasher.update(data_bytes);
        let hash_computed = hasher.finalize();

        if hash_stored != hash_computed.as_slice() {
            error!("❌ Hash mismatch: binary data may be corrupted");
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Hash mismatch: binary data may be corrupted",
            ));
        }

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


    pub fn from_csv_parallel<P: AsRef<Path> + Clone>(
        path: P,
        normalize: bool,
        use_ann: bool,
    ) -> std::io::Result<Self> {
        let file = File::open(path.clone())?;
        let mut reader = BufReader::new(file);

        let mut contents = Vec::new();
        reader.read_to_end(&mut contents)?;

        let raw = String::from_utf8_lossy(&contents);
        let lines: Vec<&str> = raw.lines().collect();

        if lines.is_empty() {
            error!("CSV file is empty → path={:?}", path.as_ref());
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "CSV file is empty",
            ));
        }

        let first_line = lines[0];
        let dims = first_line.split(',').count();
        info!("Parsing CSV with {} dimensions", dims);

        let nested: Vec<Vec<f32>> = lines
            .par_chunks(10000)
            .flat_map_iter(|chunk| {
                chunk.iter().filter_map(|line| {
                    let mut row = Vec::with_capacity(dims);
                    for s in line.split(',') {
                        match fast_parse::<f32, _>(s.trim()) {
                            Ok(v) => row.push(v),
                            Err(_) => return None,
                        }
                    }
                    if row.len() != dims {
                        return None;
                    }
                    if normalize {
                        normalize_vector_inplace(&mut row);
                    }
                    Some(row)
                })
            })
            .collect();

        let flat: Vec<f32> = nested.into_par_iter().flatten().collect();
        let arc_vectors: Arc<[f32]> = Arc::from(flat);
        let rows = arc_vectors.len() / dims;

        info!(
            "Loaded CSV with {} rows and {} dims → normalize={} ANN={} path={:?}",
            rows,
            dims,
            normalize,
            use_ann,
            path.as_ref()
        );

        let ann_index = if use_ann {
            Some(AnnIndex::build(&arc_vectors, dims, rows, 32, 42))
        } else {
            None
        };

        Ok(Self {
            vectors: arc_vectors,
            dims,
            rows,
            use_ann,
            ann_index,
        })
    }

    pub fn from_embeddings(
        vectors: Vec<Vec<f32>>,
        normalize: bool,
        use_ann: bool,
    ) -> std::io::Result<Self> {
        if vectors.is_empty() {
            error!("Received empty embeddings list");
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Empty embeddings list",
            ));
        }

        let dims = validate_dimensions(&vectors)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let rows = vectors.len();
        let mut flat = vec![0.0f32; dims * rows];

        flat.par_chunks_mut(dims)
            .zip(vectors.into_par_iter())
            .for_each(|(chunk, mut vec)| {
                if normalize {
                    normalize_vector_inplace(&mut vec);
                }
                chunk.copy_from_slice(&vec);
            });

        info!(
            "Engine from memory embeddings: rows={} dims={} normalize={} ANN={} ",
            rows, dims, normalize, use_ann
        );

        let arc_vectors: Arc<[f32]> = Arc::from(flat);

        let ann_index = if use_ann {
            Some(AnnIndex::build(&arc_vectors, dims, rows, 32, 42))
        } else {
            None
        };

        Ok(Self {
            vectors: arc_vectors,
            dims,
            rows,
            use_ann,
            ann_index,
        })
    }

    pub fn from_embeddings_custom(
        vectors: Vec<Vec<f32>>,
        normalize: bool,
        use_ann: bool,
        bits: Option<usize>,
        seed: Option<u64>,
    ) -> std::io::Result<Self> {
        if vectors.is_empty() {
            error!("Received empty embeddings list");
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Empty embeddings list",
            ));
        }

        let dims = validate_dimensions(&vectors)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let rows = vectors.len();
        let mut flat = vec![0.0f32; dims * rows];

        flat.par_chunks_mut(dims)
            .zip(vectors.into_par_iter())
            .for_each(|(chunk, mut vec)| {
                if normalize {
                    normalize_vector_inplace(&mut vec);
                }
                chunk.copy_from_slice(&vec);
            });

        info!("Engine from memory embeddings with custom ANN: rows={} dims={} normalize={} ANN={} bits={:?} seed={:?}", rows, dims, normalize, use_ann, bits, seed);

        let arc_vectors: Arc<[f32]> = Arc::from(flat);

        let ann_index = if use_ann {
            Some(AnnIndex::build(
                &arc_vectors,
                dims,
                rows,
                bits.expect("bits must be Some"),
                seed.expect("seed must be Some"),
            ))
        } else {
            None
        };

        Ok(Self {
            vectors: arc_vectors,
            dims,
            rows,
            use_ann,
            ann_index,
        })
    }

    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> Result<()> {
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

        let mut hasher = Sha256::new();
        hasher.update(&data_bytes);
        let hash = hasher.finalize();

        mmap[8 + data_bytes.len()..].copy_from_slice(&hash);
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

    //pub fn top_k_query(
    // &self,
    // query: &[f32],
    // k: usize,
    // normalize: bool,
    //) -> Result<Vec<(usize, f32)>, String> {
      //if query.len() != self.dims {
           // warn!(
           //     "Received query with wrong dimension: expected={}, got={}",
           //     self.dims,
           //     query.len()
           // );
           // return Err(format!(
           //     "Query vector has wrong dimension: expected {}, got {}",
           //     self.dims,
           //     query.len()
           // ));
      //}

      //debug!(
      //    "Running top-k query → dims={} k={} normalize={} ann={}",
      //    self.dims, k, normalize, self.use_ann
      //);

      //let start_time = Instant::now();

      //let mut query = query.to_vec();
      //if normalize {
      //    normalize_vector_inplace(&mut query);
      //}

      //let candidates: Vec<usize> = if self.use_ann {
      //    match &self.ann_index {
      //        Some(index) => {
      //            let c = index.query_candidates(&query);
      //            debug!("ANN returned {} candidates", c.len());
      //            c
      //        }
      //        None => {
      //            warn!("ANN is enabled but index is missing → using full search");
      //            (0..self.rows).collect()
      //        }
      //    }
      //} else {
      //    (0..self.rows).collect()
      //};

      //let mut results: Vec<(usize, f32)> = candidates
      //    .into_par_iter()
      //    .map(|i| {
      //        let vec_i = &self.vectors[i * self.dims..(i + 1) * self.dims];
      //        let score = cosine_similarity(&query, vec_i);
      //        (i, score)
      //    })
      //    .collect();

      //results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
      //results.truncate(k);
      //let elapsed = start_time.elapsed();
      //info!(
      //    "Top-k query completed → results={} time={:.2?}",
      //    results.len(),
      //    elapsed
      //);

      //Ok(results)
    //}

    pub fn top_k_index(&self, idx: usize, k: usize) -> std::result::Result<Vec<(usize, f32)>, String> {
        debug!("Top-k index search → idx={} k={}", idx, k);

        let query = self.get_vector(idx).ok_or_else(|| {
            warn!("Invalid index in top_k_index: {}", idx);
            std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Invalid index: {}", idx))
        })?;

        self.top_k_query(query, k, false)
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
