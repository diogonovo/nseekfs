use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use fast_float::parse as fast_parse;

use crate::utils::{normalize_vector_inplace, cosine_similarity};
use crate::ann::AnnIndex;

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
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Ficheiro binário muito pequeno"));
        }

        let dims = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let rows = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        let expected_len = 8 + 4 * dims * rows;
        if mmap.len() != expected_len {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Tamanho inesperado no ficheiro binário"));
        }

        let data: &[f32] = bytemuck::cast_slice(&mmap[8..]);
        let vectors = Arc::from(data);

        let ann_index = if use_ann {
            Some(AnnIndex::build(&vectors, dims, rows, 16, 42))
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

    pub fn from_csv_parallel<P: AsRef<Path>>(path: P, normalize: bool, use_ann: bool) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut contents = Vec::new();
        reader.read_to_end(&mut contents)?;

        let raw = String::from_utf8_lossy(&contents);
        let lines: Vec<&str> = raw.lines().collect();

        if lines.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "CSV está vazio"));
        }

        let first_line = lines[0];
        let dims = first_line.split(',').count();

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

        let flat: Vec<f32> = nested.into_iter().flatten().collect();
        let arc_vectors: Arc<[f32]> = Arc::from(flat);
        let rows = arc_vectors.len() / dims;

        let ann_index = if use_ann {
            Some(AnnIndex::build(&arc_vectors, dims, rows, 16, 42))
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

    /// Novo método: cria diretamente de Vec<Vec<f32>> (ex: HuggingFace ou OpenAI)
    pub fn from_embeddings(vectors: Vec<Vec<f32>>, normalize: bool, use_ann: bool) -> std::io::Result<Self> {
        if vectors.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Lista de embeddings vazia"));
        }

        let dims = vectors[0].len();
        if vectors.iter().any(|v| v.len() != dims) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Vetores com dimensões inconsistentes"));
        }

        let mut flat: Vec<f32> = Vec::with_capacity(vectors.len() * dims);
        for mut vec in vectors {
            if normalize {
                normalize_vector_inplace(&mut vec);
            }
            flat.extend(vec);
        }

        let arc_vectors: Arc<[f32]> = Arc::from(flat);
        let rows = arc_vectors.len() / dims;

        let ann_index = if use_ann {
            Some(AnnIndex::build(&arc_vectors, dims, rows, 16, 42))
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

    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(&path)?;
        let total_bytes = 8 + 4 * self.dims * self.rows;
        file.set_len(total_bytes as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        mmap[0..4].copy_from_slice(&(self.dims as u32).to_le_bytes());
        mmap[4..8].copy_from_slice(&(self.rows as u32).to_le_bytes());

        let byte_slice: &[u8] = bytemuck::cast_slice(&self.vectors);
        mmap[8..].copy_from_slice(byte_slice);
        mmap.flush()?;

        Ok(())
    }

    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows {
            return None;
        }
        let start = idx * self.dims;
        let end = start + self.dims;
        Some(&self.vectors[start..end])
    }

    pub fn top_k_query(&self, query: &[f32], k: usize, normalize: bool) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dims);

        let mut query = query.to_vec();
        if normalize {
            normalize_vector_inplace(&mut query);
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
                let vec_i = &self.vectors[i * self.dims..(i + 1) * self.dims];
                let score = cosine_similarity(&query, vec_i);
                (i, score)
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);
        results
    }

    pub fn top_k_index(&self, idx: usize, k: usize) -> Option<Vec<(usize, f32)>> {
        let query = self.get_vector(idx)?;
        Some(self.top_k_query(query, k, false))
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}
