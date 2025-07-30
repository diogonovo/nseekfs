use crate::io::{load_csv_f32_flat_parallel, write_bin_file, load_bin_mmap};
use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

pub struct SearchEngine {
    pub data: Vec<f32>,
    pub dims: usize,
    pub rows: usize,
}

impl SearchEngine {
    /// Cria o binário a partir de CSV (opcional normalização)
    pub fn prepare(csv_path: &str, normalize: bool, force: bool) -> Result<String, Box<dyn Error>> {
        let bin_path = csv_path.replace(".csv", ".bin");
        let csv_mtime = fs::metadata(csv_path)?.modified()?;
        let bin_exists = Path::new(&bin_path).exists();
        let reconvert = if bin_exists {
            let bin_mtime = fs::metadata(&bin_path)?.modified()?;
            csv_mtime > bin_mtime || force
        } else {
            true
        };

        if reconvert {
            println!("⚠️  A converter CSV para formato binário rápido...");
            let (mut data, rows, dims) = load_csv_f32_flat_parallel(csv_path)?;
            if normalize {
                normalize_vectors(&mut data, dims);
            }
            write_bin_file(&bin_path, &data, rows, dims)?;
            println!("✅ Conversão concluída: {}", bin_path);
        } else {
            println!("✅ Binário já atualizado: {}", bin_path);
        }
        Ok(bin_path)
    }

    /// Carrega dataset (CSV ou BIN), reconverte se necessário
    pub fn load_auto(path: &str) -> Result<Self, Box<dyn Error>> {
        let bin_path = if path.ends_with(".csv") {
            Self::prepare(path, false, false)?
        } else {
            path.to_string()
        };
        let (data, rows, dims) = load_bin_mmap(&bin_path)?;
        Ok(SearchEngine { data, rows, dims })
    }

    /// Pesquisa por índice
    pub fn top_k_similar(&self, index: usize, k: usize) -> Vec<(usize, f32)> {
        let input_start = index * self.dims;
        let input = &self.data[input_start..input_start + self.dims];
        self.compute_top_k(input, k)
    }

    /// Pesquisa por vetor externo
    pub fn top_k_query(&self, vector: &[f32], k: usize, normalize: bool) -> Result<Vec<(usize, f32)>, String> {
        if vector.len() != self.dims {
            return Err(format!(
                "❌ Dimensão incorreta: query={} dataset={}",
                vector.len(),
                self.dims
            ));
        }
        let mut query = vector.to_vec();
        if normalize {
            normalize_vector(&mut query);
        }
        Ok(self.compute_top_k(&query, k))
    }

    fn compute_top_k(&self, input: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut sims: Vec<(usize, f32)> = (0..self.rows)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dims;
                let end = start + self.dims;
                let sim = cosine_similarity(input, &self.data[start..end]);
                (i, sim)
            })
            .collect();

        if sims.len() > k {
            sims.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            sims.truncate(k);
        }
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        sims
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

fn normalize_vectors(data: &mut [f32], dims: usize) {
    data.chunks_mut(dims).for_each(|vec| normalize_vector(vec));
}

fn normalize_vector(vec: &mut [f32]) {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-8;
    vec.iter_mut().for_each(|x| *x /= norm);
}
