use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use crate::ann_opt::AnnIndex;
use crate::utils::vector::{compute_similarity, SimilarityMetric};
use log::{debug, error, info, warn};

// ========== CONSTANTES DE SEGURANÇA ==========
const MAX_DIMENSIONS: usize = 10000;
const MAX_VECTORS: usize = 100_000_000;
const MIN_FILE_SIZE: usize = 16; // 8 bytes header + 8 bytes minimum data
const MAX_TOP_K: usize = 100000;
const MEMORY_WARNING_THRESHOLD: usize = 1024 * 1024 * 1024; // 1GB

// ========== MÉTRICAS DE PERFORMANCE ==========
#[derive(Debug, Default)]
pub struct EngineMetrics {
    query_count: AtomicUsize,
    total_query_time_ms: AtomicUsize,
    ann_queries: AtomicUsize,
    exact_queries: AtomicUsize,
    simd_queries: AtomicUsize,
    scalar_queries: AtomicUsize,
}

impl EngineMetrics {
    pub fn record_query(&self, duration_ms: u64, used_ann: bool, used_simd: bool) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_query_time_ms.fetch_add(duration_ms as usize, Ordering::Relaxed);
        
        if used_ann {
            self.ann_queries.fetch_add(1, Ordering::Relaxed);
        } else {
            self.exact_queries.fetch_add(1, Ordering::Relaxed);
        }
        
        if used_simd {
            self.simd_queries.fetch_add(1, Ordering::Relaxed);
        } else {
            self.scalar_queries.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_stats(&self) -> (usize, f64, usize, usize, usize, usize) {
        let queries = self.query_count.load(Ordering::Relaxed);
        let total_time = self.total_query_time_ms.load(Ordering::Relaxed);
        let avg_time = if queries > 0 { total_time as f64 / queries as f64 } else { 0.0 };
        let ann = self.ann_queries.load(Ordering::Relaxed);
        let exact = self.exact_queries.load(Ordering::Relaxed);
        let simd = self.simd_queries.load(Ordering::Relaxed);
        let scalar = self.scalar_queries.load(Ordering::Relaxed);
        
        (queries, avg_time, ann, exact, simd, scalar)
    }
}

// ========== ENGINE PRINCIPAL MELHORADO ==========
#[derive(Clone)]
pub struct Engine {
    pub vectors: Arc<[f32]>,
    pub dims: usize,
    pub rows: usize,
    pub ann: bool,
    pub ann_index: Option<Arc<AnnIndex>>,
    
    // NOVOS CAMPOS PARA SEGURANÇA E MÉTRICAS
    creation_time: Instant,
    metrics: Arc<EngineMetrics>,
    file_path: Option<String>, // Para debugging e logs
}

// IMPORTANTE: Garantir thread safety
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Carregar engine de arquivo binário com validações extensivas
    pub fn from_bin<P: AsRef<Path>>(path: P, ann: bool) -> std::io::Result<Self> {
        let start_time = Instant::now();
        let path_ref = path.as_ref();
        
        // Verificações iniciais do arquivo
        if !path_ref.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Binary file not found: {:?}", path_ref),
            ));
        }

        if !path_ref.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Path is not a file: {:?}", path_ref),
            ));
        }

        let file = File::open(&path)?;
        let file_size = file.metadata()?.len() as usize;
        
        // Verificações de tamanho do arquivo
        if file_size < MIN_FILE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Binary file too small: {} bytes (minimum: {} bytes)", file_size, MIN_FILE_SIZE),
            ));
        }

        if file_size > MEMORY_WARNING_THRESHOLD {
            warn!("Large binary file: {:.1}GB", file_size as f64 / (1024.0_f64.powi(3)));
        }

        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary file header too small",
            ));
        }

        // Ler e validar header
        let dims = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let rows = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

        // Validações de dimensões
        if dims == 0 || dims > MAX_DIMENSIONS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid dimensions: {} (max: {})", dims, MAX_DIMENSIONS),
            ));
        }

        if rows == 0 || rows > MAX_VECTORS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid row count: {} (max: {})", rows, MAX_VECTORS),
            ));
        }

        // Verificar consistência do tamanho do arquivo
        let vector_bytes_len = 4 * dims * rows; // f32 = 4 bytes
        let expected_len = 8 + vector_bytes_len;
        
        if mmap.len() != expected_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Binary file size mismatch: expected {} bytes, found {} bytes (dims={}, rows={})",
                    expected_len, mmap.len(), dims, rows
                ),
            ));
        }

        // Verificar se há memória suficiente
        let estimated_memory = vector_bytes_len;
        if estimated_memory > MEMORY_WARNING_THRESHOLD {
            warn!(
                "Loading large dataset: {:.1}GB memory usage ({}x{} vectors)",
                estimated_memory as f64 / (1024.0_f64.powi(3)),
                rows, dims
            );
        }

        let data_bytes = &mmap[8..];

        // Conversão segura para slice de f32
        if data_bytes.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Vector data length not aligned to f32 boundary",
            ));
        }

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                data_bytes.as_ptr() as *const f32, 
                data_bytes.len() / 4
            )
        };

        if data.len() != dims * rows {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Vector data length mismatch: expected {}, got {}",
                    dims * rows, data.len()
                ),
            ));
        }

        // Validação estatística dos dados (amostra)
        let sample_size = (dims * rows).min(10000); // Verificar até 10k elementos
        let sample = &data[..sample_size];
        
        let finite_count = sample.iter().filter(|&&x| x.is_finite()).count();
        if finite_count < sample_size {
            let invalid_count = sample_size - finite_count;
            warn!(
                "Vector data contains {} invalid values out of {} sampled",
                invalid_count, sample_size
            );
            
            // Se mais de 1% dos valores são inválidos, rejeitar
            if invalid_count > sample_size / 100 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Too many invalid values in vector data: {}/{}", invalid_count, sample_size),
                ));
            }
        }

        let vectors = Arc::from(data);

        info!(
            "✅ Loaded binary index: dims={} rows={} ANN={} path={:?} size={:.1}MB",
            dims, rows, ann, path_ref,
            mmap.len() as f64 / (1024.0 * 1024.0)
        );

        // Carregar índice ANN se solicitado
        let ann_index = if ann {
            let ann_path = path_ref.with_extension("ann");
            if ann_path.exists() {
                match AnnIndex::load(&ann_path, &vectors) {
                    Ok(index) => {
                        // Verificar health do índice ANN
                        if let Err(e) = index.health_check() {
                            warn!("⚠️ ANN index health check failed: {}", e);
                            None
                        } else {
                            info!("✅ ANN index loaded and verified");
                            Some(Arc::new(index))
                        }
                    }
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

        let engine = Self {
            vectors,
            dims,
            rows,
            ann,
            ann_index,
            creation_time: start_time,
            metrics: Arc::new(EngineMetrics::default()),
            file_path: Some(path_ref.to_string_lossy().to_string()),
        };

        let load_time = start_time.elapsed();
        info!("Engine loaded in {:.2}s", load_time.as_secs_f64());

        Ok(engine)
    }

    /// Salvar engine para arquivo binário com validações
    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        
        // Validações pré-save
        if self.dims == 0 || self.rows == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Cannot save engine with zero dimensions or rows",
            ));
        }

        if self.vectors.len() != self.dims * self.rows {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Vector data inconsistency: expected {} elements, have {}",
                    self.dims * self.rows, self.vectors.len()
                ),
            ));
        }

        let data_bytes_len = self.vectors.len() * 4; // f32 = 4 bytes
        let total_len = 8 + data_bytes_len; // header + data

        info!("Saving engine to {:?} ({:.1}MB)", path, total_len as f64 / (1024.0 * 1024.0));

        let file = File::create(path)?;
        file.set_len(total_len as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Escrever header
        mmap[0..4].copy_from_slice(&(self.dims as u32).to_le_bytes());
        mmap[4..8].copy_from_slice(&(self.rows as u32).to_le_bytes());

        // Escrever dados vetoriais
        let src = self.vectors.as_ref();
        let dst = &mut mmap[8..];

        unsafe {
            let src_bytes = std::slice::from_raw_parts(
                src.as_ptr() as *const u8,
                data_bytes_len,
            );
            dst.copy_from_slice(src_bytes);
        }

        mmap.flush()?;

        info!(
            "✅ Saved engine to binary: {:?} dims={} rows={} size={} bytes",
            path, self.dims, self.rows, total_len
        );

        Ok(())
    }

    /// Obter vetor por índice com validações
    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows {
            warn!("Index out of bounds: {} (rows={})", idx, self.rows);
            return None;
        }
        
        let start = idx * self.dims;
        let end = start + self.dims;
        
        if end > self.vectors.len() {
            error!("Vector slice out of bounds: {}..{} (len={})", start, end, self.vectors.len());
            return None;
        }
        
        let slice = &self.vectors[start..end];
        
        // Verificação de sanidade do vetor
        if slice.iter().any(|&x| !x.is_finite()) {
            warn!("Vector at index {} contains invalid values", idx);
        }
        
        Some(slice)
    }

    /// Query por índice com validações
    pub fn top_k_index(&self, idx: usize, k: usize) -> std::io::Result<Vec<(usize, f32)>> {
        debug!("Top-k index search → idx={} k={}", idx, k);

        // Validações
        if k == 0 {
            return Ok(Vec::new());
        }

        if k > MAX_TOP_K {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("top_k too large: {} (max: {})", k, MAX_TOP_K),
            ));
        }

        let query = self.get_vector(idx).ok_or_else(|| {
            warn!("Invalid index in top_k_index: {}", idx);
            std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Invalid index: {}", idx))
        })?;

        self.top_k_query(query, k)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Query principal com seleção automática de método
    pub fn top_k_query(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    /// Query com métrica de similaridade específica
    pub fn top_k_query_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        let start_time = Instant::now();

        // Validações de entrada rigorosas
        if query.len() != self.dims {
            return Err(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dims, query.len()
            ));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        if k > self.rows {
            return Err(format!(
                "top_k ({}) cannot be greater than number of vectors ({})",
                k, self.rows
            ));
        }

        if k > MAX_TOP_K {
            return Err(format!(
                "top_k too large: {} (max: {})",
                k, MAX_TOP_K
            ));
        }

        // Validar query vector
        if query.iter().any(|&x| !x.is_finite()) {
            return Err("Query vector contains NaN or infinite values".to_string());
        }

        let query_norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Err("Query vector has zero norm".to_string());
        }

        // Seleção automática de método baseada em características
        let use_simd = self.should_use_simd(similarity);
        let used_ann = self.ann && self.ann_index.is_some();
        
        debug!("Query: dims={}, k={}, method={}, ann={}, similarity={:?}", 
               self.dims, k, if use_simd { "simd" } else { "scalar" }, used_ann, similarity);

        let result = if use_simd {
            self.top_k_query_simd_impl_with_similarity(query, k, similarity)
        } else {
            self.top_k_query_scalar_with_similarity(query, k, similarity)
        };

        // Registrar métricas
        let duration = start_time.elapsed();
        self.metrics.record_query(duration.as_millis() as u64, used_ann, use_simd);

        result
    }

    /// Determinar se deve usar SIMD baseado em características
    fn should_use_simd(&self, similarity: &SimilarityMetric) -> bool {
        // SIMD é mais eficaz para:
        // - Vetores com 64+ dimensões
        // - Datasets com 1000+ vetores
        // - Métricas que se beneficiam de SIMD (dot product, cosine)
        self.dims >= 64 && 
        self.rows >= 1000 && 
        matches!(similarity, SimilarityMetric::Cosine | SimilarityMetric::DotProduct)
    }

    /// Implementação scalar com validações
    pub fn top_k_query_scalar(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_scalar_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    pub fn top_k_query_scalar_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims, query.len()
            ));
        }

        // Obter candidatos (ANN ou todos)
        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => {
                    let candidates = index.query_candidates(query);
                    if candidates.is_empty() {
                        warn!("ANN returned no candidates, falling back to full search");
                        (0..self.rows).collect()
                    } else {
                        candidates
                    }
                },
                None => {
                    warn!("ANN enabled but no index available, using full search");
                    (0..self.rows).collect()
                }
            }
        } else {
            (0..self.rows).collect()
        };

        debug!("Scalar search with {} candidates", candidates.len());

        // Computar similaridades em paralelo
        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .filter_map(|i| {
                if i >= self.rows {
                    warn!("Invalid candidate index: {} >= {}", i, self.rows);
                    return None;
                }
                
                let offset = i * self.dims;
                if offset + self.dims > self.vectors.len() {
                    warn!("Vector offset out of bounds: {} + {} > {}", offset, self.dims, self.vectors.len());
                    return None;
                }
                
                let vec_i = &self.vectors[offset..offset + self.dims];
                
                // Verificar validade do vetor
                if vec_i.iter().any(|&x| !x.is_finite()) {
                    warn!("Invalid vector at index {}", i);
                    return None;
                }
                
                let score = compute_similarity(query, vec_i, similarity);
                
                if !score.is_finite() {
                    warn!("Invalid similarity score for index {}: {}", i, score);
                    return None;
                }
                
                Some((i, score))
            })
            .collect();

        // Ordenação otimizada e segura
        if results.is_empty() {
            warn!("No valid results found");
            return Ok(Vec::new());
        }

        if results.len() > k {
            // Usar select_nth para eficiência
            results.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        
        debug!("Scalar search completed: {} results", results.len());
        Ok(results)
    }

    /// Implementação SIMD com validações
    pub fn top_k_query_simd(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        self.top_k_query_simd_with_similarity(query, k, &SimilarityMetric::Cosine)
    }

    pub fn top_k_query_simd_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "Query vector has wrong dimension: expected {}, got {}",
                self.dims, query.len()
            ));
        }
        
        self.top_k_query_simd_impl_with_similarity(query, k, similarity)
    }

    /// Implementação interna SIMD
    fn top_k_query_simd_impl_with_similarity(&self, query: &[f32], k: usize, similarity: &SimilarityMetric) -> Result<Vec<(usize, f32)>, String> {
        // Obter candidatos
        let candidates: Vec<usize> = if self.ann {
            match &self.ann_index {
                Some(index) => {
                    let candidates = index.query_candidates(query);
                    if candidates.is_empty() {
                        warn!("ANN returned no candidates, falling back to full search");
                        (0..self.rows).collect()
                    } else {
                        candidates
                    }
                },
                None => (0..self.rows).collect(),
            }
        } else {
            (0..self.rows).collect()
        };

        debug!("SIMD search with {} candidates", candidates.len());

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .filter_map(|i| {
                if i >= self.rows {
                    return None;
                }
                
                let offset = i * self.dims;
                if offset + self.dims > self.vectors.len() {
                    return None;
                }
                
                let vec_i = &self.vectors[offset..offset + self.dims];
                
                let score = match similarity {
                    SimilarityMetric::DotProduct => {
                        // Usar SIMD para dot product
                        crate::query::compute_score_simd(query, vec_i)
                    },
                    SimilarityMetric::Cosine => {
                        // Para cosine, usar implementação escalar por enquanto
                        // TODO: Implementar cosine SIMD otimizada
                        compute_similarity(query, vec_i, similarity)
                    },
                    SimilarityMetric::Euclidean => {
                        // Euclidean usa implementação escalar
                        compute_similarity(query, vec_i, similarity)
                    }
                };
                
                if !score.is_finite() {
                    return None;
                }
                
                Some((i, score))
            })
            .collect();

        if results.is_empty() {
            return Ok(Vec::new());
        }

        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        debug!("SIMD search completed: {} results", results.len());
        Ok(results)
    }

    /// Query em subset específico com validações
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

        if k == 0 {
            return Ok(Vec::new());
        }

        if k > MAX_TOP_K {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("top_k too large: {} (max: {})", k, MAX_TOP_K),
            ));
        }

        if subset.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Top-k subset search → k={} subset_len={}", k, subset.len());

        // Validar subset indices
        for &idx in subset.iter().take(100) { // Verificar primeiros 100 para performance
            if idx >= self.rows {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Subset index {} out of bounds (max: {})", idx, self.rows - 1),
                ));
            }
        }

        let mut results: Vec<(usize, f32)> = subset
            .par_iter()
            .filter_map(|&i| {
                if i >= self.rows {
                    warn!("Index {} out of bounds in subset", i);
                    return None;
                }
                
                match self.get_vector(i) {
                    Some(vec_i) => {
                        let score = compute_similarity(query, vec_i, &SimilarityMetric::Cosine);
                        if score.is_finite() {
                            Some((i, score))
                        } else {
                            warn!("Invalid similarity score for subset index {}", i);
                            None
                        }
                    }
                    None => {
                        warn!("Failed to get vector at subset index {}", i);
                        None
                    }
                }
            })
            .collect();

        if results.is_empty() {
            debug!("No valid results in subset");
            return Ok(Vec::new());
        }

        results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        debug!("Subset search complete → returning top {} results", results.len());
        Ok(results)
    }

    /// Health check do engine
    pub fn health_check(&self) -> Result<(), String> {
        // Verificar consistência básica
        if self.dims == 0 {
            return Err("Invalid dimensions: 0".to_string());
        }
        
        if self.rows == 0 {
            return Err("Invalid row count: 0".to_string());
        }
        
        if self.vectors.len() != self.dims * self.rows {
            return Err(format!(
                "Vector data size mismatch: expected {}, got {}",
                self.dims * self.rows, self.vectors.len()
            ));
        }

        // Verificar alguns vetores aleatoriamente
        let sample_indices = [0, self.rows / 4, self.rows / 2, 3 * self.rows / 4, self.rows - 1];
        for &idx in &sample_indices {
            if idx < self.rows {
                if let Some(vec) = self.get_vector(idx) {
                    if vec.iter().any(|&x| !x.is_finite()) {
                        return Err(format!("Vector at index {} contains invalid values", idx));
                    }
                } else {
                    return Err(format!("Failed to retrieve vector at index {}", idx));
                }
            }
        }

        // Verificar ANN index se presente
        if let Some(ann) = &self.ann_index {
            if let Err(e) = ann.health_check() {
                return Err(format!("ANN index health check failed: {}", e));
            }
        }

        info!("Engine health check passed: {} vectors × {} dims", self.rows, self.dims);
        Ok(())
    }

    /// Obter estatísticas do engine
    pub fn get_stats(&self) -> (usize, f64, usize, usize, usize, usize, f64) {
        let (queries, avg_time, ann, exact, simd, scalar) = self.metrics.get_stats();
        let uptime = self.creation_time.elapsed().as_secs_f64();
        (queries, avg_time, ann, exact, simd, scalar, uptime)
    }

    /// Propriedades básicas
    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn file_path(&self) -> Option<&str> {
        self.file_path.as_deref()
    }

    pub fn has_ann(&self) -> bool {
        self.ann_index.is_some()
    }

    pub fn memory_usage_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }
}

// ========== IMPLEMENTAÇÃO DE DROP PARA CLEANUP ==========
impl Drop for Engine {
    fn drop(&mut self) {
        let (queries, avg_time, _, _, _, _, uptime) = self.get_stats();
        info!(
            "Engine dropped: {} vectors, {} queries processed, {:.2}ms avg query time, {:.1}s uptime",
            self.rows, queries, avg_time, uptime
        );
    }
}