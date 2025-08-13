use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, wrap_pyfunction};
use std::path::Path;
use numpy::PyReadonlyArray2;
use log::{info, warn, error};

mod utils;
mod prebin;
mod ann_opt;
mod query;
mod engine;

use prebin::prepare_bin_from_embeddings;
use engine::Engine;
use crate::utils::vector::SimilarityMetric;

// ========== CONSTANTES DE SEGURANÇA ==========
const MAX_EMBEDDINGS_SIZE: usize = 100_000_000; // 100M vectors max
const MAX_DIMENSIONS: usize = 10000;
const MIN_DIMENSIONS: usize = 8;

/// Preparar binário a partir de embeddings com validações extensivas
#[pyfunction]
pub fn py_prepare_bin_from_embeddings(
    embeddings: PyReadonlyArray2<f32>,  
    base_name: String,
    level: &str,
    ann: bool,
    normalize: bool,
    seed: u64,
    output_dir: Option<String>,
) -> PyResult<String> {
    info!("Starting py_prepare_bin_from_embeddings");
    
    // Validações de entrada
    let shape = embeddings.shape();
    if shape.len() != 2 {
        error!("Input array is not 2D: {} dimensions", shape.len());
        return Err(PyValueError::new_err("Input must be a 2D NumPy array"));
    }
    
    let rows = shape[0];
    let dims = shape[1];
    
    // Validações de limites de segurança
    if rows == 0 {
        return Err(PyValueError::new_err("Cannot create index with zero vectors"));
    }
    
    if rows > MAX_EMBEDDINGS_SIZE {
        return Err(PyValueError::new_err(
            format!("Too many vectors: {} (max: {})", rows, MAX_EMBEDDINGS_SIZE)
        ));
    }
    
    if dims < MIN_DIMENSIONS {
        return Err(PyValueError::new_err(
            format!("Too few dimensions: {} (min: {})", dims, MIN_DIMENSIONS)
        ));
    }
    
    if dims > MAX_DIMENSIONS {
        return Err(PyValueError::new_err(
            format!("Too many dimensions: {} (max: {})", dims, MAX_DIMENSIONS)
        ));
    }
    
    // Validação do level
    match level {
        "f8" | "f16" | "f32" | "f64" => {},
        _ => return Err(PyValueError::new_err(
            format!("Invalid level '{}'. Must be one of: f8, f16, f32, f64", level)
        )),
    }
    
    // Validação do base_name
    if base_name.trim().is_empty() {
        return Err(PyValueError::new_err("base_name cannot be empty"));
    }
    
    if base_name.contains('/') || base_name.contains('\\') {
        return Err(PyValueError::new_err("base_name cannot contain path separators"));
    }
    
    // Validação do seed
    if seed > u32::MAX as u64 {
        warn!("Large seed value: {} (will be truncated)", seed);
    }
    
    info!("Validated input: {} vectors × {} dims, level={}, ann={}", 
          rows, dims, level, ann);

    // Converter para slice 1D contínuo (sem cópias)
    let flat = embeddings.as_slice().map_err(|e| {
        error!("Failed to convert embeddings to slice: {}", e);
        PyValueError::new_err(format!("Failed to access embeddings data: {}", e))
    })?;
    
    // Validação de dados - amostra para performance
    let sample_size = (flat.len()).min(10000); // Verificar até 10k elementos
    let sample = &flat[..sample_size];
    
    let invalid_count = sample.iter().filter(|&&x| !x.is_finite()).count();
    if invalid_count > 0 {
        let percentage = (invalid_count as f64 / sample_size as f64) * 100.0;
        if percentage > 1.0 { // Mais de 1% inválido
            return Err(PyValueError::new_err(
                format!("Too many invalid values in embeddings: {:.1}% ({}/{})", 
                       percentage, invalid_count, sample_size)
            ));
        } else {
            warn!("Found {} invalid values in embeddings sample", invalid_count);
        }
    }

    let output_path_opt = output_dir.as_deref().map(Path::new);

    info!("Calling prepare_bin_from_embeddings");
    
    let result = prepare_bin_from_embeddings(
        flat,
        dims,
        rows,
        &base_name,
        level,
        output_path_opt,
        ann,
        normalize,
        seed,
    );
    
    match result {
        Ok(pathbuf) => {
            let path_str = pathbuf.to_string_lossy().to_string();
            info!("Successfully created binary: {}", path_str);
            Ok(path_str)
        },
        Err(e) => {
            error!("Failed to prepare binary: {}", e);
            Err(PyValueError::new_err(format!("Index creation failed: {}", e)))
        }
    }
}

/// Classe PySearchEngine melhorada com validações
#[pyclass]
struct PySearchEngine {
    engine: Engine,
}

#[pymethods]
impl PySearchEngine {
    #[new]
    fn new(path: &str, ann: Option<bool>) -> PyResult<Self> {
        info!("Creating PySearchEngine from path: {}", path);
        
        // Validações de entrada
        if path.trim().is_empty() {
            return Err(PyValueError::new_err("Path cannot be empty"));
        }
        
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Err(PyValueError::new_err(format!("File not found: {}", path)));
        }
        
        if !path_obj.is_file() {
            return Err(PyValueError::new_err(format!("Path is not a file: {}", path)));
        }
        
        let ann = ann.unwrap_or(true);

        let engine = Engine::from_bin(path, ann)
            .map_err(|e| {
                error!("Failed to load engine from {}: {}", path, e);
                PyValueError::new_err(format!("Failed to load engine: {}", e))
            })?;
        
        // Verificar health do engine carregado
        if let Err(e) = engine.health_check() {
            error!("Engine health check failed: {}", e);
            return Err(PyValueError::new_err(format!("Engine health check failed: {}", e)));
        }

        info!("PySearchEngine created successfully: {}x{} vectors", 
              engine.rows(), engine.dims());

        Ok(Self { engine })
    }

    fn dims(&self) -> usize {
        self.engine.dims()
    }

    fn rows(&self) -> usize {
        self.engine.rows()
    }

    fn get_vector(&self, idx: usize) -> PyResult<Vec<f32>> {
        // Validações
        if idx >= self.engine.rows() {
            return Err(PyValueError::new_err(
                format!("Index {} out of bounds (max: {})", idx, self.engine.rows() - 1)
            ));
        }

        self.engine
            .get_vector(idx)
            .map(|v| v.to_vec())
            .ok_or_else(|| {
                error!("Failed to get vector at index {}", idx);
                PyValueError::new_err(format!("Failed to get vector at index {}", idx))
            })
    }

    fn top_k_query(
        &self,
        query: Vec<f32>,
        k: usize,
        method: Option<String>,
        similarity: Option<String>,  
    ) -> PyResult<Vec<(usize, f32)>> {
        // Validações de entrada
        let method = method.unwrap_or_else(|| "auto".to_string());
        let similarity = similarity.unwrap_or_else(|| "cosine".to_string());

        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err(
                format!("Query vector dimension mismatch: expected {}, got {}", 
                       self.engine.dims(), query.len())
            ));
        }
        
        if k == 0 {
            return Ok(Vec::new());
        }
        
        if k > self.engine.rows() {
            return Err(PyValueError::new_err(
                format!("top_k ({}) cannot exceed number of vectors ({})", k, self.engine.rows())
            ));
        }
        
        if k > 100000 { // Limite de segurança
            return Err(PyValueError::new_err(
                format!("top_k too large: {} (max: 100000)", k)
            ));
        }

        // Validar query vector
        if query.iter().any(|&x| !x.is_finite()) {
            return Err(PyValueError::new_err("Query vector contains NaN or infinite values"));
        }
        
        let query_norm: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Err(PyValueError::new_err("Query vector has zero norm"));
        }

        // Parse similarity metric com validação
        let similarity_metric = SimilarityMetric::from_str(&similarity)
            .map_err(|e| {
                error!("Invalid similarity metric: {}", similarity);
                PyValueError::new_err(e)
            })?;

        // Validar method
        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar_with_similarity(&query, k, &similarity_metric),
            "simd" => self.engine.top_k_query_simd_with_similarity(&query, k, &similarity_metric),
            "auto" => self.engine.top_k_query_with_similarity(&query, k, &similarity_metric),
            other => {
                error!("Invalid method: {}", other);
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.", other
                )));
            }
        };

        result.map_err(|e| {
            error!("Query failed: {}", e);
            PyValueError::new_err(format!("Query failed: {}", e))
        })
    }

    fn top_k_similar(
        &self,
        idx: usize,
        k: usize,
        method: Option<String>,
        similarity: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        // Validações
        if idx >= self.engine.rows() {
            return Err(PyValueError::new_err(
                format!("Index {} out of bounds (max: {})", idx, self.engine.rows() - 1)
            ));
        }
        
        if k == 0 {
            return Ok(Vec::new());
        }
        
        if k > self.engine.rows() {
            return Err(PyValueError::new_err(
                format!("top_k ({}) cannot exceed number of vectors ({})", k, self.engine.rows())
            ));
        }

        let method = method.unwrap_or_else(|| "auto".to_string());
        let similarity = similarity.unwrap_or_else(|| "cosine".to_string());

        let query = self.engine.get_vector(idx)
            .ok_or_else(|| {
                error!("Invalid index for top_k_similar: {}", idx);
                PyValueError::new_err(format!("Invalid index: {}", idx))
            })?;

        // Parse similarity metric
        let similarity_metric = SimilarityMetric::from_str(&similarity)
            .map_err(|e| PyValueError::new_err(e))?;

        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar_with_similarity(query, k, &similarity_metric),
            "simd" => self.engine.top_k_query_simd_with_similarity(query, k, &similarity_metric),
            "auto" => self.engine.top_k_query_with_similarity(query, k, &similarity_metric),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.", other
                )));
            }
        };

        result.map_err(|e| PyValueError::new_err(format!("Similar search failed: {}", e)))
    }

    fn top_k_subset(
        &self,
        query: Vec<f32>,
        subset: Vec<usize>,
        k: usize,
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        // Validações
        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err(
                format!("Query vector dimension mismatch: expected {}, got {}", 
                       self.engine.dims(), query.len())
            ));
        }
        
        if subset.is_empty() {
            return Ok(Vec::new());
        }
        
        if k == 0 {
            return Ok(Vec::new());
        }
        
        // Validar índices do subset
        for &idx in &subset {
            if idx >= self.engine.rows() {
                return Err(PyValueError::new_err(
                    format!("Subset index {} out of bounds (max: {})", idx, self.engine.rows() - 1)
                ));
            }
        }
        
        // Validar query
        if query.iter().any(|&x| !x.is_finite()) {
            return Err(PyValueError::new_err("Query vector contains invalid values"));
        }

        let _method = method.unwrap_or_else(|| "auto".to_string());

        self.engine.top_k_subset(&query, &subset, k)
            .map_err(|e| {
                error!("Subset search failed: {}", e);
                PyValueError::new_err(format!("Subset search failed: {}", e))
            })
    }
    
    /// Método de health check exposto ao Python
    fn health_check(&self) -> PyResult<bool> {
        match self.engine.health_check() {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false) // Retornar false em vez de erro para não quebrar
            }
        }
    }
    
    /// Obter estatísticas do engine
    fn get_stats(&self) -> PyResult<(usize, f64, usize, usize, usize, usize, f64)> {
        Ok(self.engine.get_stats())
    }
    
    /// Verificar se tem ANN
    fn has_ann(&self) -> bool {
        self.engine.has_ann()
    }
    
    /// Obter uso de memória estimado
    fn memory_usage_mb(&self) -> f64 {
        self.engine.memory_usage_bytes() as f64 / (1024.0 * 1024.0)
    }
}

// ========== TRATAMENTO DE CLEANUP ==========
impl Drop for PySearchEngine {
    fn drop(&mut self) {
        info!("PySearchEngine dropped");
    }
}

/// Módulo Python principal
#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Inicializar logging
    crate::utils::logger::init_logging();
    
    info!("NSeekFS module initialized");
    
    // Adicionar classes e funções
    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(py_prepare_bin_from_embeddings, m)?)?;
    
    // Adicionar constantes úteis
    m.add("MAX_DIMENSIONS", MAX_DIMENSIONS)?;
    m.add("MIN_DIMENSIONS", MIN_DIMENSIONS)?;
    m.add("MAX_VECTORS", MAX_EMBEDDINGS_SIZE)?;
    
    // Adicionar versão
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}