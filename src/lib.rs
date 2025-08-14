use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyValueError, PyKeyError};
use std::collections::HashMap;
use std::path::Path;
use numpy::PyReadonlyArrayDyn;
use log::{error, info, warn};

mod utils;
mod engine;
mod ann_opt;
mod prepare;

use crate::engine::Engine;
use crate::utils::vector::SimilarityMetric;
use crate::prepare::prepare_bin_from_embeddings;

const MAX_DIMENSIONS: usize = 10000;
const MIN_DIMENSIONS: usize = 1;
const MAX_EMBEDDINGS_SIZE: usize = 100_000_000;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyQueryResult {
    #[pyo3(get)] pub results: Vec<PyQueryItem>,
    #[pyo3(get)] pub query_time_ms: f64,
    #[pyo3(get)] pub method_used: String,
    #[pyo3(get)] pub candidates_generated: usize,
    #[pyo3(get)] pub simd_used: bool,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyQueryItem {
    #[pyo3(get)] pub idx: usize,
    #[pyo3(get)] pub score: f32,
}

#[pymethods]
impl PyQueryItem {
    fn __repr__(&self) -> String { format!("QueryItem(idx={}, score={:.6})", self.idx, self.score) }
    fn __getitem__(&self, key: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match key {
                "idx" => Ok(self.idx.into_py(py)),
                "score" => Ok(self.score.into_py(py)),
                _ => Err(PyKeyError::new_err(format!("Key '{}' not found", key))),
            }
        })
    }
}

#[pymethods]
impl PyQueryResult {
    fn __repr__(&self) -> String {
        format!("QueryResult(results={}, time={:.3}ms, method={})", self.results.len(), self.query_time_ms, self.method_used)
    }
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let results_list = PyList::empty(py);
            for item in &self.results {
                let item_dict = PyDict::new(py);
                item_dict.set_item("idx", item.idx)?;
                item_dict.set_item("score", item.score)?;
                results_list.append(item_dict)?;
            }
            dict.set_item("results", results_list)?;
            dict.set_item("query_time_ms", self.query_time_ms)?;
            dict.set_item("method_used", &self.method_used)?;
            dict.set_item("candidates_generated", self.candidates_generated)?;
            dict.set_item("simd_used", self.simd_used)?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (embeddings, dims, rows, base_name, level, output_dir, ann, normalize, seed=None))]
fn py_prepare_bin_from_embeddings(
    embeddings: PyReadonlyArrayDyn<f32>,
    dims: usize,
    rows: usize,
    base_name: String,
    level: String,
    output_dir: String,
    ann: bool,
    normalize: bool,
    seed: Option<u64>,
) -> PyResult<String> {
    info!("Creating binary index: {}x{} vectors, level={}, ann={}", rows, dims, level, ann);

    if dims == 0 || dims > MAX_DIMENSIONS {
        return Err(PyValueError::new_err(format!("Invalid dimensions: {} (range: 1-{})", dims, MAX_DIMENSIONS)));
    }
    if rows == 0 || rows > MAX_EMBEDDINGS_SIZE {
        return Err(PyValueError::new_err(format!("Invalid rows: {} (range: 1-{})", rows, MAX_EMBEDDINGS_SIZE)));
    }
    if embeddings.len() != dims * rows {
        return Err(PyValueError::new_err(format!("Embeddings size mismatch: expected {}, got {}", dims * rows, embeddings.len())));
    }
    if base_name.trim().is_empty() { return Err(PyValueError::new_err("Base name cannot be empty")); }

    match level.as_str() { "f32" | "f16" | "f8" => {}, _ => return Err(PyValueError::new_err("Invalid level: use 'f32', 'f16', or 'f8'")) }

    let seed = seed.unwrap_or(42);
    let flat = embeddings.as_slice().map_err(|e| {
        error!("Failed to convert embeddings to slice: {}", e);
        PyValueError::new_err(format!("Failed to access embeddings data: {}", e))
    })?;

    // sample sanity
    let sample_size = flat.len().min(10000);
    let invalid = flat[..sample_size].iter().filter(|&&x| !x.is_finite()).count();
    if invalid > 0 {
        let pct = (invalid as f64 / sample_size as f64) * 100.0;
        if pct > 1.0 { return Err(PyValueError::new_err(format!("Too many invalid values: {:.1}% ({}/{})", pct, invalid, sample_size))); }
        warn!("Found {} invalid values in sample", invalid);
    }

    let output_path_opt = Some(Path::new(&output_dir));
    info!("Calling prepare_bin_from_embeddings");

    // permitir que o Rust trabalhe sem GIL
    let pathbuf = Python::with_gil(|py| {
        py.allow_threads(|| {
            prepare_bin_from_embeddings(flat, dims, rows, &base_name, &level, output_path_opt, ann, normalize, seed)
        })
    }).map_err(|e| {
        error!("Failed to prepare binary: {}", e);
        PyValueError::new_err(format!("Index creation failed: {}", e))
    })?;

    Ok(pathbuf.to_string_lossy().to_string())
}

#[pyclass]
struct PySearchEngine { engine: Engine }

#[pymethods]
impl PySearchEngine {
    #[new]
    fn new(path: &str, ann: Option<bool>) -> PyResult<Self> {
        info!("Creating PySearchEngine from path: {}", path);
        if path.trim().is_empty() { return Err(PyValueError::new_err("Path cannot be empty")); }
        let path_obj = Path::new(path);
        if !path_obj.exists() { return Err(PyValueError::new_err(format!("File not found: {}", path))); }
        if !path_obj.is_file() { return Err(PyValueError::new_err(format!("Path is not a file: {}", path))); }
        let ann = ann.unwrap_or(true);

        let engine = Python::with_gil(|py| {
            py.allow_threads(|| Engine::from_bin(path, ann))
        }).map_err(|e| {
            error!("Failed to load engine from {}: {}", path, e);
            PyValueError::new_err(format!("Failed to load engine: {}", e))
        })?;

        if let Err(e) = engine.health_check() {
            error!("Engine health check failed: {}", e);
            return Err(PyValueError::new_err(format!("Engine health check failed: {}", e)));
        }
        info!("PySearchEngine created successfully: {}x{} vectors", engine.rows(), engine.dims());
        Ok(Self { engine })
    }

    fn dims(&self) -> usize { self.engine.dims() }
    fn rows(&self) -> usize { self.engine.rows() }

    fn get_vector(&self, idx: usize) -> PyResult<Vec<f32>> {
        if idx >= self.engine.rows() {
            return Err(PyValueError::new_err(format!("Index {} out of bounds (max: {})", idx, self.engine.rows()-1)));
        }
        self.engine.get_vector(idx).map(|v| v.to_vec()).ok_or_else(|| {
            error!("Failed to get vector at index {}", idx);
            PyValueError::new_err(format!("Failed to get vector at index {}", idx))
        })
    }

    fn query_with_timing(&self, py: Python<'_>, query: Vec<f32>, top_k: usize, method: Option<String>) -> PyResult<PyQueryResult> {
        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err(format!("Query dimension mismatch: expected {}, got {}", self.engine.dims(), query.len())));
        }
        if top_k == 0 {
            return Ok(PyQueryResult{results:vec![],query_time_ms:0.0,method_used:"none".into(),candidates_generated:0,simd_used:false});
        }
        if top_k > 100000 { return Err(PyValueError::new_err("top_k too large")); }

        let method_ref = method.as_deref();
        let result = py.allow_threads(|| self.engine.query_unified(&query, top_k, method_ref))
            .map_err(|e| PyValueError::new_err(format!("Query failed: {}", e)))?;

        let py_results = result.results.into_iter().map(|(idx,score)| PyQueryItem{idx,score}).collect();
        Ok(PyQueryResult{
            results:py_results,
            query_time_ms: result.query_time_ms,
            method_used: result.method_used,
            candidates_generated: result.candidates_generated,
            simd_used: result.simd_used,
        })
    }

    fn query_ann_debug(&self, py: Python<'_>, query: Vec<f32>, top_k: usize) -> PyResult<PyQueryResult> {
        if query.len()!=self.engine.dims(){ return Err(PyValueError::new_err("Query dimension mismatch")); }
        let (results, timing) = py.allow_threads(|| self.engine.query_ann_with_timing(&query, top_k))
            .map_err(|e| PyValueError::new_err(format!("ANN query failed: {}", e)))?;
        Ok(PyQueryResult{
            results: results.into_iter().map(|(i,s)| PyQueryItem{idx:i,score:s}).collect(),
            query_time_ms: timing, method_used:"ann".into(),
            candidates_generated:0, simd_used: query.len()>=64
        })
    }

    fn query_exact_debug(&self, py: Python<'_>, query: Vec<f32>, top_k: usize) -> PyResult<PyQueryResult> {
        if query.len()!=self.engine.dims(){ return Err(PyValueError::new_err("Query dimension mismatch")); }
        let (results, timing) = py.allow_threads(|| self.engine.query_exact_with_timing(&query, top_k))
            .map_err(|e| PyValueError::new_err(format!("Exact query failed: {}", e)))?;
        Ok(PyQueryResult{
            results: results.into_iter().map(|(i,s)| PyQueryItem{idx:i,score:s}).collect(),
            query_time_ms: timing, method_used:"exact".into(),
            candidates_generated:self.engine.rows(), simd_used: query.len()>=64
        })
    }

    #[pyo3(signature = (query, k, method=None, similarity=None))]
    fn top_k_query(&self, py: Python<'_>, query: Vec<f32>, k: usize, method: Option<String>, similarity: Option<String>) -> PyResult<Vec<(usize,f32)>> {
        let _similarity = similarity.unwrap_or_else(|| "cosine".to_string());
        if query.len()!=self.engine.dims(){ return Err(PyValueError::new_err("Query vector dimension mismatch")); }
        if k==0 { return Ok(vec![]); }
        if k>self.engine.rows(){ return Err(PyValueError::new_err("top_k cannot exceed number of vectors")); }
        if k>100000{ return Err(PyValueError::new_err("top_k too large")); }
        if query.iter().any(|&x| !x.is_finite()) { return Err(PyValueError::new_err("Query vector contains NaN/Inf")); }
        let py_result = self.query_with_timing(py, query, k, method)?;
        Ok(py_result.results.into_iter().map(|it| (it.idx, it.score)).collect())
    }

    fn top_k_similar(&self, py: Python<'_>, idx: usize, k: usize, method: Option<String>, similarity: Option<String>) -> PyResult<Vec<(usize,f32)>> {
        if idx>=self.engine.rows(){ return Err(PyValueError::new_err("Index out of bounds")); }
        if k==0 { return Ok(vec![]); }
        if k>self.engine.rows(){ return Err(PyValueError::new_err("top_k cannot exceed number of vectors")); }

        let method = method.unwrap_or_else(|| "auto".to_string());
        let similarity = similarity.unwrap_or_else(|| "cosine".to_string());
        let query = self.engine.get_vector(idx).ok_or_else(|| PyValueError::new_err("Invalid index"))?;

        let metric = SimilarityMetric::from_str(&similarity).map_err(PyValueError::new_err)?;
        let res = py.allow_threads(|| match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar_with_similarity(query, k, &metric),
            "simd"   => self.engine.top_k_query_scalar_with_similarity(query, k, &metric),
            "auto"   => self.engine.top_k_query_with_similarity(query, k, &metric),
            other => Err(format!("Invalid method: {}", other)),
        }).map_err(|e| PyValueError::new_err(format!("Similar search failed: {}", e)))?;
        Ok(res)
    }

    fn top_k_subset(&self, py: Python<'_>, query: Vec<f32>, subset: Vec<usize>, k: usize, _method: Option<String>) -> PyResult<Vec<(usize,f32)>> {
        if query.len()!=self.engine.dims(){ return Err(PyValueError::new_err("Query vector dimension mismatch")); }
        if subset.is_empty(){ return Ok(vec![]); }
        if k==0 { return Ok(vec![]); }
        for &i in &subset { if i>=self.engine.rows(){ return Err(PyValueError::new_err("Subset index out of bounds")); } }
        if query.iter().any(|&x| !x.is_finite()) { return Err(PyValueError::new_err("Query vector contains invalid values")); }

        py.allow_threads(|| self.engine.top_k_subset(&query, &subset, k))
            .map_err(|e| PyValueError::new_err(format!("Subset search failed: {}", e)))
    }

    fn health_check(&self) -> PyResult<bool> {
        match self.engine.health_check() { Ok(_) => Ok(true), Err(e) => { warn!("Health failed: {}", e); Ok(false) } }
    }
    fn get_stats(&self) -> PyResult<(usize,f64,usize,usize,usize,usize,f64)> { Ok(self.engine.get_stats()) }
    fn has_ann(&self) -> bool { self.engine.has_ann() }
    fn memory_usage_mb(&self) -> f64 { self.engine.memory_usage_bytes() as f64 / (1024.0*1024.0) }
}

impl Drop for PySearchEngine { fn drop(&mut self) { info!("PySearchEngine dropped"); } }

#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    crate::utils::logger::init_logging();
    info!("NSeekFS module initialized with timing support");
    m.add_class::<PySearchEngine>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyQueryItem>()?;
    m.add_function(wrap_pyfunction!(py_prepare_bin_from_embeddings, m)?)?;
    m.add("MAX_DIMENSIONS", MAX_DIMENSIONS)?;
    m.add("MIN_DIMENSIONS", MIN_DIMENSIONS)?;
    m.add("MAX_VECTORS", MAX_EMBEDDINGS_SIZE)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
