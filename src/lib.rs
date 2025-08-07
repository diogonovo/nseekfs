use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, wrap_pyfunction};

mod utils;
mod prebin;
mod ann_opt;
mod query;
mod engine;

use prebin::prepare_bin_from_embeddings;
use engine::Engine;
use log::info;

#[pyfunction]
fn py_prepare_bin_from_embeddings(
    embeddings: Vec<Vec<f32>>,
    dims: usize,
    output_path: &str,
    level: &str,
    ann: bool,
    normalize: bool,
    seed: u64,
) -> PyResult<String> {
    let rows = embeddings.len();
    let flat: Vec<f32> = embeddings.into_iter().flatten().collect();

    prepare_bin_from_embeddings(&flat, dims, rows, output_path, level, ann, normalize, seed)
        .map(|pathbuf| pathbuf.to_string_lossy().to_string()) 
        .map_err(|e| PyValueError::new_err(e))
}

#[pyclass]
struct PySearchEngine {
    engine: Engine,
}

#[pymethods]
impl PySearchEngine {
    #[new]
    fn new(path: &str, normalize: Option<bool>, ann: Option<bool>) -> PyResult<Self> {
        let ann = ann.unwrap_or(true);
        let engine = Engine::from_bin(path, ann)
            .map_err(|e| PyValueError::new_err(format!("Error loading engine: {}", e)))?;

        Ok(Self { engine })
    }

    fn dims(&self) -> usize {
        self.engine.dims()
    }

    fn rows(&self) -> usize {
        self.engine.rows()
    }

    fn get_vector(&self, idx: usize) -> PyResult<Vec<f32>> {
        self.engine
            .get_vector(idx)
            .map(|v| v.to_vec())
            .ok_or_else(|| PyValueError::new_err("Index out of limits"))
    }

    fn top_k_query(
        &self,
        query: Vec<f32>,
        k: usize,
        method: Option<String>,
        similarity: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let method = method.unwrap_or_else(|| "simd".to_string());

        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar(&query, k),
            "simd" => self.engine.top_k_query_simd(&query, k),
            "auto" => self.engine.top_k_query(&query, k),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )))
            }
        };

        result.map_err(|e| PyValueError::new_err(e))
    }

    fn top_k_similar(
        &self,
        idx: usize,
        k: usize,
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let method = method.unwrap_or_else(|| "simd".to_string());

        let query = self
            .engine
            .get_vector(idx)
            .ok_or_else(|| PyValueError::new_err("Invalid index"))?;

        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar(query, k),
            "simd" => self.engine.top_k_query_simd(query, k),
            "auto" => self.engine.top_k_query(query, k),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )))
            }
        };

        result.map_err(|e| PyValueError::new_err(e))
    }

    fn top_k_subset(
        &self,
        query: Vec<f32>,
        subset: Vec<usize>,
        k: usize,
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let method = method.unwrap_or_else(|| "simd".to_string());

        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let result = match method.as_str() {
            "scalar" | "simd" | "auto" => {
                self.engine.top_k_subset(&query, &subset, k)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )))
            }
        };

        result.map_err(|e| PyValueError::new_err(e))
    }
}

#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    crate::utils::logger::init_logging();

    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(py_prepare_bin_from_embeddings, m)?)?;

    Ok(())
}
