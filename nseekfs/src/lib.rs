use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, exceptions::PyValueError};

mod engine;
mod io;
mod utils;
mod ann;

use engine::Engine;

#[pyfunction]
fn prepare_engine(path: &str, normalize: bool, force: bool, use_ann: bool) -> PyResult<String> {
    match io::write_bin_file(path, normalize, force, use_ann) {
        Ok(p) => Ok(p),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn prepare_engine_from_embeddings(
    py_embeddings: &PyAny,
    base_path: &str,
    precision: &str,
    normalize: bool,
    use_ann: bool,
) -> PyResult<String> {
    let mut embeddings: Vec<Vec<f32>> = py_embeddings
        .extract::<Vec<Vec<f32>>>()
        .map_err(|e| PyValueError::new_err(format!("Erro ao extrair embeddings: {}", e)))?;

    match io::write_bin_from_embeddings(&mut embeddings, base_path, precision, normalize, use_ann) {
        Ok(p) => Ok(p),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn prepare_engine_py(path: &str, normalize: bool, force: bool, use_ann: bool) -> PyResult<String> {
    prepare_engine(path, normalize, force, use_ann)
        .map_err(|e| PyValueError::new_err(format!("Erro ao preparar engine: {}", e)))
}

#[pyclass]
struct PySearchEngine {
    engine: Engine,
}

#[pymethods]
impl PySearchEngine {
    #[new]
    fn new(path: &str, normalize: Option<bool>, use_ann: Option<bool>) -> PyResult<Self> {
        let normalize = normalize.unwrap_or(false);
        let use_ann = use_ann.unwrap_or(true);

        let engine = if path.ends_with(".bin") {
            Engine::from_bin(path, use_ann)
        } else {
            Engine::from_csv_parallel(path, normalize, use_ann)
        }.map_err(|e| PyValueError::new_err(format!("Erro ao carregar engine: {}", e)))?;

        Ok(Self { engine })
    }

    #[staticmethod]
    fn from_embeddings(py_embeddings: &PyAny, normalize: Option<bool>, use_ann: Option<bool>) -> PyResult<Self> {
        let normalize = normalize.unwrap_or(true);
        let use_ann = use_ann.unwrap_or(true);

        let embeddings: Vec<Vec<f32>> = py_embeddings
            .extract::<Vec<Vec<f32>>>()
            .map_err(|e| PyValueError::new_err(format!("Erro ao converter embeddings: {}", e)))?;

        let engine = Engine::from_embeddings(embeddings, normalize, use_ann)
            .map_err(|e| PyValueError::new_err(format!("Erro ao construir engine: {}", e)))?;

        Ok(Self { engine })
    }

    fn dims(&self) -> usize {
        self.engine.dims()
    }

    fn rows(&self) -> usize {
        self.engine.rows()
    }

    fn get_vector(&self, idx: usize) -> PyResult<Vec<f32>> {
        self.engine.get_vector(idx)
            .map(|v| v.to_vec())
            .ok_or_else(|| PyValueError::new_err("Índice fora dos limites"))
    }

    fn top_k_query(&self, query: Vec<f32>, k: usize, normalize: Option<bool>) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.engine.dims() {
            return Err(PyValueError::new_err("Dimensão do vetor de consulta não corresponde ao engine"));
        }

        let normalize = normalize.unwrap_or(true);
        Ok(self.engine.top_k_query(&query, k, normalize))
    }

    fn top_k_similar(&self, idx: usize, k: usize) -> PyResult<Vec<(usize, f32)>> {
        self.engine.top_k_index(idx, k)
            .ok_or_else(|| PyValueError::new_err("Índice inválido"))
    }
}

#[pymodule]
fn nseekfs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(prepare_engine, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_from_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_py, m)?)?;
    Ok(())
}
