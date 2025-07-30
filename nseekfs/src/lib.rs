use pyo3::prelude::*;
use crate::engine::SearchEngine;

mod io;
mod engine;

#[pyclass]
struct PySearchEngine {
    inner: SearchEngine,
}

#[pymethods]
impl PySearchEngine {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let engine = SearchEngine::load_auto(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PySearchEngine { inner: engine })
    }

    fn top_k_similar(&self, index: usize, k: usize) -> Vec<(usize, f32)> {
        self.inner.top_k_similar(index, k)
    }

    fn top_k_query(&self, vector: Vec<f32>, k: usize, normalize: bool) -> PyResult<Vec<(usize, f32)>> {
        self.inner.top_k_query(&vector, k, normalize)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// ✅ Novo método: retorna o número de vetores
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// ✅ Novo método: retorna o número de dimensões
    fn dims(&self) -> usize {
        self.inner.dims
    }
}


#[pyfunction]
fn PrepareEngine(path: String, normalize: bool, force: bool) -> PyResult<String> {
    engine::SearchEngine::prepare(&path, normalize, force)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(PrepareEngine, m)?)?;
    Ok(())
}
