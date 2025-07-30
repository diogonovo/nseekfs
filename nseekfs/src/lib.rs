use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;

use std::fs::File;
use std::path::Path;

mod engine;
mod io;

use engine::SearchEngine;
use io::{load_csv_bin, save_bin};

#[pyclass]
pub struct PySearchEngine {
    engine: SearchEngine,
}

#[pymethods]
impl PySearchEngine {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let engine = load_csv_bin(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { engine })
    }

    #[staticmethod]
    pub fn from_vectors(py: Python<'_>, obj: &PyAny) -> PyResult<Self> {
        let list: Vec<Vec<f32>> = if obj.is_instance_of::<PyList>() {
            obj.extract()?
        } else {
            let np = obj.call_method1("astype", ("float32",))?;
            np.extract()?
        };

        let mut engine = SearchEngine::new();
        for v in list {
            engine.add(v);
        }
        Ok(Self { engine })
    }

    pub fn top_k_query(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let results = self.engine.top_k(&query, k);
        Ok(results)
    }

    pub fn top_k_subset(&self, py: Python<'_>, query: Vec<f32>, subset: Vec<usize>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let results = self.engine.top_k_subset(&query, &subset, k);
        Ok(results)
    }

    pub fn dims(&self) -> usize {
        self.engine.dims()
    }

    pub fn rows(&self) -> usize {
        self.engine.rows()
    }

    pub fn get_vector(&self, idx: usize) -> PyResult<Vec<f32>> {
        self.engine.get_vector(idx).ok_or_else(|| PyValueError::new_err("Índice inválido"))
    }
}

#[pyfunction]
pub fn prepare_bin_from_numpy(py: Python<'_>, obj: &PyAny, level: &str, base_path: &str) -> PyResult<String> {
    let array: Vec<Vec<f32>> = if obj.is_instance_of::<PyList>() {
        obj.extract()?
    } else {
        let np = obj.call_method1("astype", ("float32",))?;
        np.extract()?
    };

    let path = format!("{}_{}.bin", base_path, level);
    save_bin(&array, Path::new(&path)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(path)
}

#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(prepare_bin_from_numpy, m)?)?;
    Ok(())
}
