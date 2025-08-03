use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, wrap_pyfunction};

mod ann;
mod engine;
mod io;
mod utils;

use engine::Engine;
use log::{error, info};

#[pyfunction]
fn prepare_engine(path: &str, normalize: bool, force: bool, use_ann: bool) -> PyResult<String> {
    info!(
        "Preparing engine from path='{}' (normalize={}, force={}, ann={})",
        path, normalize, force, use_ann
    );
    match io::write_bin_file(path, normalize, force, use_ann) {
        Ok(p) => {
            info!("Engine binary written to '{}'", p);
            Ok(p)
        }
        Err(e) => {
            error!("Failed to write engine binary: {}", e);
            Err(PyValueError::new_err(e))
        }
    }
}

#[pyfunction]
fn prepare_engine_from_embeddings(
    py_embeddings: &Bound<'_, PyAny>,
    base_path: &str,
    precision: &str,
    normalize: bool,
    use_ann: bool,
) -> PyResult<String> {
    info!(
        "Preparing engine from in-memory embeddings (precision={}, normalize={}, ann={})",
        precision, normalize, use_ann
    );

    let mut embeddings: Vec<Vec<f32>> = py_embeddings.extract().map_err(|e| {
        error!("Failed to extract embeddings from Python: {}", e);
        PyValueError::new_err(format!("Failed to extract embeddings: {}", e))
    })?;

    match io::write_bin_from_embeddings(&mut embeddings, base_path, precision, normalize, use_ann) {
        Ok(p) => {
            info!("Engine binary written to '{}'", p);
            Ok(p)
        }
        Err(e) => {
            error!("Failed to write engine from embeddings: {}", e);
            Err(PyValueError::new_err(e))
        }
    }
}

#[pyfunction]
fn prepare_engine_py(path: &str, normalize: bool, force: bool, use_ann: bool) -> PyResult<String> {
    prepare_engine(path, normalize, force, use_ann).map_err(|e| {
        error!("Error preparing engine: {}", e);
        PyValueError::new_err(format!("Error preparing engine: {}", e))
    })
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

        info!(
            "Initializing PySearchEngine with path='{}' (normalize={}, ann={})",
            path, normalize, use_ann
        );

        let engine = if path.ends_with(".bin") {
            Engine::from_bin(path, use_ann)
        } else {
            Engine::from_csv_parallel(path, normalize, use_ann)
        }
        .map_err(|e| {
            error!("Error loading engine {}", e);
            PyValueError::new_err(format!("Error loading engine: {}", e))
        })?;

        Ok(Self { engine })
    }

    #[staticmethod]
    fn from_embeddings(
        py_embeddings: &Bound<'_, PyAny>,
        normalize: Option<bool>,
        use_ann: Option<bool>,
    ) -> PyResult<Self> {
        let normalize = normalize.unwrap_or(true);
        let use_ann = use_ann.unwrap_or(true);

        info!(
            "Building engine from_embeddings (normalize={}, ann={})",
            normalize, use_ann
        );

        let embeddings: Vec<Vec<f32>> = py_embeddings.extract().map_err(|e| {
            error!("Error converting embeddings: {}", e);
            PyValueError::new_err(format!("Error converting embeddings: {}", e))
        })?;

        let engine = Engine::from_embeddings(embeddings, normalize, use_ann).map_err(|e| {
            error!("Error building engine: {}", e);
            PyValueError::new_err(format!("Error building engine: {}", e))
        })?;

        Ok(Self { engine })
    }

    #[staticmethod]
    fn from_embeddings_custom(
        py_embeddings: &Bound<'_, PyAny>,
        normalize: Option<bool>,
        use_ann: Option<bool>,
        bits: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let normalize = normalize.unwrap_or(true);
        let use_ann = use_ann.unwrap_or(true);

        let embeddings: Vec<Vec<f32>> = py_embeddings.extract().map_err(|e| {
            error!("Error converting embeddings: {}", e);
            PyValueError::new_err(format!("Error converting embeddings: {}", e))
        })?;

        let engine = Engine::from_embeddings_custom(embeddings, normalize, use_ann, bits, seed)
            .map_err(|e| {
                error!("Error building engine: {}", e);
                PyValueError::new_err(format!("Error building engine: {}", e))
            })?;

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
            .ok_or_else(|| {
                error!("Index out of limits");
                PyValueError::new_err("Index out of limits")
            })
    }

    fn top_k_query(
        &self,
        query: Vec<f32>,
        k: usize,
        normalize: Option<bool>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.engine.dims() {
            error!("Query vector dimension does not match engine");
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let normalize = normalize.unwrap_or(true);
        self.engine.top_k_query(&query, k, normalize).map_err(|e| {
            error!("Error: {}", e);
            PyValueError::new_err(e)
        })
    }

    fn top_k_similar(&self, idx: usize, k: usize) -> PyResult<Vec<(usize, f32)>> {
        self.engine.top_k_index(idx, k).map_err(|e| {
            error!("Invalid index: {}", e);
            PyValueError::new_err("Invalid index")
        })
    }

    fn top_k_subset(
        &self,
        query: Vec<f32>,
        subset: Vec<usize>,
        k: usize,
        normalize: Option<bool>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.engine.dims() {
            error!("Query vector dimension does not match engine");
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let normalize = normalize.unwrap_or(true);
        self.engine
            .top_k_subset(&query, &subset, k, normalize)
            .map_err(|e| {
                error!("Error: {}", e);
                PyValueError::new_err(e)
            })
    }
}

#[pymodule]
fn nseek(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    crate::utils::logger::init_logging();

    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(prepare_engine, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_from_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_py, m)?)?;
    Ok(())
}
