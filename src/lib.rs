use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, wrap_pyfunction};

mod ann;
mod engine;
mod io;
mod utils;
mod prebin;
mod ann_opt;

use prebin::prepare_bin_from_embeddings;
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

#[pyfunction]
fn py_prepare_bin_from_embeddings(
    embeddings: Vec<Vec<f32>>,
    dims: usize,
    output_path: &str,
    level: &str,
    use_ann: bool,
    normalize: bool,
    seed: u64,
) -> PyResult<String> {
    let rows = embeddings.len();
    let flat: Vec<f32> = embeddings.into_iter().flatten().collect();

    prebin::prepare_bin_from_embeddings(
        &flat, dims, rows, output_path, level, use_ann, normalize, seed
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(output_path.to_string())
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
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.engine.dims() {
            error!("Query vector dimension does not match engine");
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let normalize = normalize.unwrap_or(true);
        let method = method.unwrap_or_else(|| "simd".to_string());

        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar(&query, k, normalize),
            "simd" => self.engine.top_k_query_simd(&query, k, normalize),
            "auto" => self.engine.top_k_query(&query, k, normalize),
            other => {
                error!("Invalid method: {}", other);
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )));
            }
        };

        result.map_err(|e| {
            error!("Error: {}", e);
            PyValueError::new_err(e)
        })
    }


    fn top_k_similar(
        &self,
        idx: usize,
        k: usize,
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let method = method.unwrap_or_else(|| "simd".to_string());

        let query = self.engine.get_vector(idx).ok_or_else(|| {
            error!("Index out of bounds: {}", idx);
            PyValueError::new_err("Invalid index")
        })?;

        let result = match method.as_str() {
            "scalar" => self.engine.top_k_query_scalar(query, k, false),
            "simd" => self.engine.top_k_query_simd(query, k, false),
            "auto" => self.engine.top_k_query(query, k, false),
            other => {
                error!("Invalid method: {}", other);
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )));
            }
        };

        result.map_err(|e| {
            error!("Error: {}", e);
            PyValueError::new_err(e)
        })
    }


    fn top_k_subset(
        &self,
        query: Vec<f32>,
        subset: Vec<usize>,
        k: usize,
        normalize: Option<bool>,
        method: Option<String>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.engine.dims() {
            error!("Query vector dimension does not match engine");
            return Err(PyValueError::new_err(
                "Query vector dimension does not match engine",
            ));
        }

        let normalize = normalize.unwrap_or(true);
        let method = method.unwrap_or_else(|| "simd".to_string());

        let result = match method.as_str() {
            "scalar" | "simd" | "auto" => {
                // todas usam a mesma função base
                self.engine.top_k_subset(&query, &subset, k, normalize)
            }
            other => {
                error!("Invalid method: {}", other);
                return Err(PyValueError::new_err(format!(
                    "Invalid method: '{}'. Use 'simd', 'scalar', or 'auto'.",
                    other
                )));
            }
        };

        result.map_err(|e| {
            error!("Error: {}", e);
            PyValueError::new_err(e)
        })
    }

}

#[pymodule]
fn nseekfs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    crate::utils::logger::init_logging();

    m.add_class::<PySearchEngine>()?;
    m.add_function(wrap_pyfunction!(prepare_engine, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_from_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_engine_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_prepare_bin_from_embeddings, m)?)?;
    Ok(())
}
