use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod io;
mod engine;

#[pyfunction]
fn load_vectors(path: String) -> PyResult<()> {
    engine::load_vectors(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn get_cached_vectors() -> PyResult<Vec<Vec<f32>>> {
    Ok(engine::get_cached_vectors())
}

#[pyfunction]
fn top_k_similar(index: usize, k: usize) -> PyResult<Vec<(usize, f32)>> {
    Ok(engine::top_k_similar(index, k))
}

#[pymodule]
fn nseekfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(get_cached_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_similar, m)?)?;
    Ok(())
}
