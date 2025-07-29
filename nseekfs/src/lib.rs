use pyo3::prelude::*;
use crate::engine::{top_k_similar, normalize_all};
use crate::io::load_csv_f32;

mod engine;
mod io;

#[pymodule]
fn nseekfs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_csv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_similar, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_load_csv_f32(path: String) -> PyResult<Vec<Vec<f32>>> {
    let mut raw = load_csv_f32(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    normalize_all(&mut raw);
    Ok(raw)
}


#[pyfunction]
fn py_top_k_similar(input: Vec<f32>, data: Vec<Vec<f32>>, k: usize) -> PyResult<Vec<(usize, f32)>> {
    Ok(top_k_similar(&input, &data, k))
}