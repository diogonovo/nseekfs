use log::{error, info, warn};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn normalize_vector_inplace(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    } else {
        warn!("Vector has zero norm and was not normalized");
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn read_csv_matrix(path: &str) -> Result<Vec<Vec<f32>>, String> {
    let file = File::open(path).map_err(|e| {
        error!("Failed to open CSV '{}': {}", path, e);
        format!("Failed to open CSV '{}': {}", path, e)
    })?;
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            error!("Error reading line {} of CSV '{}': {}", i + 1, path, e);
            format!("Error at line {} in CSV: {}", i + 1, e)
        })?;

        let row: Result<Vec<f32>, _> = line.split(',').map(|s| s.trim().parse::<f32>()).collect();

        match row {
            Ok(vec) => result.push(vec),
            Err(_) => {
                error!("Failed to parse line {} to floats: '{}'", i + 1, line);
                return Err(format!("Failed to parse line {} to floats", i + 1));
            }
        }
    }

    if result.is_empty() {
        error!("CSV is empty or malformed → '{}'", path);
        return Err("CSV is empty or malformed".into());
    }

    info!(
        "Successfully loaded CSV file with {} rows from '{}'",
        result.len(),
        path
    );
    Ok(result)
}

pub fn normalize_matrix(matrix: &mut Vec<Vec<f32>>) {
    info!("Normalizing matrix with {} vectors", matrix.len());
    for vec in matrix.iter_mut() {
        normalize_vector_inplace(vec);
    }
}

pub fn validate_dimensions(matrix: &[Vec<f32>]) -> Result<usize, String> {
    if matrix.is_empty() {
        error!("Matrix is empty");
        return Err("Input is empty".into());
    }

    let expected = matrix[0].len();
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != expected {
            error!(
                "Inconsistent dimensions at row {}: expected {}, found {}",
                i,
                expected,
                row.len()
            );
            return Err(format!(
                "Inconsistent dimension at row {}: expected {}, found {}",
                i,
                expected,
                row.len()
            ));
        }
    }

    Ok(expected)
}
