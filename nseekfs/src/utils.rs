use std::fs::File;
use std::io::{BufRead, BufReader};


pub fn normalize_vector_inplace(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn read_csv_matrix(path: &str) -> Result<Vec<Vec<f32>>, String> {
    let file = File::open(path)
        .map_err(|e| format!("❌ Erro ao abrir CSV '{}': {}", path, e))?;
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| format!("❌ Erro na linha {} do CSV: {}", i + 1, e))?;

        let row: Result<Vec<f32>, _> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();

        match row {
            Ok(vec) => result.push(vec),
            Err(_) => return Err(format!("❌ Erro ao converter linha {} para floats", i + 1)),
        }
    }

    if result.is_empty() {
        return Err("❌ CSV vazio ou mal formatado".into());
    }

    Ok(result)
}

pub fn normalize_matrix(matrix: &mut Vec<Vec<f32>>) {
    for vec in matrix.iter_mut() {
        normalize_vector_inplace(vec);
    }
}
