use std::fs::File;
use std::io::{BufRead, BufReader};

/// Normaliza um vetor inplace (divide cada componente pelo seu L2-norm)
pub fn normalize_vector_inplace(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Similaridade cosseno entre dois vetores
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Lê um CSV de vetores float e devolve Vec<Vec<f32>>
pub fn read_csv_matrix(path: &str) -> Result<Vec<Vec<f32>>, String> {
    let file = File::open(path).map_err(|e| format!("Erro ao abrir CSV: {}", e))?;
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Erro ao ler linha: {}", e))?;
        let row: Result<Vec<f32>, _> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();

        match row {
            Ok(vec) => result.push(vec),
            Err(_) => return Err("Erro ao converter CSV para floats".into()),
        }
    }

    Ok(result)
}

/// Normaliza uma matriz de vetores inplace
pub fn normalize_matrix(matrix: &mut Vec<Vec<f32>>) {
    for vec in matrix.iter_mut() {
        normalize_vector_inplace(vec);
    }
}
