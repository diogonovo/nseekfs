/// Vector utility functions for similarity calculations

#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl SimilarityMetric {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(SimilarityMetric::Cosine),
            "euclidean" => Ok(SimilarityMetric::Euclidean),
            "dot_product" | "dot" => Ok(SimilarityMetric::DotProduct),
            _ => Err(format!("Unsupported similarity metric: {}", s)),
        }
    }
}

pub fn compute_similarity(a: &[f32], b: &[f32], metric: &SimilarityMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::Euclidean => {
            // Return negative distance so higher values = more similar (for sorting)
            // But make it meaningful: -distance becomes similarity score
            let distance = euclidean_distance(a, b);
            -distance
        },
        SimilarityMetric::DotProduct => dot_product(a, b),
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        dot_product / norm_product
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn normalize_vector_inplace(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_similarity_metric_from_str() {
        assert_eq!(SimilarityMetric::from_str("cosine").unwrap(), SimilarityMetric::Cosine);
        assert_eq!(SimilarityMetric::from_str("euclidean").unwrap(), SimilarityMetric::Euclidean);
        assert_eq!(SimilarityMetric::from_str("dot_product").unwrap(), SimilarityMetric::DotProduct);
        assert_eq!(SimilarityMetric::from_str("dot").unwrap(), SimilarityMetric::DotProduct);
        
        assert!(SimilarityMetric::from_str("invalid").is_err());
    }

    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![3.0, 4.0];
        normalize_vector_inplace(&mut vec);
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}