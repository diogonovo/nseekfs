use std::cmp::Ordering;

#[derive(Clone)]
pub struct SearchEngine {
    data: Vec<Vec<f32>>,
    dims: usize,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            dims: 0,
        }
    }

    pub fn add(&mut self, vector: Vec<f32>) {
        if self.dims == 0 {
            self.dims = vector.len();
        }
        self.data.push(vector);
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn rows(&self) -> usize {
        self.data.len()
    }

    pub fn get_vector(&self, idx: usize) -> Option<Vec<f32>> {
        self.data.get(idx).cloned()
    }

    pub fn top_k(&self, query: &Vec<f32>, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn top_k_subset(&self, query: &Vec<f32>, subset: &Vec<usize>, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = subset
            .iter()
            .filter_map(|&i| self.data.get(i).map(|v| (i, cosine_similarity(query, v))))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

impl SearchEngine {
    pub fn all_vectors(&self) -> &Vec<Vec<f32>> {
        &self.data
    }
}
