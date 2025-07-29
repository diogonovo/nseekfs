use crate::io::load_csv_f32;
use std::sync::Mutex;
use std::error::Error;

static mut CACHED: Option<Mutex<Vec<Vec<f32>>>> = None;

pub fn load_vectors(path: &str) -> Result<(), Box<dyn Error>> {
    let raw = load_csv_f32(path)?;
    unsafe {
        CACHED = Some(Mutex::new(raw));
    }
    Ok(())
}

pub fn get_cached_vectors() -> Vec<Vec<f32>> {
    unsafe {
        CACHED.as_ref().unwrap().lock().unwrap().clone()
    }
}

pub fn top_k_similar(index: usize, k: usize) -> Vec<(usize, f32)> {
    let cache = get_cached_vectors();
    let input = &cache[index];
    let mut sims: Vec<(usize, f32)> = cache.iter().enumerate()
        .map(|(i, v)| {
            let sim = cosine_similarity(input, v);
            (i, sim)
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.truncate(k);
    sims
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}
