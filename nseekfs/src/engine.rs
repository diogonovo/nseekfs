use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use wide::f32x8;
use ordered_float::OrderedFloat;

pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut i = 0;
    let len = a.len().min(b.len());

    while i + 8 <= len {
        let a_vec = f32x8::new([
            a[i], a[i + 1], a[i + 2], a[i + 3],
            a[i + 4], a[i + 5], a[i + 6], a[i + 7],
        ]);
        let b_vec = f32x8::new([
            b[i], b[i + 1], b[i + 2], b[i + 3],
            b[i + 4], b[i + 5], b[i + 6], b[i + 7],
        ]);
        sum += a_vec * b_vec;
        i += 8;
    }

    let mut remainder = 0.0;
    while i < len {
        remainder += a[i] * b[i];
        i += 1;
    }

    sum.reduce_add() + remainder
}

pub fn normalize_all(vectors: &mut [Vec<f32>]) {
    vectors.par_iter_mut().for_each(|v| {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm != 0.0 {
            for val in v.iter_mut() {
                *val /= norm;
            }
        }
    });
}

pub fn top_k_similar(input: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let partial_heaps: Vec<BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>> = vectors
        .par_chunks(1000)
        .map(|chunk| {
            let mut heap = BinaryHeap::with_capacity(k);
            for (i, v) in chunk.iter().enumerate() {
                let score = cosine_similarity_simd(input, v);
                let entry = Reverse((OrderedFloat(score), i));
                if heap.len() < k {
                    heap.push(entry);
                } else if let Some(&Reverse((lowest, _))) = heap.peek() {
                    if OrderedFloat(score) > lowest {
                        heap.pop();
                        heap.push(entry);
                    }
                }
            }
            heap
        })
        .collect();

    let mut final_heap = BinaryHeap::with_capacity(k);
    for heap in partial_heaps {
        for Reverse((score, i)) in heap {
            let entry = Reverse((score, i));
            if final_heap.len() < k {
                final_heap.push(entry);
            } else if let Some(&Reverse((lowest, _))) = final_heap.peek() {
                if score > lowest {
                    final_heap.pop();
                    final_heap.push(entry);
                }
            }
        }
    }

    let mut results: Vec<_> = final_heap.into_sorted_vec();
    results.reverse(); // do maior para menor
    results
        .into_iter()
        .map(|Reverse((score, i))| (i, score.0))
        .collect()
}
