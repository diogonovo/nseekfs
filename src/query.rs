use wide::f32x8;
use std::convert::TryInto;

const LANES: usize = 8;

// Standalone SIMD dot product function (exported for use in engine.rs)
pub fn compute_score_simd(query: &[f32], vec: &[f32]) -> f32 {
    let chunks = query.len() / LANES;
    let mut simd_sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let q = f32x8::new(query[i * LANES..i * LANES + 8].try_into().unwrap());
        let v = f32x8::new(vec[i * LANES..i * LANES + 8].try_into().unwrap());
        simd_sum += q * v;
    }

    let mut total = simd_sum.reduce_add();
    for i in (chunks * LANES)..query.len() {
        total += query[i] * vec[i];
    }

    total
}
