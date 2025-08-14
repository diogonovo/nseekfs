/*
🔧 ANN INTEGRATION BRIDGE
==========================
Camada de compatibilidade leve: reexporta helpers e dá builders opcionais.
*/

use std::path::Path;
use crate::ann_opt::{AnnIndex, StateOfArtIvfPqIndex};

/// Convenience builder idêntico ao de `ann_opt`, exposto aqui para quem
/// importe `ann_bridge` em vez de `ann_opt`.
pub fn build_state_of_art_ann_index(
    vectors: &[f32],
    dims: usize,
    rows: usize,
    _bits: usize, // legacy, ignorado
    seed: u64,
) -> Box<dyn AnnIndex> {
    let recall_target = 0.95;
    let normalized = true;

    let index = StateOfArtIvfPqIndex::build_with_target_recall(
        vectors, dims, rows, recall_target, normalized, Some(seed),
    );
    Box::new(index)
}

/// Stub de load (ainda não implementado).
pub fn load_state_of_art_ann_index<P: AsRef<Path>>(
    _path: P,
    _vectors: &[f32],
) -> std::io::Result<Box<dyn AnnIndex>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Loading state-of-art indexes not yet implemented",
    ))
}

/// Heurística de “atalho” para exact search (reexport simples).
#[inline]
pub fn should_use_exact_search(rows: usize) -> bool {
    rows < 20_000
}
