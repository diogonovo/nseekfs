use rayon::prelude::*;
use std::fs::{File, OpenOptions, remove_file, rename, create_dir_all};
use std::path::{Path, PathBuf};
use wide::f32x8;
use memmap2::MmapMut;
use crate::ann_opt::AnnIndex;
use std::io::{Read, Write};  

pub fn prepare_bin_from_embeddings(
    embeddings: &[f32],
    dims: usize,
    rows: usize,
    output_path_str: &str,
    level: &str,
    use_ann: bool,
    normalize: bool,
    seed: u64,
) -> Result<PathBuf, String> {
    if dims == 0 || rows == 0 {
        return Err("Empty embeddings input".into());
    }
    if embeddings.len() != dims * rows {
        return Err("Embedding data shape mismatch".into());
    }

    let mut data = embeddings.to_vec();

    if normalize {
        normalize_rows_simd(&mut data, dims);
    }

    if level != "f32" {
        quantize_in_place(&mut data, level)?;
    }

    // Usa o caminho passado pelo Python (100% respeitado agora)
    let output_path = PathBuf::from(output_path_str);

    if use_ann {
        let ann_file = output_path.with_extension("ann");
        build_ann_index(&data, dims, rows, seed, &ann_file)?;
    }

    write_bin_mmap(&data, dims, rows, &output_path)?;

    Ok(output_path)
}

fn normalize_rows_simd(data: &mut [f32], dims: usize) {
    data.par_chunks_mut(dims).for_each(|row| {
        let mut sum = 0.0;
        for chunk in row.chunks_exact(8) {
            let simd = f32x8::new(chunk.try_into().unwrap());
            let squared = simd * simd;
            sum += squared.reduce_add();
        }
        for val in &row[chunked_len(dims)..] {
            sum += val * val;
        }
        let norm = sum.sqrt().max(1e-12);
        for val in row.iter_mut() {
            *val /= norm;
        }
    });
}

#[inline(always)]
fn chunked_len(len: usize) -> usize {
    len / 8 * 8
}

fn quantize_in_place(data: &mut [f32], level: &str) -> Result<(), String> {
    match level {
        "f16" => Ok(()), // To be implemented
        "f8" => Ok(()),  // To be implemented
        "f64" => Ok(()), // To be implemented
        _ => Err("Unsupported quantization level".into())
    }
}

fn build_ann_index(data: &[f32], dims: usize, rows: usize, seed: u64, ann_path: &Path) -> Result<(), String> {
    if let Some(parent) = ann_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create ANN directory: {}", e))?;
    }

    let ann = AnnIndex::build(data, dims, rows, 32, seed);
    ann.save(ann_path).map_err(|e| format!("Failed to save ANN: {}", e))?;
    Ok(())
}

fn write_bin_mmap(data: &[f32], dims: usize, rows: usize, path: &Path) -> Result<(), String> {
    if data.len() != dims * rows {
        return Err(format!(
            "❌ Mismatch: data.len() = {}, but dims * rows = {} * {} = {}",
            data.len(),
            dims,
            rows,
            dims * rows
        ));
    }

    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let tmp_path = path.with_extension("tmp");

    if tmp_path.exists() {
        remove_file(&tmp_path).map_err(|e| format!("❌ Failed to remove tmp '{}': {}", tmp_path.display(), e))?;
    }

    if path.exists() {
        remove_file(path).map_err(|e| {
            format!(
                "❌ Não foi possível sobrescrever '{}': {}.\n💡 Fecha aplicações que o usem ou muda o nome.",
                path.display(), e
            )
        })?;
    }

    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };

    let total_size = 8 + data_bytes.len();

    let file = OpenOptions::new()
        .read(true)   // ✅ necessário para mmap no Windows
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| format!("Failed to open tmp '{}': {}", tmp_path.display(), e))?;

    file.set_len(total_size as u64).map_err(|e| e.to_string())?;
    let mut mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| e.to_string())? };

    mmap[..4].copy_from_slice(&(dims as u32).to_le_bytes());
    mmap[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
    mmap[8..8 + data_bytes.len()].copy_from_slice(data_bytes);
    mmap.flush().map_err(|e| e.to_string())?;

    let actual_size = std::fs::metadata(&tmp_path)
        .map_err(|e| format!("Metadata error: {}", e))?
        .len();

    if actual_size != total_size as u64 {
        return Err(format!(
            "❌ Binary size mismatch: expected {}, got {}",
            total_size, actual_size
        ));
    }

    rename(&tmp_path, path).map_err(|e| format!("Rename failed: {}", e))?;

    let mut buf = [0u8; 8];
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let dims_from_bin = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let rows_from_bin = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    println!("📦 BIN HEADER: dims={}, rows={}", dims_from_bin, rows_from_bin);

    Ok(())
}


