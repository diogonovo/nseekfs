use rayon::prelude::*;
use std::fs::{File, remove_file, rename, create_dir_all, OpenOptions};
use std::path::Path;
use wide::f32x8;
use memmap2::MmapMut;
use sha2::{Sha256, Digest};
use crate::ann_opt::AnnIndex;
use std::io::Read;

pub fn prepare_bin_from_embeddings(
    embeddings: &[f32],
    dims: usize,
    rows: usize,
    output_path: &str,
    level: &str,
    use_ann: bool,
    normalize: bool,
    seed: u64,
) -> Result<(), String> {
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

    if use_ann {
        let ann_file = format!("{}.ann", output_path);
        build_ann_index(&data, dims, rows, seed, &ann_file)?;
    }

    write_bin_mmap(&data, dims, rows, output_path)?;

    Ok(())
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

fn build_ann_index(data: &[f32], dims: usize, rows: usize, seed: u64, ann_path: &str) -> Result<(), String> {
    let ann = AnnIndex::build(data, dims, rows, 32, seed);
    ann.save(ann_path).map_err(|e| format!("Failed to save ANN: {}", e))?;
    Ok(())
}

fn write_bin_mmap(data: &[f32], dims: usize, rows: usize, path: &str) -> Result<(), String> {
    if data.len() != dims * rows {
        return Err(format!(
            "❌ Mismatch: data.len() = {}, but dims * rows = {} * {} = {}",
            data.len(),
            dims,
            rows,
            dims * rows
        ));
    }

    let tmp_path = format!("{}.tmp", path);

    if let Some(parent) = Path::new(&tmp_path).parent() {
        create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let _ = std::fs::remove_file(&tmp_path);
    let _ = std::fs::remove_file(&path);

    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };

    let mut hasher = Sha256::new();
    hasher.update(data_bytes);
    let hash = hasher.finalize();

    let total_size = 8 + data_bytes.len() + hash.len();

    let file = File::create(&tmp_path).map_err(|e| e.to_string())?;
    file.set_len(total_size as u64).map_err(|e| e.to_string())?;
    let mut mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| e.to_string())? };

    mmap[..4].copy_from_slice(&(dims as u32).to_le_bytes());
    mmap[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
    mmap[8..8 + data_bytes.len()].copy_from_slice(data_bytes);
    mmap[8 + data_bytes.len()..].copy_from_slice(&hash);
    mmap.flush().map_err(|e| e.to_string())?;

    println!("✅ Binary file written to: {}", path);
    println!("    dims = {}, rows = {}, data.len() = {}", dims, rows, data.len());
    println!("    expected size = {}", total_size);

    let meta = std::fs::metadata(&tmp_path).map_err(|e| format!("Metadata error: {}", e))?;
    let actual_size = meta.len();
    println!("    actual size   = {}", actual_size);

    if actual_size != total_size as u64 {
        return Err(format!(
            "❌ Binary size mismatch: expected {}, got {}",
            total_size, actual_size
        ));
    }

    std::fs::rename(&tmp_path, path).map_err(|e| format!("Rename failed: {}", e))?;
    let mut buf = [0u8; 8];
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let dims_from_bin = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let rows_from_bin = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    println!("📦 BIN HEADER: dims={}, rows={}", dims_from_bin, rows_from_bin);

    Ok(())
}



