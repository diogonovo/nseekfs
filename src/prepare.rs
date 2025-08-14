use rayon::prelude::*;
use std::fs::{OpenOptions, remove_file, rename, create_dir_all};
use std::path::{Path, PathBuf};
use wide::f32x8;
use memmap2::MmapMut;
use crate::ann_opt::{AnnIndex, build_state_of_art_ann_index, should_use_exact_search};
use std::io::Read;
use std::time::Instant;
use dirs;
use std::io::{self, Write};
use log::{info, warn, debug};

// ========== CONSTANTES DE SEGURANÇA ==========
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024 * 1024; // 100GB max
const MIN_FREE_SPACE_RATIO: f64 = 0.1; // 10% espaço livre mínimo (reservado para futura verificação real)
const CHUNK_SIZE: usize = 1000; // Processar normalização em chunks

/// Preparar binário a partir de embeddings com validações extensivas
pub fn prepare_bin_from_embeddings(
    embeddings: &[f32],
    dims: usize,
    rows: usize,
    base_name: &str,
    level: &str,
    output_dir: Option<&std::path::Path>,
    ann: bool,
    normalize: bool,
    seed: u64,
) -> Result<std::path::PathBuf, String> {
    let total_start = Instant::now();
    info!("⏱️ Starting prepare_bin_from_embeddings");
    io::stdout().flush().unwrap();

    // ========== VALIDAÇÕES CRÍTICAS ==========
    if dims == 0 || rows == 0 {
        return Err("Invalid dimensions: dims and rows must be > 0".into());
    }
    if embeddings.len() != dims * rows {
        return Err(format!(
            "Embedding data shape mismatch: expected {} elements ({}x{}), got {}",
            dims * rows, dims, rows, embeddings.len()
        ));
    }
    if dims < 8 {
        return Err(format!("Minimum 8 dimensions required, got {}", dims));
    }
    if dims > 10000 {
        return Err(format!("Maximum 10000 dimensions allowed, got {}", dims));
    }
    if rows > 100_000_000 {
        return Err(format!("Maximum 100M vectors allowed, got {}", rows));
    }
    if base_name.trim().is_empty() {
        return Err("Base name cannot be empty".into());
    }
    if base_name.contains('/') || base_name.contains('\\') {
        return Err("Base name cannot contain path separators".into());
    }
    match level {
        "f8" | "f16" | "f32" | "f64" => {},
        _ => return Err(format!("Invalid level '{}'. Must be f8, f16, f32, or f64", level)),
    }

    // Validar dados de entrada (amostra)
    let sample_size = (embeddings.len()).min(10000);
    let invalid_count = embeddings[..sample_size].iter().filter(|&&x| !x.is_finite()).count();
    if invalid_count > 0 {
        let percentage = (invalid_count as f64 / sample_size as f64) * 100.0;
        if percentage > 1.0 {
            return Err(format!(
                "Too many invalid values in embeddings: {:.1}% ({}/{})",
                percentage, invalid_count, sample_size
            ));
        } else {
            warn!("Found {} invalid values in embeddings sample", invalid_count);
        }
    }

    info!("✅ Input validation passed: {} vectors × {} dims", rows, dims);

    // ========== ESTIMATIVA DE ESPAÇO E MEMÓRIA ==========
    let estimated_memory = dims * rows * 4; // f32 = 4 bytes
    let estimated_file_size = 8 + estimated_memory; // header + data

    info!("📊 Estimated memory usage: {:.1}MB", estimated_memory as f64 / (1024.0 * 1024.0));
    info!("📊 Estimated file size: {:.1}MB", estimated_file_size as f64 / (1024.0 * 1024.0));

    if estimated_file_size as u64 > MAX_FILE_SIZE {
        return Err(format!(
            "Estimated file size too large: {:.1}GB (max: {:.1}GB)",
            estimated_file_size as f64 / (1024.0_f64.powi(3)),
            MAX_FILE_SIZE as f64 / (1024.0_f64.powi(3))
        ));
    }

    // ========== PROCESSAMENTO DOS DADOS ==========
    let step1 = Instant::now();
    let mut data = embeddings.to_vec();
    info!("⏱️ [{:.2?}] ✅ Copied embeddings to internal vector", step1.elapsed());
    io::stdout().flush().unwrap();

    if normalize {
        let t = Instant::now();
        normalize_rows_safe(&mut data, dims)?;
        info!("⏱️ [{:.2?}] ✅ Normalization completed", t.elapsed());
        io::stdout().flush().unwrap();
    }

    if level != "f32" {
        let t = Instant::now();
        quantize_in_place_safe(&mut data, level)?;
        info!("⏱️ [{:.2?}] ✅ Quantization to '{}' completed", t.elapsed(), level);
        io::stdout().flush().unwrap();
    }

    // ========== RESOLVER CAMINHO DE SAÍDA ==========
    let output_path = resolve_bin_path_safe(output_dir, base_name, level)?;
    info!("⏱️ [{:.2?}] 📁 Binary path resolved: {:?}", total_start.elapsed(), output_path);
    io::stdout().flush().unwrap();

    // Verificar espaço em disco disponível (teste simples)
    if let Some(parent) = output_path.parent() {
        check_disk_space(parent, estimated_file_size as u64)?;
    }

    // ========== CONSTRUIR ÍNDICE ANN SE SOLICITADO ==========
    let mut wrote_real_ann = false;
    if ann {
        let ann_start = Instant::now();
        let ann_file = output_path.with_extension("ann");
        wrote_real_ann = build_ann_index_safe(&data, dims, rows, seed, &ann_file)?;
        info!(
            "✅ [{:.2?}] ANN index {}: {:?}",
            ann_start.elapsed(),
            if wrote_real_ann { "created" } else { "stub written" },
            ann_file
        );
    }

    // ========== ESCREVER ARQUIVO BINÁRIO ==========
    let bin_start = std::time::Instant::now();
    write_bin_mmap_safe(&data, dims, rows, &output_path)?;
    info!("✅ [{:.2?}] BIN file written successfully", bin_start.elapsed());
    io::stdout().flush().unwrap();

    let total_time = total_start.elapsed();
    info!("🎯 Total prepare_bin_from_embeddings time: {:.2?}", total_time);
    io::stdout().flush().unwrap();

    Ok(output_path)
}

/// Normalização segura por chunks para evitar problemas de memória
fn normalize_rows_safe(data: &mut [f32], dims: usize) -> Result<(), String> {
    if data.len() % dims != 0 {
        return Err("Data length not divisible by dimensions".into());
    }

    let rows = data.len() / dims;
    info!("Normalizing {} rows with {} dimensions", rows, dims);

    data.par_chunks_mut(dims)
        .enumerate()
        .try_for_each(|(row_idx, row)| -> Result<(), String> {
            let mut sum = 0.0f32;

            if dims >= 8 {
                let chunks_8 = dims / 8;
                for i in 0..chunks_8 {
                    let start = i * 8;
                    let chunk = &row[start..start + 8];
                    let simd = f32x8::new([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                    let squared = simd * simd;
                    sum += squared.reduce_add();
                }
                for val in &row[chunks_8 * 8..] { sum += val * val; }
            } else {
                for val in row.iter() { sum += val * val; }
            }

            let norm = sum.sqrt();

            if norm == 0.0 {
                return Err(format!("Zero vector found at row {}", row_idx));
            }
            if !norm.is_finite() {
                return Err(format!("Invalid norm at row {}: {}", row_idx, norm));
            }
            if norm < 1e-10 {
                warn!("Very small norm at row {}: {}", row_idx, norm);
            }

            for val in row.iter_mut() {
                *val /= norm;
            }

            Ok(())
        })?;

    info!("Normalization completed successfully");
    Ok(())
}

/// Quantização segura com validações (in-place, mantém f32 no binário)
fn quantize_in_place_safe(data: &mut [f32], level: &str) -> Result<(), String> {
    info!("Quantizing {} elements to level {}", data.len(), level);

    match level {
        "f32" => Ok(()),
        "f16" => {
            data.par_iter_mut().try_for_each(|val| -> Result<(), String> {
                if !val.is_finite() {
                    return Err(format!("Invalid value in f16 quantization: {}", val));
                }
                let f16_val = half::f16::from_f32(*val);
                *val = f16_val.to_f32();
                if !val.is_finite() {
                    return Err("f16 quantization produced invalid value".into());
                }
                Ok(())
            })?;
            Ok(())
        },
        "f8" => {
            data.par_iter_mut().try_for_each(|val| -> Result<(), String> {
                if !val.is_finite() {
                    return Err(format!("Invalid value in f8 quantization: {}", val));
                }
                let clamped = val.clamp(-2.0, 2.0);
                let quantized = (clamped * 127.0 / 2.0).round() * 2.0 / 127.0;
                if !quantized.is_finite() {
                    return Err("f8 quantization produced invalid value".into());
                }
                *val = quantized;
                Ok(())
            })?;
            Ok(())
        },
        "f64" => {
            // Ainda não suportamos f64 no binário — manter f32
            Ok(())
        },
        _ => Err(format!("Unsupported quantization level: {}", level)),
    }
}

/// Construir índice ANN com validações de segurança
fn build_ann_index_safe(
    data: &[f32],
    dims: usize,
    rows: usize,
    seed: u64,
    ann_path: &Path
) -> Result<bool, String> {
    if data.len() != dims * rows {
        return Err("Data size mismatch for ANN index".into());
    }
    if dims < 8 {
        return Err(format!("ANN requires at least 8 dimensions, got {}", dims));
    }

    if let Some(parent) = ann_path.parent() {
        if !parent.exists() {
            create_dir_all(parent)
                .map_err(|e| format!("Failed to create ANN directory {}: {}", parent.display(), e))?;
        }
    }

    if should_use_exact_search(rows) {
        std::fs::write(ann_path, b"NSEEK-NO-ANN")
            .map_err(|e| format!("Failed to write ANN stub {}: {}", ann_path.display(), e))?;
        warn!("Dataset small ({} rows) — wrote ANN stub instead of full ANN index", rows);
        return Ok(false);
    }

    let nbits: usize = 8;
    let ann: Box<dyn AnnIndex> = build_state_of_art_ann_index(data, dims, rows, nbits, seed);

    if let Err(e) = ann.health_check() {
        return Err(format!("ANN index health check failed: {}", e));
    }

    ann.save(ann_path)
        .map_err(|e| format!("Failed to save ANN index to {}: {}", ann_path.display(), e))?;

    info!("ANN index saved successfully to: {:?}", ann_path);
    Ok(true)
}

/// Escrever arquivo binário com mmap seguro (formato: [u32 dims][u32 rows][f32...])
fn write_bin_mmap_safe(data: &[f32], dims: usize, rows: usize, path: &Path) -> Result<(), String> {
    if data.len() != dims * rows {
        return Err(format!(
            "Data size mismatch: expected {} elements ({}x{}), got {}",
            dims * rows, dims, rows, data.len()
        ));
    }

    let invalid_count = data.iter().filter(|&&x| !x.is_finite()).count();
    if invalid_count > 0 {
        return Err(format!("Cannot write {} invalid values to binary file", invalid_count));
    }

    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    let tmp_path = path.with_extension("tmp");
    if tmp_path.exists() {
        remove_file(&tmp_path).map_err(|e| format!("Failed to remove existing temp file {}: {}", tmp_path.display(), e))?;
    }
    if path.exists() {
        remove_file(path).map_err(|e| format!("Failed to remove existing file {}: {}", path.display(), e))?;
    }

    let data_bytes_len = data.len() * 4;
    let total_size = 8 + data_bytes_len;

    info!("Creating binary file: {} bytes", total_size);

    let file = OpenOptions::new()
        .read(true).write(true).create(true).truncate(true)
        .open(&tmp_path)
        .map_err(|e| format!("Failed to create temp file {}: {}", tmp_path.display(), e))?;

    file.set_len(total_size as u64).map_err(|e| format!("Failed to set file length: {}", e))?;

    let mut mmap = unsafe {
        MmapMut::map_mut(&file).map_err(|e| format!("Failed to memory map file: {}", e))?
    };

    // Header
    mmap[0..4].copy_from_slice(&(dims as u32).to_le_bytes());
    mmap[4..8].copy_from_slice(&(rows as u32).to_le_bytes());

    // Data
    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data_bytes_len)
    };
    mmap[8..8 + data_bytes_len].copy_from_slice(data_bytes);

    mmap.flush().map_err(|e| format!("Failed to flush memory map: {}", e))?;
    drop(mmap);

    let metadata = std::fs::metadata(&tmp_path).map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let actual_size = metadata.len();
    if actual_size != total_size as u64 {
        remove_file(&tmp_path).ok();
        return Err(format!("File size verification failed: expected {}, got {}", total_size, actual_size));
    }

    // Verify header
    let mut verification_file = std::fs::File::open(&tmp_path).map_err(|e| format!("Failed to open file for verification: {}", e))?;
    let mut header_buf = [0u8; 8];
    use std::io::Read as _;
    verification_file.read_exact(&mut header_buf).map_err(|e| format!("Failed to read header for verification: {}", e))?;
    let dims_read = u32::from_le_bytes(header_buf[0..4].try_into().unwrap()) as usize;
    let rows_read = u32::from_le_bytes(header_buf[4..8].try_into().unwrap()) as usize;
    if dims_read != dims || rows_read != rows {
        remove_file(&tmp_path).ok();
        return Err(format!("Header verification failed: expected {}x{}, got {}x{}", dims, rows, dims_read, rows_read));
    }

    rename(&tmp_path, path).map_err(|e| format!("Failed to move temp file to final location: {}", e))?;
    info!("Binary file written and verified successfully: {:?}", path);
    info!("📦 Final file size: {:.2}MB", actual_size as f64 / (1024.0 * 1024.0));
    Ok(())
}

/// Resolver caminho do binário com validações de segurança
pub fn resolve_bin_path_safe(
    output_dir: Option<&Path>,
    base_name: &str,
    level: &str,
) -> Result<PathBuf, String> {
    if base_name.trim().is_empty() {
        return Err("Base name cannot be empty".into());
    }
    if base_name.contains('/') || base_name.contains('\\') || base_name.contains("..") {
        return Err("Base name contains invalid characters".into());
    }
    match level {
        "f8" | "f16" | "f32" | "f64" => {},
        _ => return Err(format!("Invalid level for path: {}", level)),
    }

    let final_path: PathBuf = match output_dir {
        Some(dir) => {
            if !dir.exists() {
                debug!("Output directory doesn't exist, will create: {:?}", dir);
            } else if !dir.is_dir() {
                return Err(format!("Output path is not a directory: {:?}", dir));
            }
            dir.join(format!("{}_{}.bin", base_name, level))
        }
        None => {
            let home = dirs::home_dir().ok_or("Failed to get home directory")?;
            let nseek_dir = home.join(".nseek").join("indexes").join(base_name);
            nseek_dir.join(format!("{}_{}.bin", base_name, level))
        }
    };

    if let Some(parent) = final_path.parent() {
        if !parent.exists() {
            match create_dir_all(parent) {
                Ok(_) => { info!("Created directory: {:?}", parent); }
                Err(_e) => {
                    warn!("Failed to create directory {:?}, using fallback", parent);
                    let fallback = std::env::temp_dir().join("nseek_fallback").join(base_name);
                    create_dir_all(&fallback).map_err(|e| {
                        format!("Failed to create fallback directory {}: {}", fallback.display(), e)
                    })?;
                    return Ok(fallback.join(format!("{}.bin", level)));
                }
            }
        }
    }

    if let Some(parent) = final_path.parent() {
        if parent.exists() {
            let test_file = parent.join(".nseek_write_test");
            match std::fs::File::create(&test_file) {
                Ok(_) => { let _ = std::fs::remove_file(&test_file); }
                Err(_) => {
                    warn!("No write permission for {:?}, using fallback", parent);
                    let fallback = std::env::temp_dir().join("nseek_fallback").join(base_name);
                    create_dir_all(&fallback).map_err(|e| format!("Failed to create fallback directory: {}", e))?;
                    return Ok(fallback.join(format!("{}.bin", level)));
                }
            }
        }
    }

    Ok(final_path)
}

/// Verificar espaço em disco disponível (teste de escrita simples)
fn check_disk_space(path: &Path, required_bytes: u64) -> Result<(), String> {
    if !path.exists() {
        debug!("Directory doesn't exist yet: {:?}", path);
        return Ok(());
    }
    let test_file = path.join(".nseek_space_test");
    match std::fs::File::create(&test_file) {
        Ok(_) => {
            let _ = std::fs::remove_file(&test_file);
            if required_bytes > 10 * 1024 * 1024 * 1024 {
                warn!("Large file creation: {:.1}GB - ensure sufficient disk space",
                      required_bytes as f64 / (1024.0_f64.powi(3)));
            }
            Ok(())
        }
        Err(e) => Err(format!("Cannot write to directory {}: {}", path.display(), e))
    }
}

#[inline(always)]
fn chunked_len(len: usize) -> usize { len / 8 * 8 }
