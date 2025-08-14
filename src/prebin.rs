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

// extras
use bytemuck::{cast_slice, Pod, Zeroable};
use std::cmp::min;
use half;

// ========== CONSTANTES DE SEGURANÇA ==========
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024 * 1024; // 100GB max
const CHUNK_SIZE: usize = 1000; // Processar normalização em chunks

// ========= FORMATO DO BINÁRIO =========
//
// Header (fixo, little-endian):
//   magic    [u8;4]  = b"NSEK"
//   version  u32     = 1
//   dims     u32
//   rows     u32
//   level    u32     (1=f8,2=f16,3=f32,4=f64)
//   flags    u32     (bit0: normalized, bit1: has_ann)
//   reserved [u8;32]
//
// Payload por nível:
//   f32: [rows*dims] f32
//   f16: [rows*dims] u16   (half bits)
//   f8 : [rows*dims] u8 + [rows] f32 (scales por vetor)
//
// ======================================

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
enum BinLevel {
    F8  = 1,
    F16 = 2,
    F32 = 3,
    F64 = 4, // reservado (escrevemos como f32)
}
impl BinLevel {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f8"  => Ok(BinLevel::F8),
            "f16" => Ok(BinLevel::F16),
            "f32" => Ok(BinLevel::F32),
            "f64" => Ok(BinLevel::F64),
            _ => Err(format!("Invalid level '{}'", s)),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct BinHeader {
    magic:    [u8; 4],   // b"NSEK"
    version:  u32,       // 1
    dims:     u32,
    rows:     u32,
    level:    u32,       // BinLevel as u32
    flags:    u32,       // bit0: normalized, bit1: has_ann
    reserved: [u8; 32],  // padding p/ futuro
}

const BIN_MAGIC: [u8;4] = *b"NSEK";
const BIN_VERSION: u32 = 1;

#[derive(Debug)]
enum Payload {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8 { codes: Vec<u8>, scales: Vec<f32> },
}

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

    // ======= validações de entrada =======
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

    let level_enum = BinLevel::from_str(level)?;

    // amostra p/ invalidos
    let sample_size = min(embeddings.len(), 10000);
    let invalid_count = embeddings[..sample_size].iter()
        .filter(|&&x| !x.is_finite())
        .count();
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

    // ======= estimativa de tamanho =======
    let header_size = std::mem::size_of::<BinHeader>();
    let estimated_file_size = match level_enum {
        BinLevel::F32 => header_size + dims * rows * 4,
        BinLevel::F16 => header_size + dims * rows * 2,
        BinLevel::F8  => header_size + dims * rows * 1 + rows * 4,
        BinLevel::F64 => header_size + dims * rows * 8, // reservado
    } as u64;

    info!("📊 Estimated file size: {:.1}MB", estimated_file_size as f64 / (1024.0 * 1024.0));
    if estimated_file_size > MAX_FILE_SIZE {
        return Err(format!(
            "Estimated file size too large: {:.1}GB (max: {:.1}GB)",
            estimated_file_size as f64 / (1024.0_f64.powi(3)),
            MAX_FILE_SIZE as f64 / (1024.0_f64.powi(3))
        ));
    }

    // ======= processamento =======
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

    // payload por nível
    let payload = make_payload(data, dims, rows, level_enum)?;

    // ======= caminho de saída =======
    let output_path = resolve_bin_path_safe(output_dir, base_name, level)?;
    info!("⏱️ [{:.2?}] 📁 Binary path resolved: {:?}", total_start.elapsed(), output_path);
    io::stdout().flush().unwrap();

    // espaço em disco (teste simples)
    if let Some(parent) = output_path.parent() {
        check_disk_space(parent, estimated_file_size)?;
    }

    // ======= ANN opcional =======
    let mut has_ann_flag = false;
    if ann {
        info!("🧪 [{:.2?}] Starting ANN index construction...", total_start.elapsed());
        io::stdout().flush().unwrap();

        // reconstruir f32 para o ANN consoante o payload
        let data_for_ann: Vec<f32> = match &payload {
            Payload::F32(v) => v.clone(),
            Payload::F16(v) => v.par_iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect(),
            Payload::Q8 { codes, scales } => {
                let mut out = vec![0f32; dims * rows];
                out.par_chunks_mut(dims)
                    .enumerate()
                    .for_each(|(ri, out_row)| {
                        let scale = scales[ri];
                        let base = ri * dims;
                        for j in 0..dims {
                            let code = codes[base + j] as i32 - 127;
                            out_row[j] = (code as f32 / 127.0) * scale;
                        }
                    });
                out
            }
        };

        let ann_start = Instant::now();
        let ann_file = output_path.with_extension("ann");
        has_ann_flag = build_ann_index_safe(&data_for_ann, dims, rows, seed, &ann_file)?;
        info!("✅ [{:.2?}] ANN index {}: {:?}",
            ann_start.elapsed(),
            if has_ann_flag { "created" } else { "stub written" }, ann_file);
        io::stdout().flush().unwrap();
    }

    // ======= escrever BIN (uma vez) =======
    info!("🧪 [{:.2?}] Writing BIN file...", total_start.elapsed());
    io::stdout().flush().unwrap();

    let bin_start = Instant::now();
    write_bin_mmap_safe(
        &payload,
        dims,
        rows,
        level_enum,
        normalize,
        has_ann_flag,
        &output_path
    )?;
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
            // Calcular norma usando SIMD quando possível
            let mut sum = 0.0f32;

            if dims >= 8 {
                let chunks_8 = dims / 8;
                for i in 0..chunks_8 {
                    let start = i * 8;
                    let chunk = &row[start..start + 8];
                    let simd = f32x8::from([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                    let squared = simd * simd;
                    sum += squared.reduce_add();
                }
                // Restantes elementos
                for val in &row[chunks_8 * 8..] {
                    sum += val * val;
                }
            } else {
                // Fallback escalar
                for val in row.iter() {
                    sum += val * val;
                }
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

/// Gerar payload conforme o nível (quantização real p/ f16 e f8)
fn make_payload(mut data: Vec<f32>, dims: usize, rows: usize, level: BinLevel) -> Result<Payload, String> {
    match level {
        BinLevel::F32 => Ok(Payload::F32(data)),
        BinLevel::F16 => {
            let mut out: Vec<u16> = Vec::with_capacity(data.len());
            out.resize(data.len(), 0);
            out.par_iter_mut().zip(data.par_iter()).try_for_each(|(dst, src)| -> Result<(), String> {
                if !src.is_finite() { return Err(format!("Invalid value in f16 quantization: {}", src)); }
                let h = half::f16::from_f32(*src);
                *dst = h.to_bits();
                Ok(())
            })?;
            Ok(Payload::F16(out))
        }
        BinLevel::F8 => {
            // Q8 por vetor: code = round((x/scale)*127), scale = max(|x|)
            let mut codes = vec![0u8; data.len()];
            let mut scales = vec![0f32; rows];

            data.par_chunks(dims)
                .zip(codes.par_chunks_mut(dims))
                .enumerate()
                .try_for_each(|(ri, (row, out_row))| -> Result<(), String> {
                    // escala por linha
                    let mut max_abs = 0f32;
                    for &v in row.iter() {
                        if !v.is_finite() { return Err("Invalid value in f8 quantization".into()); }
                        let a = v.abs();
                        if a > max_abs { max_abs = a; }
                    }
                    let scale = if max_abs < 1e-12 { 1e-12 } else { max_abs };
                    scales[ri] = scale;

                    // quantizar
                    for (j, &v) in row.iter().enumerate() {
                        let q = (v / scale * 127.0).round();
                        let clamped = q.clamp(-127.0, 127.0) as i32;
                        // mapear [-127..127] -> [0..254]
                        out_row[j] = (clamped + 127) as u8;
                    }
                    Ok(())
                })?;

            Ok(Payload::Q8 { codes, scales })
        }
        BinLevel::F64 => {
            // ainda não suportado — escreve-se como f32 para já
            Ok(Payload::F32(data))
        }
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

    // Se muito pequeno, grava stub e não cria ANN real
    if should_use_exact_search(rows) {
        std::fs::write(ann_path, b"NSEEK-NO-ANN")
            .map_err(|e| format!("Failed to write ANN stub {}: {}", ann_path.display(), e))?;
        warn!("Dataset small ({} rows) — wrote ANN stub instead of full ANN index", rows);
        return Ok(false);
    }

    let nbits: usize = 8; // parâmetro legado/ignorado internamente
    let ann: Box<dyn AnnIndex> = build_state_of_art_ann_index(data, dims, rows, nbits, seed);

    if let Err(e) = ann.health_check() {
        return Err(format!("ANN index health check failed: {}", e));
    }

    ann.save(ann_path)
        .map_err(|e| format!("Failed to save ANN index to {}: {}", ann_path.display(), e))?;

    info!("ANN index saved successfully to: {:?}", ann_path);
    Ok(true)
}

/// Escrever arquivo binário com mmap seguro (header + payload por nível)
fn write_bin_mmap_safe(
    payload: &Payload,
    dims: usize,
    rows: usize,
    level: BinLevel,
    normalized: bool,
    has_ann: bool,
    path: &Path
) -> Result<(), String> {
    // Criar diretório pai se necessário
    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|e| {
            format!("Failed to create directory {}: {}", parent.display(), e)
        })?;
    }

    // arquivo temporário para operação atômica
    let tmp_path = path.with_extension("tmp");
    if tmp_path.exists() { remove_file(&tmp_path).ok(); }
    if path.exists()     { remove_file(path).ok(); }

    // calcular tamanho total
    let header_size = std::mem::size_of::<BinHeader>();
    let data_bytes_len: usize = match payload {
        Payload::F32(v) => v.len() * 4,
        Payload::F16(v) => v.len() * 2,
        Payload::Q8 { codes, scales } => {
            if scales.len() != rows { return Err("Q8 scales length mismatch".into()); }
            codes.len() + rows * 4
        }
    };
    let total_size = header_size + data_bytes_len;

    info!("Creating binary file ({} bytes): {:?}", total_size, path);

    // Criar e mapear arquivo
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| format!("Failed to create temp file {}: {}", tmp_path.display(), e))?;

    file.set_len(total_size as u64).map_err(|e| format!("Failed to set file length: {}", e))?;

    let mut mmap = unsafe {
        MmapMut::map_mut(&file).map_err(|e| format!("Failed to memory map file: {}", e))?
    };

    // Header
    let flags: u32 = (if normalized {1} else {0}) | (if has_ann {2} else {0});
    let header = BinHeader {
        magic: BIN_MAGIC,
        version: BIN_VERSION,
        dims: dims as u32,
        rows: rows as u32,
        level: level as u32,
        flags,
        reserved: [0u8;32],
    };
    mmap[0..header_size].copy_from_slice(cast_slice(&[header]));

    // Payload
    let mut off = header_size;
    match payload {
        Payload::F32(v) => {
            let bytes: &[u8] = cast_slice(v.as_slice());
            mmap[off .. off + bytes.len()].copy_from_slice(bytes);
        }
        Payload::F16(v) => {
            let bytes: &[u8] = cast_slice(v.as_slice());
            mmap[off .. off + bytes.len()].copy_from_slice(bytes);
        }
        Payload::Q8 { codes, scales } => {
            // codes
            mmap[off .. off + codes.len()].copy_from_slice(codes.as_slice());
            off += codes.len();
            // scales
            let bytes: &[u8] = cast_slice(scales.as_slice());
            mmap[off .. off + bytes.len()].copy_from_slice(bytes);
        }
    }

    // Flush
    mmap.flush().map_err(|e| format!("Failed to flush memory map: {}", e))?;
    drop(mmap);

    // Verificar tamanho
    let actual_size = std::fs::metadata(&tmp_path).map_err(|e| format!("Failed to get file metadata: {}", e))?.len() as usize;
    if actual_size != total_size {
        remove_file(&tmp_path).ok();
        return Err(format!("File size verification failed: expected {}, got {}", total_size, actual_size));
    }

    // Mover arquivo temporário para destino
    rename(&tmp_path, path).map_err(|e| format!("Failed to move temp file to final location: {}", e))?;

    info!("Binary file written and verified successfully: {:?} ({:.2}MB)", path, actual_size as f64 / (1024.0 * 1024.0));
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
            let nseek_dir = home.join(".nseek").join("indexes");
            nseek_dir.join(format!("{}_{}.bin", base_name, level))
        }
    };

    // criar diretório pai
    if let Some(parent) = final_path.parent() {
        if !parent.exists() {
            create_dir_all(parent).map_err(|e| {
                format!("Failed to create directory {}: {}", parent.display(), e)
            })?;
        }
    }

    Ok(final_path)
}

/// Verificar espaço em disco disponível (básico)
fn check_disk_space(path: &Path, required_bytes: u64) -> Result<(), String> {
    if !path.exists() {
        debug!("Directory doesn't exist yet: {:?}", path);
        return Ok(()); // Assumir OK
    }
    let test_file = path.join(".nseek_space_test");
    match std::fs::File::create(&test_file) {
        Ok(_) => {
            let _ = std::fs::remove_file(&test_file);
            if required_bytes > 10 * 1024 * 1024 * 1024 {
                warn!(
                    "Large file creation: {:.1}GB - ensure sufficient disk space",
                    required_bytes as f64 / (1024.0_f64.powi(3))
                );
            }
            Ok(())
        }
        Err(e) => Err(format!("Cannot write to directory {}: {}", path.display(), e))
    }
}

#[inline(always)]
fn chunked_len(len: usize) -> usize {
    len / 8 * 8
}
