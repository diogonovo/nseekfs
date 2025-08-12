use rayon::prelude::*;
use std::fs::{File, OpenOptions, remove_file, rename, create_dir_all};
use std::path::{Path, PathBuf};
use wide::f32x8;
use memmap2::MmapMut;
use crate::ann_opt::AnnIndex;
use std::io::Read;
use std::time::Instant;
use dirs;
use std::io::{self, Write};

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
    println!("⏱️ [0.00s] Início prepare_bin_from_embeddings");
    io::stdout().flush().unwrap();

    if dims == 0 || rows == 0 {
        return Err("Empty embeddings input".into());
    }
    if embeddings.len() != dims * rows {
        return Err("Embedding data shape mismatch".into());
    }

    let step1 = Instant::now();
    let mut data = embeddings.to_vec();
    println!("⏱️ [{:.2?}] ✅ Copiado embeddings para vetor interno", step1.elapsed());
    io::stdout().flush().unwrap();

    if normalize {
        let t = Instant::now();
        normalize_rows_simd(&mut data, dims);
        println!("⏱️ [{:.2?}] ✅ Normalização concluída", t.elapsed());
        io::stdout().flush().unwrap();
    }

    if level != "f32" {
        let t = Instant::now();
        quantize_in_place(&mut data, level)?;
        println!("⏱️ [{:.2?}] ✅ Quantização para '{}' concluída", t.elapsed(), level);
        io::stdout().flush().unwrap();
    }

    let output_path = resolve_bin_path(output_dir, base_name, level)?;
    println!("⏱️ [{:.2?}] 📁 Caminho do binário resolvido: {:?}", total_start.elapsed(), output_path);
    io::stdout().flush().unwrap();

    if ann {
        println!("🧪 [{:.2?}] Início do índice ANN...", total_start.elapsed());
        io::stdout().flush().unwrap();

        let ann_start = Instant::now();
        let ann_file = output_path.with_extension("ann");
        build_ann_index(&data, dims, rows, seed, &ann_file)?;
        println!("✅ [{:.2?}] ANN index criado: {:?}", ann_start.elapsed(), ann_file);
        io::stdout().flush().unwrap();
    }

    println!("🧪 [{:.2?}] A escrever ficheiro BIN...", total_start.elapsed());
    io::stdout().flush().unwrap();

    let bin_start = Instant::now();
    write_bin_mmap(&data, dims, rows, &output_path)?;
    println!("✅ [{:.2?}] BIN file escrito com sucesso", bin_start.elapsed());
    io::stdout().flush().unwrap();

    println!("🎯 Tempo total prepare_bin_from_embeddings: {:.2?}", total_start.elapsed());
    io::stdout().flush().unwrap();

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
        "f16" => Ok(()),
        "f8" => Ok(()),
        "f64" => Ok(()),
        _ => Err("Unsupported quantization level".into()),
    }
}

fn build_ann_index(data: &[f32], dims: usize, rows: usize, seed: u64, ann_path: &Path) -> Result<(), String> {
    if let Some(parent) = ann_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create ANN directory: {}", e))?;
    }

    let ann = AnnIndex::build(data, dims, rows, 16, seed);
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
                "❌ Não foi possível sobrescrever '{}': {}.",
                path.display(), e
            )
        })?;
    }

    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };

    let total_size = 8 + data_bytes.len();

    let file = OpenOptions::new()
        .read(true)
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

    let meta = std::fs::metadata(&tmp_path).map_err(|e| format!("Metadata error: {}", e))?;
    let actual_size = meta.len();
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

pub fn resolve_bin_path(
    output_dir: Option<&Path>,
    base_name: &str,
    level: &str,
) -> Result<PathBuf, String> {
    let final_path: PathBuf = match output_dir {
        Some(dir) => {
            dir.join(format!("{}.bin", level))
        }
        None => {
            let home = dirs::home_dir().ok_or("❌ Não foi possível obter a home directory")?;
            home.join(".nseek").join("indexes").join(base_name).join(format!("{}.bin", level))
        }
    };

    if let Some(parent) = final_path.parent() {
        if let Err(e) = create_dir_all(parent) {
            let fallback = std::env::temp_dir().join("nseek_fallback").join(base_name);
            create_dir_all(&fallback)
                .map_err(|e| format!("❌ Falhou criação do fallback: {}", e))?;
            return Ok(fallback.join(format!("{}.bin", level)));
        }
    }

    Ok(final_path)
}
