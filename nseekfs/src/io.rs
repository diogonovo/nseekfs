// ===========================
// src/io.rs
// ===========================

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::utils::{read_csv_matrix, normalize_matrix};

pub fn write_bin_file(
    csv_path: &str,
    normalize: bool,
    force: bool,
    use_ann: bool,
) -> Result<String, String> {
    let path = Path::new(csv_path);
    let stem = if path.extension().is_none() {
        path.to_str().ok_or("Invalid base path")?
    } else {
        path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or("Invalid base path")?
    };


    let bin_filename = if use_ann {
        format!("{}_f32_ann.bin", stem)
    } else {
        format!("{}_f32.bin", stem)
    };

    let bin_path = path.with_file_name(bin_filename);

    if bin_path.exists() && !force {
        return Ok(bin_path.to_string_lossy().into_owned());
    }

    let mut data = read_csv_matrix(csv_path)?;
    if normalize {
        normalize_matrix(&mut data);
    }

    write_bin_data(&data, &bin_path)?;

    Ok(bin_path.to_string_lossy().into_owned())
}

pub fn write_bin_from_embeddings(
    data: &mut Vec<Vec<f32>>,
    base_path: &str,
    precision: &str,
    normalize: bool,
    use_ann: bool,
) -> Result<String, String> {
    if data.is_empty() || data[0].is_empty() {
        return Err("Embeddings vazios ou mal formatados".into());
    }

    if normalize {
        normalize_matrix(data);
    }

    let suffix = if use_ann {
        format!("{}_ann.bin", precision)
    } else {
        format!("{}.bin", precision)
    };

    let bin_path: PathBuf = Path::new(&format!("{}_{}", base_path, suffix)).to_path_buf();

    write_bin_data(data, &bin_path)?;
    Ok(bin_path.to_string_lossy().into_owned())
}

fn write_bin_data(data: &Vec<Vec<f32>>, bin_path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(bin_path)
        .map_err(|e| format!("❌ Não consegui abrir {} para escrita: {}", bin_path.display(), e))?;

    let mut writer = BufWriter::new(file);

    let dims = data[0].len() as u32;
    let rows = data.len() as u32;

    writer
        .write_all(&dims.to_le_bytes())
        .map_err(|e| format!("Erro ao escrever dims: {}", e))?;
    writer
        .write_all(&rows.to_le_bytes())
        .map_err(|e| format!("Erro ao escrever rows: {}", e))?;

    for row in data {
        if row.len() != dims as usize {
            return Err("Row size mismatch".into());
        }
        let bytes = bytemuck::cast_slice::<f32, u8>(&row);
        writer
            .write_all(bytes)
            .map_err(|e| format!("Erro ao escrever vetor: {}", e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Erro ao finalizar escrita: {}", e))?;

    Ok(())
}
