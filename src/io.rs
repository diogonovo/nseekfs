use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::utils::vector::{normalize_matrix, read_csv_matrix, validate_dimensions};
use log::{error, info};

pub fn write_bin_file(
    csv_path: &str,
    normalize: bool,
    force: bool,
    ann: bool,
) -> Result<String, String> {
    let path = Path::new(csv_path);

    let stem = if path.extension().is_none() {
        path.to_str().ok_or("Invalid base path")?
    } else {
        path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or("Invalid base name")?
    };

    let bin_filename = if ann {
        format!("{}_f32_ann.bin", stem)
    } else {
        format!("{}_f32.bin", stem)
    };

    let bin_path = path.with_file_name(bin_filename);

    if bin_path.exists() && !force {
        info!(
            "Binary file already exists → Skipping write: {}",
            bin_path.display()
        );
        return Ok(bin_path.to_string_lossy().into_owned());
    }

    info!("Reading CSV file: {}", csv_path);
    let mut data = read_csv_matrix(csv_path).map_err(|e| {
        error!("Failed to read CSV: {}", e);
        e
    })?;

    if normalize {
        info!("Normalizing matrix...");
        normalize_matrix(&mut data);
    }

    info!("Writing binary file to: {}", bin_path.display());
    write_bin_data(&data, &bin_path)?;
    Ok(bin_path.to_string_lossy().into_owned())
}

pub fn write_bin_from_embeddings(
    data: &mut Vec<Vec<f32>>,
    base_path: &str,
    precision: &str,
    normalize: bool,
    ann: bool,
) -> Result<String, String> {
    if data.is_empty() || data[0].is_empty() {
        error!("Received empty or malformed embeddings");
        return Err("Empty or malformed embeddings".into());
    }

    let _dims = validate_dimensions(data).map_err(|e| {
        error!("Embedding dimension error: {}", e);
        format!("Embedding dimension error: {}", e)
    })?;

    if normalize {
        info!("Normalizing embeddings matrix...");
        normalize_matrix(data);
    }

    let suffix = if ann {
        format!("{}_ann.bin", precision)
    } else {
        format!("{}.bin", precision)
    };

    let bin_path = Path::new(&format!("{}_{}", base_path, suffix)).to_path_buf();
    info!("Writing in-memory embeddings to: {}", bin_path.display());
    write_bin_data(data, &bin_path)?;
    Ok(bin_path.to_string_lossy().into_owned())
}

fn write_bin_data(data: &Vec<Vec<f32>>, bin_path: &Path) -> Result<(), String> {
    if data.is_empty() || data[0].is_empty() {
        error!("Input vectors are invalid or empty");
        return Err("Invalid input vectors".into());
    }

    let dims = data[0].len() as u32;
    let rows = data.len() as u32;

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(bin_path)
        .map_err(|e| {
            error!("Failed to open {} for writing: {}", bin_path.display(), e);
            format!("Failed to open {} for writing: {}", bin_path.display(), e)
        })?;

    let mut writer = BufWriter::new(file);

    writer
        .write_all(&dims.to_le_bytes())
        .map_err(|e| format!("Error writing header (dims): {}", e))?;

    writer
        .write_all(&rows.to_le_bytes())
        .map_err(|e| format!("Error writing header (rows): {}", e))?;

    for (i, row) in data.iter().enumerate() {
        if row.len() != dims as usize {
            error!("Line {} has incorrect dimensions", i);
            return Err(format!("Row {} has incorrect dimensions", i));
        }


        let bytes = bytemuck::cast_slice::<f32, u8>(row);
        writer
            .write_all(bytes)
            .map_err(|e| format!("Error writing vector {}: {}", i, e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Error finishing writing: {}", e))?;

    info!("Successfully wrote binary file: {}", bin_path.display());
    Ok(())
}
