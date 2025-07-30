use memmap2::Mmap;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

/// Parser CSV paralelo otimizado
pub fn load_csv_f32_flat_parallel(path: &str) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    let parsed: Vec<Vec<f32>> = lines
        .par_iter()
        .map(|line| {
            line.split(',')
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect::<Vec<f32>>()
        })
        .collect();

    let rows = parsed.len();
    let dims = parsed[0].len();
    let mut data = Vec::with_capacity(rows * dims);
    for row in parsed {
        data.extend_from_slice(&row);
    }
    Ok((data, rows, dims))
}

/// Escreve binário otimizado
pub fn write_bin_file(path: &str, data: &[f32], rows: usize, dims: usize) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    file.write_all(&(rows as u32).to_le_bytes())?;
    file.write_all(&(dims as u32).to_le_bytes())?;
    file.write_all(&[0u8; 4])?; // reservado
    let bytes = bytemuck::cast_slice(data);
    file.write_all(bytes)?;
    Ok(())
}

/// Lê binário usando mmap
pub fn load_bin_mmap(path: &str) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let rows = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
    let dims = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
    let data_bytes = &mmap[12..];
    let floats: &[f32] = bytemuck::cast_slice(data_bytes);
    Ok((floats.to_vec(), rows, dims))
}
