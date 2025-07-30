use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::error::Error;
use memmap2::Mmap;
use bytemuck::{Pod, cast_slice, cast_slice_mut};

pub fn load_csv_bin<P: AsRef<Path>>(path: P) -> Result<crate::engine::SearchEngine, Box<dyn Error>> {
    let path = path.as_ref();
    let bin_path = path.with_extension("bin");

    if bin_path.exists() {
        return load_bin(&bin_path);
    } else {
        let engine = load_csv(path)?;
        save_bin(engine.all_vectors(), &bin_path)?;
        Ok(engine)
    }
}

pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<crate::engine::SearchEngine, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut engine = crate::engine::SearchEngine::new();

    for line in reader.lines() {
        let line = line?;
        let vec: Vec<f32> = line.split(',').map(|s| s.trim().parse().unwrap_or(0.0)).collect();
        engine.add(vec);
    }

    Ok(engine)
}

pub fn save_bin(vectors: &Vec<Vec<f32>>, path: &Path) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    for vec in vectors {
        let bytes = cast_slice::<f32, u8>(&vec);
        file.write_all(bytes)?;
    }
    Ok(())
}

pub fn load_bin(path: &Path) -> Result<crate::engine::SearchEngine, Box<dyn Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let total_len = mmap.len();
    let dims_guess = 768; // fallback se não souber
    let float_count = total_len / std::mem::size_of::<f32>();

    let flat: &[f32] = cast_slice(&mmap);
    let dims = dims_guess;
    let rows = float_count / dims;

    let mut engine = crate::engine::SearchEngine::new();
    for i in 0..rows {
        let start = i * dims;
        let end = start + dims;
        engine.add(flat[start..end].to_vec());
    }
    Ok(engine)
}
