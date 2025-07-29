use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

pub fn load_csv_f32(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let row: Vec<f32> = line.split(',')
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();
        data.push(row);
    }

    Ok(data)
}
