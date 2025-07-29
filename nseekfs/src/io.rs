use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use csv::ReaderBuilder;

pub fn load_csv_f32(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(BufReader::new(file));
    let mut vectors = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row = record.iter()
            .map(|x| x.parse::<f32>())
            .collect::<Result<Vec<f32>, _>>()?;
        vectors.push(row);
    }

    Ok(vectors)
}