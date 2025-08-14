use std::fs::{File, read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use memmap2::Mmap;
use crate::ann_opt::AnnIndex;
use crate::utils::vector::{compute_similarity, SimilarityMetric};
use log::{info, warn};

const MAX_DIMENSIONS: usize = 10000;
const MAX_VECTORS: usize = 100_000_000;
const MIN_FILE_SIZE_LEGACY: usize = 8; // legacy header
const MAX_TOP_K: usize = 100000;
const MEMORY_WARNING_THRESHOLD: usize = 1024 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct EngineMetrics {
    query_count: AtomicUsize,
    total_query_time_ms: AtomicUsize,
    ann_queries: AtomicUsize,
    exact_queries: AtomicUsize,
    simd_queries: AtomicUsize,
    scalar_queries: AtomicUsize,
}
impl EngineMetrics {
    pub fn record_query(&self, duration_ms: u64, used_ann: bool, used_simd: bool) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_query_time_ms.fetch_add(duration_ms as usize, Ordering::Relaxed);
        if used_ann { self.ann_queries.fetch_add(1, Ordering::Relaxed); }
        else { self.exact_queries.fetch_add(1, Ordering::Relaxed); }
        if used_simd { self.simd_queries.fetch_add(1, Ordering::Relaxed); }
        else { self.scalar_queries.fetch_add(1, Ordering::Relaxed); }
    }
    pub fn get_stats(&self) -> (usize, f64, usize, usize, usize, usize) {
        let q = self.query_count.load(Ordering::Relaxed);
        let t = self.total_query_time_ms.load(Ordering::Relaxed);
        let avg = if q>0 { t as f64 / q as f64 } else { 0.0 };
        (q, avg,
         self.ann_queries.load(Ordering::Relaxed),
         self.exact_queries.load(Ordering::Relaxed),
         self.simd_queries.load(Ordering::Relaxed),
         self.scalar_queries.load(Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct QueryResult {
    pub results: Vec<(usize, f32)>,
    pub query_time_ms: f64,
    pub method_used: String,
    pub candidates_generated: usize,
    pub simd_used: bool,
}

#[derive(Clone)]
pub struct Engine {
    pub vectors: Arc<[f32]>,
    pub dims: usize,
    pub rows: usize,
    pub ann: bool,
    pub ann_index: Option<Arc<dyn AnnIndex>>,
    creation_time: Instant,
    metrics: Arc<EngineMetrics>,
    file_path: Option<String>,
}
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ElemType { F32=0, F16=1, F8=2 }

impl Engine {
    pub fn from_bin<P: AsRef<Path>>(path: P, ann: bool) -> std::io::Result<Self> {
        let start_time = Instant::now();
        let path_ref = path.as_ref();
        if !path_ref.exists() { return Err(err("Binary file not found")); }
        if !path_ref.is_file() { return Err(err("Path is not a file")); }

        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < MIN_FILE_SIZE_LEGACY { return Err(err("Binary file too small")); }

        // Detect format
        let (dims, rows, elem, data_offset) = if mmap.len() >= 8 && &mmap[0..8] == b"NSEEKBIN" {
            // new format
            let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
            if version < 2 { return Err(err("Unsupported NSEEKBIN version")); }
            let elem = match mmap[12] { 0 => ElemType::F32, 1 => ElemType::F16, 2 => ElemType::F8, _ => ElemType::F32 };
            let dims = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
            let rows = u32::from_le_bytes(mmap[20..24].try_into().unwrap()) as usize;
            (dims, rows, elem, 24usize)
        } else {
            // legacy: [u32 dims][u32 rows] then f32
            let dims = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
            let rows = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
            (dims, rows, ElemType::F32, 8usize)
        };

        // validations
        if dims == 0 || dims > MAX_DIMENSIONS { return Err(err("Invalid dims")); }
        if rows == 0 || rows > MAX_VECTORS { return Err(err("Invalid rows")); }

        let elem_size = match elem { ElemType::F32=>4, ElemType::F16=>2, ElemType::F8=>1 };
        let expected = data_offset + dims*rows*elem_size;
        if expected != mmap.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Data size mismatch: expected {}, got {}", expected, mmap.len())));
        }

        // load into f32 RAM
        let data_slice = &mmap[data_offset..];
        let vec_f32: Vec<f32> = match elem {
            ElemType::F32 => {
                data_slice.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            ElemType::F16 => {
                data_slice.chunks_exact(2)
                    .map(|c| {
                        let u = u16::from_le_bytes(c.try_into().unwrap());
                        half::f16::from_bits(u).to_f32()
                    })
                    .collect()
            }
            ElemType::F8 => {
                data_slice.iter().map(|&b| {
                    let q = b as i8 as f32;
                    (q * 2.0) / 127.0
                }).collect()
            }
        };

        // sample sanity
        let sample = vec_f32.iter().step_by((vec_f32.len()/1000).max(1)).take(1000);
        let mut invalid = 0;
        for v in sample { if !v.is_finite() { invalid += 1; } }
        if invalid > 0 { warn!("Vector data has {} invalid values (sample) – continuing", invalid); }

        let vectors = Arc::from(vec_f32);

        info!("✅ Loaded binary index: dims={} rows={} elem={:?} ANN={} path={:?} size={:.1}MB",
              dims, rows, elem, ann, path_ref, mmap.len() as f64 / (1024.0*1024.0));

        // ANN handling (no loader yet)
        let ann_index = if ann {
            let ann_path = path_ref.with_extension("ann");
            if ann_path.exists() {
                match read(&ann_path) {
                    Ok(bytes) => {
                        if bytes.as_slice() == b"NSEEK-NO-ANN" {
                            warn!("ANN stub found at {:?} (exact search will be used).", ann_path);
                            None
                        } else {
                            warn!("ANN file {:?} present but loading is not implemented yet; ignoring.", ann_path);
                            None
                        }
                    }
                    Err(e) => { warn!("Failed to read ANN file {:?}: {}", ann_path, e); None }
                }
            } else {
                warn!("ANN requested but file {:?} not found", ann_path);
                None
            }
        } else { None };

        let engine = Self {
            vectors, dims, rows, ann, ann_index,
            creation_time: start_time,
            metrics: Arc::new(EngineMetrics::default()),
            file_path: Some(path_ref.to_string_lossy().to_string()),
        };
        info!("Engine loaded in {:.2}s", start_time.elapsed().as_secs_f64());
        Ok(engine)
    }

    pub fn save_to_bin<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        let path = path.as_ref();
        let mut file = File::create(path)?;
        // write new format (NSEEKBIN v2 f32)
        file.write_all(b"NSEEKBIN")?;
        file.write_all(&2u32.to_le_bytes())?;
        file.write_all(&[0u8,0,0,0])?; // elem=f32, padding
        file.write_all(&(self.dims as u32).to_le_bytes())?;
        file.write_all(&(self.rows as u32).to_le_bytes())?;
        for &v in self.vectors.iter() { file.write_all(&v.to_le_bytes())?; }
        file.flush()?;
        info!("Engine saved to {:?}: {}x{} vectors", path, self.rows, self.dims);
        Ok(())
    }

    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows { return None; }
        let s = idx*self.dims; let e = s + self.dims;
        Some(&self.vectors[s..e])
    }

    pub fn query_unified(&self, query: &[f32], k: usize, force_method: Option<&str>) -> Result<QueryResult,String> {
        if query.len() != self.dims { return Err(format!("Query dimension mismatch: expected {}, got {}", self.dims, query.len())); }
        if k==0 { return Ok(QueryResult{results:vec![],query_time_ms:0.0,method_used:"none".into(),candidates_generated:0,simd_used:false}); }
        if k>MAX_TOP_K { return Err(format!("k too large: {} (max: {})", k, MAX_TOP_K)); }

        let start = Instant::now();
        let simd_used = query.len() >= 64;

        let method = match force_method {
            Some("ann") => { if self.ann_index.is_none() { return Err("ANN requested but not available".into()); } "ann" }
            Some("exact") => "exact",
            Some(other) => return Err(format!("Invalid method: {}", other)),
            None => if self.ann_index.is_some() && self.rows>1000 { "ann" } else { "exact" }
        };

        let (results, cands) = match method {
            "ann" => {
                let ann_index = self.ann_index.as_ref().unwrap();
                let candidates = ann_index.get_candidates(query);
                let similarity = SimilarityMetric::Cosine;
                let ranked = self.rank_candidates(query, &candidates, k, &similarity)?;
                (ranked, candidates.len())
            }
            "exact" => {
                let similarity = SimilarityMetric::Cosine;
                let r = self.top_k_query_with_similarity(query, k, &similarity)?;
                (r, self.rows)
            }
            _ => unreachable!()
        };

        let ms = start.elapsed().as_secs_f64()*1000.0;
        self.metrics.record_query(ms as u64, method=="ann", simd_used);

        Ok(QueryResult{
            results, query_time_ms: ms, method_used: method.to_string(),
            candidates_generated: cands, simd_used
        })
    }

    fn rank_candidates(&self, query:&[f32], candidates:&[usize], k:usize, sim:&SimilarityMetric) -> Result<Vec<(usize,f32)>,String>{
        if candidates.is_empty() { return Ok(vec![]); }
        let mut scored = Vec::with_capacity(candidates.len());
        for &idx in candidates {
            if let Some(v) = self.get_vector(idx) {
                let s = compute_similarity(query, v, sim);
                scored.push((idx, s));
            }
        }
        scored.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    pub fn query_ann_with_timing(&self, query:&[f32], k:usize) -> Result<(Vec<(usize,f32)>, f64), String> {
        let start = Instant::now();
        if self.ann_index.is_none() { return Err("ANN not available".into()); }
        let ann_index = self.ann_index.as_ref().unwrap();
        let candidates = ann_index.get_candidates(query);
        let res = self.rank_candidates(query, &candidates, k, &SimilarityMetric::Cosine)?;
        let ms = start.elapsed().as_secs_f64()*1000.0;
        self.metrics.record_query(ms as u64, true, query.len()>=64);
        Ok((res, ms))
    }

    pub fn query_exact_with_timing(&self, query:&[f32], k:usize) -> Result<(Vec<(usize,f32)>, f64), String> {
        let start = Instant::now();
        let res = self.top_k_query_with_similarity(query, k, &SimilarityMetric::Cosine)?;
        let ms = start.elapsed().as_secs_f64()*1000.0;
        self.metrics.record_query(ms as u64, false, query.len()>=64);
        Ok((res, ms))
    }

    pub fn top_k_query_with_similarity(&self, query:&[f32], k:usize, sim:&SimilarityMetric)->Result<Vec<(usize,f32)>,String>{
        if query.len()>=64 { self.top_k_query_scalar_with_similarity(query,k,sim) } else { self.top_k_query_scalar_with_similarity(query,k,sim) }
    }

    pub fn top_k_query_scalar_with_similarity(&self, query:&[f32], k:usize, sim:&SimilarityMetric)->Result<Vec<(usize,f32)>,String>{
        let mut scores = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            if let Some(v) = self.get_vector(i) {
                let s = compute_similarity(query,v,sim);
                scores.push((i,s));
            }
        }
        scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    pub fn top_k_subset(&self, query:&[f32], subset:&[usize], k:usize) -> Result<Vec<(usize,f32)>,String>{
        if subset.is_empty() { return Ok(vec![]); }
        let mut scores = Vec::with_capacity(subset.len());
        let sim = SimilarityMetric::Cosine;
        for &idx in subset {
            if idx >= self.rows { continue; }
            if let Some(v)=self.get_vector(idx) {
                scores.push((idx, compute_similarity(query,v,&sim)));
            }
        }
        scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    pub fn health_check(&self)->Result<(),String>{
        if self.dims==0 || self.rows==0 { return Err("Engine has zero dimensions or rows".into()); }
        if self.vectors.len()!=self.dims*self.rows { return Err("Vector data size inconsistent".into()); }
        let sample = self.rows.min(10);
        for i in 0..sample {
            if let Some(v)=self.get_vector(i) {
                for &x in v { if !x.is_finite(){ return Err(format!("Invalid value at row {}", i)); } }
            }
        }
        if let Some(ann)=&self.ann_index { ann.health_check().map_err(|e| format!("ANN health: {}", e))?; }
        Ok(())
    }

    pub fn get_stats(&self)->(usize,f64,usize,usize,usize,usize,f64){
        let (q,avg,ann,exact,simd,scalar)=self.metrics.get_stats();
        let up = self.creation_time.elapsed().as_secs_f64();
        (q,avg,ann,exact,simd,scalar,up)
    }
    pub fn dims(&self)->usize{ self.dims }
    pub fn rows(&self)->usize{ self.rows }
    pub fn file_path(&self)->Option<&str>{ self.file_path.as_deref() }
    pub fn has_ann(&self)->bool{ self.ann_index.is_some() }
    pub fn memory_usage_bytes(&self)->usize{ self.vectors.len()*std::mem::size_of::<f32>() }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let (q,avg,_,_,_,_,uptime) = self.get_stats();
        info!("Engine dropped: {} vectors, {} queries, {:.2}ms avg, {:.1}s uptime", self.rows, q, avg, uptime);
    }
}

fn err(msg:&str)->std::io::Error { std::io::Error::new(std::io::ErrorKind::InvalidData, msg) }
