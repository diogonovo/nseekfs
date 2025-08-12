use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, BufReader, Write, Read, Seek, SeekFrom};
use std::path::Path;
use std::collections::{HashMap, HashSet};
use smallvec::SmallVec;
use dashmap::DashMap;
use log::{debug, info, warn};

// ========== COMPATIBILIDADE: Campos públicos originais ==========
#[derive(Clone)]
pub struct AnnIndex {
    pub dims: usize,
    pub bits: usize, // Bits da tabela 16-bit (compatibilidade)
    pub projections: Vec<Vec<f32>>, // Projeções da tabela 16-bit (compatibilidade)
    pub buckets: HashMap<u16, SmallVec<[usize; 16]>>, // Tabela 16-bit original

    // ========== NOVOS CAMPOS (Internos) ==========
    seed: u64,
    config: AnnConfig,
    
    // Tabela principal 64-bit (melhor resolução)
    main_projections: Vec<Vec<f32>>,
    main_bits: usize,
    main_buckets64: HashMap<u64, SmallVec<[usize; 32]>>,
    
    // Multi-tabelas 32-bit (diversidade)
    num_tables: usize,
    multi_projections: Vec<Vec<Vec<f32>>>,
    multi_bits32: usize,
    multi_tables32: Vec<HashMap<u32, SmallVec<[usize; 16]>>>,
}

#[derive(Clone, Debug)]
pub struct AnnConfig {
    pub target_candidates: usize,
    pub min_candidates: usize,
    pub max_hamming_64: usize,
    pub max_hamming_32: usize,
    pub max_hamming_16: usize,
    pub random_sample_size: usize,
}

impl AnnConfig {
    fn for_dataset_size(rows: usize) -> Self {
        match rows {
            0..=1_000 => AnnConfig {
                target_candidates: 50,
                min_candidates: 20,
                max_hamming_64: 12,
                max_hamming_32: 8,
                max_hamming_16: 3,
                random_sample_size: 100,
            },
            1_001..=10_000 => AnnConfig {
                target_candidates: 100,
                min_candidates: 40,
                max_hamming_64: 16,
                max_hamming_32: 10,
                max_hamming_16: 4,
                random_sample_size: 200,
            },
            10_001..=100_000 => AnnConfig {
                target_candidates: 200,
                min_candidates: 80,
                max_hamming_64: 20,
                max_hamming_32: 12,
                max_hamming_16: 5,
                random_sample_size: 500,
            },
            100_001..=1_000_000 => AnnConfig {
                target_candidates: 500,
                min_candidates: 200,
                max_hamming_64: 24,
                max_hamming_32: 15,
                max_hamming_16: 6,
                random_sample_size: 1000,
            },
            _ => AnnConfig {
                target_candidates: 1000,
                min_candidates: 400,
                max_hamming_64: 28,
                max_hamming_32: 18,
                max_hamming_16: 8,
                random_sample_size: 2000,
            },
        }
    }
}

impl AnnIndex {
    // ========== CONFIGURAÇÃO ADAPTATIVA ==========
    fn calculate_optimal_config(rows: usize, _dims: usize, requested_bits: usize) -> (usize, usize, usize, usize) {
        // Retorna: (bits16_compat, bits32_multi, bits64_main, num_tables)
        let bits16 = requested_bits.min(16);
        let (bits32, bits64, tables) = match rows {
            0..=1_000 => (12, 24, 2),
            1_001..=10_000 => (16, 32, 3),
            10_001..=100_000 => (20, 40, 4),
            100_001..=1_000_000 => (24, 48, 5),
            _ => (28, 56, 6),
        };
        (bits16, bits32, bits64, tables)
    }

    // ========== HASH FUNCTIONS ==========
    #[inline]
    fn hash_signs_u16(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u16 {
        let mut hash = 0u16;
        for (j, proj) in projections.iter().enumerate().take(bits.min(16)) {
            let dot = dot_product_optimized(vec, proj);
            if dot >= 0.0 { 
                hash |= 1 << j; 
            }
        }
        hash
    }

    #[inline]
    fn hash_signs_u32(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u32 {
        let mut hash = 0u32;
        for (j, proj) in projections.iter().enumerate().take(bits.min(32)) {
            let dot = dot_product_optimized(vec, proj);
            if dot >= 0.0 { 
                hash |= 1 << j; 
            }
        }
        hash
    }

    #[inline]
    fn hash_signs_u64(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u64 {
        let mut hash = 0u64;
        for (j, proj) in projections.iter().enumerate().take(bits.min(64)) {
            let dot = dot_product_optimized(vec, proj);
            if dot >= 0.0 { 
                hash |= 1u64 << j; 
            }
        }
        hash
    }

    // ========== BUILD ==========
    pub fn build(vectors: &[f32], dims: usize, rows: usize, bits: usize, seed: u64) -> Self {
        assert_eq!(vectors.len(), dims * rows, "Invalid vector length");
        
        let (bits16, bits32, bits64, num_tables) = Self::calculate_optimal_config(rows, dims, bits);
        let config = AnnConfig::for_dataset_size(rows);

        info!("Building ANN index: {} vectors, {} dims", rows, dims);
        info!("Config: 16-bit({}) + 32-bit({}) + 64-bit({}) + {} tables", 
              bits16, bits32, bits64, num_tables);

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // 1) COMPATIBILIDADE: Tabela 16-bit (campos públicos originais)
        let projections_16: Vec<Vec<f32>> = (0..bits16)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let buckets_u16 = DashMap::<u16, SmallVec<[usize; 16]>>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let vec_slice = &vectors[i * dims..(i + 1) * dims];
            let hash = Self::hash_signs_u16(vec_slice, &projections_16, bits16);
            buckets_u16.entry(hash).or_default().push(i);
        });
        let buckets_compat = buckets_u16.into_iter().collect::<HashMap<_, _>>();

        // 2) TABELA PRINCIPAL 64-bit (alta resolução)
        let main_projections: Vec<Vec<f32>> = (0..bits64)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let main_buckets = DashMap::<u64, SmallVec<[usize; 32]>>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let vec_slice = &vectors[i * dims..(i + 1) * dims];
            let hash = Self::hash_signs_u64(vec_slice, &main_projections, bits64);
            main_buckets.entry(hash).or_default().push(i);
        });
        let main_buckets64 = main_buckets.into_iter().collect::<HashMap<_, _>>();

        // 3) MULTI-TABELAS 32-bit (diversidade)
        let multi_data: Vec<(Vec<Vec<f32>>, HashMap<u32, SmallVec<[usize; 16]>>)> = 
            (0..num_tables).into_par_iter().map(|table_idx| {
                let mut table_rng = StdRng::seed_from_u64(seed + (table_idx as u64) * 10000);
                let table_projections: Vec<Vec<f32>> = (0..bits32)
                    .map(|_| (0..dims).map(|_| normal.sample(&mut table_rng) as f32).collect())
                    .collect();

                let table_buckets = DashMap::<u32, SmallVec<[usize; 16]>>::new();
                (0..rows).into_par_iter().for_each(|i| {
                    let vec_slice = &vectors[i * dims..(i + 1) * dims];
                    let hash = Self::hash_signs_u32(vec_slice, &table_projections, bits32);
                    table_buckets.entry(hash).or_default().push(i);
                });

                let table_map = table_buckets.into_iter().collect::<HashMap<_, _>>();
                (table_projections, table_map)
            }).collect();

        let (multi_projections, multi_tables32): (Vec<_>, Vec<_>) = multi_data.into_iter().unzip();

        info!("ANN index built successfully. Buckets: 16-bit({}), 64-bit({}), 32-bit({:?})",
              buckets_compat.len(), main_buckets64.len(), 
              multi_tables32.iter().map(|t| t.len()).collect::<Vec<_>>());

        Self {
            dims,
            bits: bits16,
            projections: projections_16,
            buckets: buckets_compat,
            seed,
            config,
            main_projections,
            main_bits: bits64,
            main_buckets64,
            num_tables,
            multi_projections,
            multi_bits32: bits32,
            multi_tables32,
        }
    }

    // ========== QUERY CANDIDATES ==========
    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        assert_eq!(query.len(), self.dims);

        let mut candidates_set: HashSet<usize> = HashSet::new();

        // 1) TABELA PRINCIPAL 64-bit + multi-probe
        if self.main_bits > 0 && !self.main_projections.is_empty() {
            let hash64 = Self::hash_signs_u64(query, &self.main_projections, self.main_bits);
            
            // Hash exato
            if let Some(cands) = self.main_buckets64.get(&hash64) {
                candidates_set.extend(cands.iter().cloned());
                debug!("64-bit exact: {} candidates", cands.len());
            }
            
            // Multi-probe Hamming distance = 1
            if candidates_set.len() < self.config.min_candidates {
                let max_flips = self.main_bits.min(self.config.max_hamming_64);
                for bit in 0..max_flips {
                    let flipped_hash = hash64 ^ (1u64 << bit);
                    if let Some(cands) = self.main_buckets64.get(&flipped_hash) {
                        candidates_set.extend(cands.iter().cloned());
                        if candidates_set.len() >= self.config.target_candidates {
                            break;
                        }
                    }
                }
                debug!("64-bit multi-probe: {} total candidates", candidates_set.len());
            }
        }

        // 2) MULTI-TABELAS 32-bit
        if self.multi_bits32 > 0 && !self.multi_projections.is_empty() {
            for (table_idx, table) in self.multi_tables32.iter().enumerate() {
                if candidates_set.len() >= self.config.target_candidates {
                    break;
                }

                let hash32 = Self::hash_signs_u32(query, &self.multi_projections[table_idx], self.multi_bits32);
                
                // Hash exato
                if let Some(cands) = table.get(&hash32) {
                    candidates_set.extend(cands.iter().cloned());
                }
                
                // Multi-probe se necessário
                if candidates_set.len() < self.config.min_candidates {
                    let max_flips = self.multi_bits32.min(self.config.max_hamming_32);
                    for bit in 0..max_flips {
                        let flipped_hash = hash32 ^ (1u32 << bit);
                        if let Some(cands) = table.get(&flipped_hash) {
                            candidates_set.extend(cands.iter().cloned());
                            if candidates_set.len() >= self.config.target_candidates {
                                break;
                            }
                        }
                    }
                }
            }
            debug!("32-bit tables: {} total candidates", candidates_set.len());
        }

        // 3) COMPATIBILIDADE: Tabela 16-bit original
        let hash16 = Self::hash_signs_u16(query, &self.projections, self.bits);
        if let Some(cands) = self.buckets.get(&hash16) {
            candidates_set.extend(cands.iter().cloned());
            debug!("16-bit compat: {} candidates", cands.len());
        }

        // Multi-probe na tabela 16-bit se ainda insuficiente
        if candidates_set.len() < self.config.min_candidates && self.bits > 0 {
            let max_radius = self.config.max_hamming_16.min(3); // Limite para evitar explosão
            
            // Hamming distance = 1
            for i in 0..self.bits.min(16) {
                let hash = hash16 ^ (1u16 << i);
                if let Some(cands) = self.buckets.get(&hash) {
                    candidates_set.extend(cands.iter().cloned());
                    if candidates_set.len() >= self.config.target_candidates {
                        break;
                    }
                }
            }
            
            // Hamming distance = 2 (limitado)
            if candidates_set.len() < self.config.min_candidates && max_radius >= 2 {
                for i in 0..self.bits.min(16) {
                    for j in (i + 1)..self.bits.min(16) {
                        let hash = hash16 ^ (1u16 << i) ^ (1u16 << j);
                        if let Some(cands) = self.buckets.get(&hash) {
                            candidates_set.extend(cands.iter().cloned());
                            if candidates_set.len() >= self.config.target_candidates {
                                break;
                            }
                        }
                    }
                    if candidates_set.len() >= self.config.target_candidates {
                        break;
                    }
                }
            }
        }

        // 4) FALLBACK: Amostra aleatória se ainda insuficiente
        if candidates_set.len() < self.config.min_candidates {
            self.add_random_fallback(&mut candidates_set);
        }

        let mut result: Vec<usize> = candidates_set.into_iter().collect();
        result.sort_unstable();
        result.dedup();
        
        debug!("Total candidates returned: {}", result.len());
        result
    }

    fn add_random_fallback(&self, candidates: &mut HashSet<usize>) {
        // Estimar número total de vetores baseado nos buckets
        let total_vectors = self.main_buckets64.values()
            .map(|bucket| bucket.len())
            .sum::<usize>()
            .max(self.buckets.values().map(|b| b.len()).sum::<usize>())
            .max(1);

        use rand::seq::SliceRandom;
        let mut rng = StdRng::seed_from_u64(self.seed + 99999);
        let mut indices: Vec<usize> = (0..total_vectors).collect();
        indices.shuffle(&mut rng);

        let needed = self.config.min_candidates.saturating_sub(candidates.len());
        let sample_size = needed.min(self.config.random_sample_size);
        
        candidates.extend(indices.into_iter().take(sample_size));
        debug!("Added {} random fallback candidates", sample_size);
    }

    pub fn query(&self, query: &[f32], top_k: usize, vectors: &[f32]) -> Vec<(usize, f32)> {
        let candidates = self.query_candidates(query);
        
        if candidates.is_empty() {
            warn!("No candidates found for query");
            return Vec::new();
        }

        debug!("Processing {} candidates for top-{}", candidates.len(), top_k);

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .filter_map(|i| {
                let offset = i * self.dims;
                if offset + self.dims <= vectors.len() {
                    let vec_slice = &vectors[offset..offset + self.dims];
                    let score = dot_product_optimized(query, vec_slice);
                    Some((i, score))
                } else {
                    warn!("Index {} out of bounds, skipping", i);
                    None
                }
            })
            .collect();

        // Ordenação otimizada
        if results.len() > top_k {
            results.select_nth_unstable_by(top_k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(top_k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        debug!("Returning {} results", results.len());
        results
    }

    // ========== PERSISTÊNCIA COM VERSIONAMENTO ==========
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);

        // Header com magic number e versão
        file.write_all(b"NSEEKANN")?;
        file.write_all(&3u32.to_le_bytes())?; // Version 3

        // Metadados básicos
        file.write_all(&(self.dims as u32).to_le_bytes())?;
        file.write_all(&(self.bits as u32).to_le_bytes())?;
        file.write_all(&(self.main_bits as u32).to_le_bytes())?;
        file.write_all(&(self.multi_bits32 as u32).to_le_bytes())?;
        file.write_all(&(self.num_tables as u32).to_le_bytes())?;
        file.write_all(&self.seed.to_le_bytes())?;

        // Config
        file.write_all(&(self.config.target_candidates as u32).to_le_bytes())?;
        file.write_all(&(self.config.min_candidates as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_hamming_64 as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_hamming_32 as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_hamming_16 as u32).to_le_bytes())?;
        file.write_all(&(self.config.random_sample_size as u32).to_le_bytes())?;

        // 1. Projeções 16-bit (compatibilidade)
        file.write_all(&(self.projections.len() as u32).to_le_bytes())?;
        for proj in &self.projections {
            for &val in proj {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // 2. Buckets 16-bit (compatibilidade)
        file.write_all(&(self.buckets.len() as u32).to_le_bytes())?;
        for (&hash, ids) in &self.buckets {
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids {
                file.write_all(&(id as u32).to_le_bytes())?;
            }
        }

        // 3. Projeções 64-bit (tabela principal)
        file.write_all(&(self.main_projections.len() as u32).to_le_bytes())?;
        for proj in &self.main_projections {
            for &val in proj {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // 4. Buckets 64-bit (tabela principal)
        file.write_all(&(self.main_buckets64.len() as u32).to_le_bytes())?;
        for (&hash, ids) in &self.main_buckets64 {
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids {
                file.write_all(&(id as u32).to_le_bytes())?;
            }
        }

        // 5. Multi-projeções 32-bit
        file.write_all(&(self.multi_projections.len() as u32).to_le_bytes())?;
        for table_projections in &self.multi_projections {
            file.write_all(&(table_projections.len() as u32).to_le_bytes())?;
            for proj in table_projections {
                for &val in proj {
                    file.write_all(&val.to_le_bytes())?;
                }
            }
        }

        // 6. Multi-tabelas 32-bit
        file.write_all(&(self.multi_tables32.len() as u32).to_le_bytes())?;
        for table in &self.multi_tables32 {
            file.write_all(&(table.len() as u32).to_le_bytes())?;
            for (&hash, ids) in table {
                file.write_all(&hash.to_le_bytes())?;
                file.write_all(&(ids.len() as u32).to_le_bytes())?;
                for &id in ids {
                    file.write_all(&(id as u32).to_le_bytes())?;
                }
            }
        }
        
        info!("ANN index saved successfully");
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, _vectors: &[f32]) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);

        // Tentar ler magic number
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;

        if &magic == b"NSEEKANN" {
            // Formato novo com versionamento
            let mut u32_buf = [0u8; 4];
            file.read_exact(&mut u32_buf)?;
            let version = u32::from_le_bytes(u32_buf);

            match version {
                3 => Self::load_version_3(file),
                2 => Self::load_version_2(file),
                _ => {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported ANN version: {}", version)
                    ))
                }
            }
        } else {
            // Formato legado - fazer seek back e carregar como v1
            file.seek(SeekFrom::Start(0))?;
            Self::load_legacy_format(file)
        }
    }

    fn load_version_3<R: Read>(mut file: R) -> std::io::Result<Self> {
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];

        // Metadados básicos
        file.read_exact(&mut u32_buf)?; let dims = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let bits = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let main_bits = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let multi_bits32 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let num_tables = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u64_buf)?; let seed = u64::from_le_bytes(u64_buf);

        // Config
        file.read_exact(&mut u32_buf)?; let target_candidates = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let min_candidates = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_hamming_64 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_hamming_32 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_hamming_16 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let random_sample_size = u32::from_le_bytes(u32_buf) as usize;

        let config = AnnConfig {
            target_candidates,
            min_candidates,
            max_hamming_64,
            max_hamming_32,
            max_hamming_16,
            random_sample_size,
        };

        // 1. Projeções 16-bit
        file.read_exact(&mut u32_buf)?; let num_proj16 = u32::from_le_bytes(u32_buf) as usize;
        let mut projections = vec![vec![0f32; dims]; num_proj16];
        for proj in &mut projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // 2. Buckets 16-bit
        file.read_exact(&mut u32_buf)?; let num_buckets16 = u32::from_le_bytes(u32_buf) as usize;
        let mut buckets = HashMap::with_capacity(num_buckets16);
        for _ in 0..num_buckets16 {
            let hash = read_u16(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
            let mut ids = SmallVec::with_capacity(len);
            for _ in 0..len {
                ids.push(read_u32(&mut file)? as usize);
            }
            buckets.insert(hash, ids);
        }

        // 3. Projeções 64-bit
        file.read_exact(&mut u32_buf)?; let num_proj64 = u32::from_le_bytes(u32_buf) as usize;
        let mut main_projections = vec![vec![0f32; dims]; num_proj64];
        for proj in &mut main_projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // 4. Buckets 64-bit
        file.read_exact(&mut u32_buf)?; let num_buckets64 = u32::from_le_bytes(u32_buf) as usize;
        let mut main_buckets64 = HashMap::with_capacity(num_buckets64);
        for _ in 0..num_buckets64 {
            let hash = read_u64(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
            let mut ids = SmallVec::with_capacity(len);
            for _ in 0..len {
                ids.push(read_u32(&mut file)? as usize);
            }
            main_buckets64.insert(hash, ids);
        }

        // 5. Multi-projeções 32-bit
        file.read_exact(&mut u32_buf)?; let num_multi_proj = u32::from_le_bytes(u32_buf) as usize;
        let mut multi_projections = Vec::with_capacity(num_multi_proj);
        for _ in 0..num_multi_proj {
            file.read_exact(&mut u32_buf)?; let table_proj_count = u32::from_le_bytes(u32_buf) as usize;
            let mut table_projections = vec![vec![0f32; dims]; table_proj_count];
            for proj in &mut table_projections {
                for val in proj.iter_mut() {
                    *val = read_f32(&mut file)?;
                }
            }
            multi_projections.push(table_projections);
        }

        // 6. Multi-tabelas 32-bit
        file.read_exact(&mut u32_buf)?; let num_multi_tables = u32::from_le_bytes(u32_buf) as usize;
        let mut multi_tables32 = Vec::with_capacity(num_multi_tables);
        for _ in 0..num_multi_tables {
            file.read_exact(&mut u32_buf)?; let table_size = u32::from_le_bytes(u32_buf) as usize;
            let mut table = HashMap::with_capacity(table_size);
            for _ in 0..table_size {
                let hash = read_u32(&mut file)?;
                file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
                let mut ids = SmallVec::with_capacity(len);
                for _ in 0..len {
                    ids.push(read_u32(&mut file)? as usize);
                }
                table.insert(hash, ids);
            }
            multi_tables32.push(table);
        }

        info!("ANN index v3 loaded successfully: {} dims, {} tables", dims, num_tables);

        Ok(Self {
            dims,
            bits,
            projections,
            buckets,
            seed,
            config,
            main_projections,
            main_bits,
            main_buckets64,
            num_tables,
            multi_projections,
            multi_bits32,
            multi_tables32,
        })
    }

    fn load_version_2<R: Read>(mut file: R) -> std::io::Result<Self> {
        // Implementação simplificada da versão 2 - compatibilidade básica
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];

        file.read_exact(&mut u32_buf)?; let dims = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let bits = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u64_buf)?; let seed = u64::from_le_bytes(u64_buf);

        // Para v2, carregar apenas dados básicos e criar config padrão
        let config = AnnConfig::for_dataset_size(1000); // Config padrão

        // Ler projeções básicas
        file.read_exact(&mut u32_buf)?; let num_proj = u32::from_le_bytes(u32_buf) as usize;
        let mut projections = vec![vec![0f32; dims]; num_proj];
        for proj in &mut projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // Ler buckets básicos
        file.read_exact(&mut u32_buf)?; let num_buckets = u32::from_le_bytes(u32_buf) as usize;
        let mut buckets = HashMap::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            let hash = read_u16(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
            let mut ids = SmallVec::with_capacity(len);
            for _ in 0..len {
                ids.push(read_u32(&mut file)? as usize);
            }
            buckets.insert(hash, ids);
        }

        warn!("Loading ANN v2 format with minimal features");

        Ok(Self {
            dims,
            bits,
            projections,
            buckets,
            seed,
            config,
            main_projections: Vec::new(),
            main_bits: 0,
            main_buckets64: HashMap::new(),
            num_tables: 0,
            multi_projections: Vec::new(),
            multi_bits32: 0,
            multi_tables32: Vec::new(),
        })
    }

    fn load_legacy_format<R: Read>(mut file: R) -> std::io::Result<Self> {
        let mut u32_buf = [0u8; 4];

        // Formato legado original
        file.read_exact(&mut u32_buf)?; let dims = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let bits = u32::from_le_bytes(u32_buf) as usize;

        // Ler projeções
        let mut projections = vec![vec![0f32; dims]; bits];
        for proj in &mut projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // Ler buckets
        file.read_exact(&mut u32_buf)?; let num_buckets = u32::from_le_bytes(u32_buf);
        let mut buckets = HashMap::new();
        for _ in 0..num_buckets {
            let hash = read_u16(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf);
            let mut ids = SmallVec::with_capacity(len as usize);
            for _ in 0..len {
                ids.push(read_u32(&mut file)? as usize);
            }
            buckets.insert(hash, ids);
        }

        warn!("Loading legacy ANN format with basic features only");

        Ok(Self {
            dims,
            bits,
            projections,
            buckets,
            seed: 42, // Default seed
            config: AnnConfig::for_dataset_size(1000), // Default config
            main_projections: Vec::new(),
            main_bits: 0,
            main_buckets64: HashMap::new(),
            num_tables: 0,
            multi_projections: Vec::new(),
            multi_bits32: 0,
            multi_tables32: Vec::new(),
        })
    }
}

// ========== HELPERS OTIMIZADOS ==========
#[inline]
fn dot_product_optimized(a: &[f32], b: &[f32]) -> f32 {
    // Versão otimizada com loop unrolling
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    
    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base] 
             + a[base + 1] * b[base + 1]
             + a[base + 2] * b[base + 2] 
             + a[base + 3] * b[base + 3];
    }
    
    // Remainder
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }
    
    sum
}

// Helper functions para leitura de dados
#[inline] fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(f32::from_le_bytes(buf))
}
#[inline] fn read_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; Ok(u16::from_le_bytes(buf))
}
#[inline] fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(u32::from_le_bytes(buf))
}
#[inline] fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(u64::from_le_bytes(buf))
}