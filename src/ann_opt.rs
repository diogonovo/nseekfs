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
use log::{debug, info, warn, error};
use std::sync::atomic::{AtomicUsize, Ordering};

// ========== CONSTANTES DE SEGURANÇA ==========
const MAX_BUCKET_SIZE: usize = 2000; // Limite para prevenir memory explosion
const MAX_TOTAL_CANDIDATES: usize = 50000; // Limite global de candidatos
const MAX_HAMMING_RADIUS: usize = 8; // Limite de multi-probe para evitar explosão combinatorial
const MIN_DATASET_SIZE: usize = 10; // Dataset mínimo para ANN fazer sentido

// ========== MÉTRICAS INTERNAS ==========
#[derive(Debug, Default)]
pub struct AnnMetrics {
    pub total_queries: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub fallback_activations: AtomicUsize,
    pub avg_candidates_generated: AtomicUsize,
}

// Implementar Clone manualmente
impl Clone for AnnMetrics {
    fn clone(&self) -> Self {
        AnnMetrics {
            total_queries: AtomicUsize::new(self.total_queries.load(Ordering::Relaxed)),
            cache_hits: AtomicUsize::new(self.cache_hits.load(Ordering::Relaxed)),
            fallback_activations: AtomicUsize::new(self.fallback_activations.load(Ordering::Relaxed)),
            avg_candidates_generated: AtomicUsize::new(self.avg_candidates_generated.load(Ordering::Relaxed)),
        }
    }
}

impl AnnMetrics {
    pub fn record_query(&self, candidates: usize, used_fallback: bool) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.avg_candidates_generated.fetch_add(candidates, Ordering::Relaxed);
        if used_fallback {
            self.fallback_activations.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_stats(&self) -> (usize, usize, usize, f64) {
        let total = self.total_queries.load(Ordering::Relaxed);
        let cache = self.cache_hits.load(Ordering::Relaxed);
        let fallback = self.fallback_activations.load(Ordering::Relaxed);
        let avg_candidates = if total > 0 {
            self.avg_candidates_generated.load(Ordering::Relaxed) as f64 / total as f64
        } else { 0.0 };
        (total, cache, fallback, avg_candidates)
    }
}

// ========== CONFIGURAÇÃO MELHORADA ==========
#[derive(Clone, Debug)]
pub struct AnnConfig {
    pub target_candidates: usize,
    pub min_candidates: usize,
    pub max_candidates: usize, // NOVO: Limite superior
    pub max_hamming_64: usize,
    pub max_hamming_32: usize,
    pub max_hamming_16: usize,
    pub random_sample_size: usize,
    pub enable_metrics: bool, // NOVO: Toggle para métricas
}

impl AnnConfig {
    fn for_dataset_size(rows: usize) -> Self {
        // Validação de tamanho mínimo
        if rows < MIN_DATASET_SIZE {
            warn!("Dataset muito pequeno para ANN ({}), usando configuração mínima", rows);
        }

        let config = match rows {
            0..=1_000 => AnnConfig {
                target_candidates: 50,
                min_candidates: 10,
                max_candidates: 200,
                max_hamming_64: 8,
                max_hamming_32: 6,
                max_hamming_16: 3,
                random_sample_size: 50,
                enable_metrics: true,
            },
            1_001..=10_000 => AnnConfig {
                target_candidates: 100,
                min_candidates: 25,
                max_candidates: 500,
                max_hamming_64: 12,
                max_hamming_32: 8,
                max_hamming_16: 4,
                random_sample_size: 100,
                enable_metrics: true,
            },
            10_001..=100_000 => AnnConfig {
                target_candidates: 200,
                min_candidates: 50,
                max_candidates: 1000,
                max_hamming_64: 16,
                max_hamming_32: 10,
                max_hamming_16: 5,
                random_sample_size: 200,
                enable_metrics: true,
            },
            100_001..=1_000_000 => AnnConfig {
                target_candidates: 500,
                min_candidates: 100,
                max_candidates: 2000,
                max_hamming_64: 20,
                max_hamming_32: 12,
                max_hamming_16: 6,
                random_sample_size: 500,
                enable_metrics: true,
            },
            _ => AnnConfig {
                target_candidates: 1000,
                min_candidates: 200,
                max_candidates: 5000,
                max_hamming_64: 24,
                max_hamming_32: 15,
                max_hamming_16: 8,
                random_sample_size: 1000,
                enable_metrics: true,
            },
        };

        // Validação dos limites
        assert!(config.min_candidates <= config.target_candidates);
        assert!(config.target_candidates <= config.max_candidates);
        assert!(config.max_candidates <= MAX_TOTAL_CANDIDATES);

        config
    }

    // NOVO: Configuração conservadora para datasets pequenos
    pub fn conservative() -> Self {
        AnnConfig {
            target_candidates: 25,
            min_candidates: 5,
            max_candidates: 100,
            max_hamming_64: 4,
            max_hamming_32: 3,
            max_hamming_16: 2,
            random_sample_size: 25,
            enable_metrics: false,
        }
    }
}

// ========== BUCKET SEGURO ==========
#[derive(Clone, Debug)]
struct SafeBucket {
    items: SmallVec<[usize; 16]>,
    is_full: bool,
}

impl SafeBucket {
    fn new() -> Self {
        Self {
            items: SmallVec::new(),
            is_full: false,
        }
    }

    fn push(&mut self, item: usize) -> bool {
        if self.items.len() >= MAX_BUCKET_SIZE {
            if !self.is_full {
                warn!("Bucket atingiu tamanho máximo ({}), ignorando novos itens", MAX_BUCKET_SIZE);
                self.is_full = true;
            }
            return false;
        }
        self.items.push(item);
        true
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> impl Iterator<Item = &usize> {
        self.items.iter()
    }

    fn extend_from_slice(&mut self, other: &[usize]) {
        for &item in other {
            if !self.push(item) {
                break;
            }
        }
    }
}

// ========== ÍNDICE ANN MELHORADO ==========
#[derive(Clone)]
pub struct AnnIndex {
    // Campos públicos para compatibilidade
    pub dims: usize,
    pub bits: usize,
    pub projections: Vec<Vec<f32>>,
    pub buckets: HashMap<u16, SmallVec<[usize; 16]>>, // Mantido para compatibilidade

    // Campos internos melhorados
    seed: u64,
    config: AnnConfig,
    total_vectors: usize, // NOVO: Track do tamanho real do dataset
    metrics: AnnMetrics, // NOVO: Métricas internas
    
    // Tabela principal 64-bit (melhor resolução)
    main_projections: Vec<Vec<f32>>,
    main_bits: usize,
    main_buckets64: HashMap<u64, SafeBucket>, // NOVO: Buckets seguros
    
    // Multi-tabelas 32-bit (diversidade)
    num_tables: usize,
    multi_projections: Vec<Vec<Vec<f32>>>,
    multi_bits32: usize,
    multi_tables32: Vec<HashMap<u32, SafeBucket>>, // NOVO: Buckets seguros
}

// IMPORTANTE: Implementar Send + Sync para thread safety
unsafe impl Send for AnnIndex {}
unsafe impl Sync for AnnIndex {}

impl AnnIndex {
    // ========== CONFIGURAÇÃO ADAPTATIVA MELHORADA ==========
    fn calculate_optimal_config(rows: usize, dims: usize, requested_bits: usize) -> (usize, usize, usize, usize) {
        // Validações de entrada
        if rows < MIN_DATASET_SIZE {
            warn!("Dataset muito pequeno ({}), ANN pode não ser eficaz", rows);
        }

        if dims < 8 {
            error!("Dimensões muito baixas ({}), ANN não funcionará corretamente", dims);
            return (8, 8, 16, 1); // Configuração mínima válida
        }

        let bits16 = requested_bits.min(16).max(8); // Mínimo 8 bits
        let (bits32, bits64, tables) = match rows {
            0..=1_000 => (12, 20, 2),
            1_001..=10_000 => (16, 28, 3),
            10_001..=100_000 => (20, 36, 4),
            100_001..=1_000_000 => (24, 44, 5),
            _ => (28, 52, 6), // Limitado para evitar overflow
        };

        // Validações finais
        assert!(bits16 <= 16);
        assert!(bits32 <= 32);
        assert!(bits64 <= 64);
        assert!(tables >= 1 && tables <= 10);

        (bits16, bits32, bits64, tables)
    }

    // ========== HASH FUNCTIONS COM VALIDAÇÃO ==========
    #[inline]
    fn hash_signs_u16(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u16 {
        if vec.is_empty() || projections.is_empty() {
            return 0;
        }

        let mut hash = 0u16;
        let safe_bits = bits.min(16).min(projections.len());
        
        for (j, proj) in projections.iter().enumerate().take(safe_bits) {
            if proj.len() != vec.len() {
                error!("Dimension mismatch in hash_signs_u16: {} vs {}", proj.len(), vec.len());
                continue;
            }
            
            let dot = dot_product_safe(vec, proj);
            if dot >= 0.0 { 
                hash |= 1 << j; 
            }
        }
        hash
    }

    #[inline]
    fn hash_signs_u32(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u32 {
        if vec.is_empty() || projections.is_empty() {
            return 0;
        }

        let mut hash = 0u32;
        let safe_bits = bits.min(32).min(projections.len());
        
        for (j, proj) in projections.iter().enumerate().take(safe_bits) {
            if proj.len() != vec.len() {
                continue;
            }
            
            let dot = dot_product_safe(vec, proj);
            if dot >= 0.0 { 
                hash |= 1 << j; 
            }
        }
        hash
    }

    #[inline]
    fn hash_signs_u64(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u64 {
        if vec.is_empty() || projections.is_empty() {
            return 0;
        }

        let mut hash = 0u64;
        let safe_bits = bits.min(64).min(projections.len());
        
        for (j, proj) in projections.iter().enumerate().take(safe_bits) {
            if proj.len() != vec.len() {
                continue;
            }
            
            let dot = dot_product_safe(vec, proj);
            if dot >= 0.0 { 
                hash |= 1u64 << j; 
            }
        }
        hash
    }

    // ========== BUILD COM VALIDAÇÕES EXTENSIVAS ==========
    pub fn build(vectors: &[f32], dims: usize, rows: usize, bits: usize, seed: u64) -> Self {
        // Validações críticas
        if vectors.len() != dims * rows {
            panic!("Invalid vector data: expected {} elements, got {}", dims * rows, vectors.len());
        }

        if dims < 8 {
            panic!("Minimum 8 dimensions required for ANN, got {}", dims);
        }

        if rows < MIN_DATASET_SIZE {
            warn!("Dataset muito pequeno ({}) para ANN, consider exact search", rows);
        }

        let (bits16, bits32, bits64, num_tables) = Self::calculate_optimal_config(rows, dims, bits);
        let config = AnnConfig::for_dataset_size(rows);

        info!("Building ANN index: {} vectors, {} dims", rows, dims);
        info!("Config: 16-bit({}) + 32-bit({}) + 64-bit({}) + {} tables", 
              bits16, bits32, bits64, num_tables);
        info!("Safety limits: max_bucket={}, max_candidates={}", 
              MAX_BUCKET_SIZE, config.max_candidates);

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // 1) COMPATIBILIDADE: Tabela 16-bit (campos públicos originais)
        let projections_16: Vec<Vec<f32>> = (0..bits16)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let buckets_u16 = DashMap::<u16, SmallVec<[usize; 16]>>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let vec_slice = &vectors[i * dims..(i + 1) * dims];
            
            // Validação da slice
            if vec_slice.len() != dims {
                error!("Invalid vector slice at index {}: expected {} dims, got {}", i, dims, vec_slice.len());
                return;
            }
            
            let hash = Self::hash_signs_u16(vec_slice, &projections_16, bits16);
            let mut bucket = buckets_u16.entry(hash).or_default();
            
            // Aplicar limite de bucket
            if bucket.len() < MAX_BUCKET_SIZE {
                bucket.push(i);
            } else if bucket.len() == MAX_BUCKET_SIZE {
                warn!("Bucket 16-bit #{} atingiu limite máximo", hash);
            }
        });
        let buckets_compat = buckets_u16.into_iter().collect::<HashMap<_, _>>();

        // 2) TABELA PRINCIPAL 64-bit (alta resolução) com buckets seguros
        let main_projections: Vec<Vec<f32>> = (0..bits64)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let main_buckets = DashMap::<u64, SafeBucket>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let vec_slice = &vectors[i * dims..(i + 1) * dims];
            if vec_slice.len() != dims {
                return;
            }
            
            let hash = Self::hash_signs_u64(vec_slice, &main_projections, bits64);
            main_buckets.entry(hash).or_insert_with(SafeBucket::new).push(i);
        });
        let main_buckets64 = main_buckets.into_iter().collect::<HashMap<_, _>>();

        // 3) MULTI-TABELAS 32-bit (diversidade) com buckets seguros
        let multi_data: Vec<(Vec<Vec<f32>>, HashMap<u32, SafeBucket>)> = 
            (0..num_tables).into_par_iter().map(|table_idx| {
                let mut table_rng = StdRng::seed_from_u64(seed + (table_idx as u64) * 10000);
                let table_projections: Vec<Vec<f32>> = (0..bits32)
                    .map(|_| (0..dims).map(|_| normal.sample(&mut table_rng) as f32).collect())
                    .collect();

                let table_buckets = DashMap::<u32, SafeBucket>::new();
                (0..rows).into_par_iter().for_each(|i| {
                    let vec_slice = &vectors[i * dims..(i + 1) * dims];
                    if vec_slice.len() != dims {
                        return;
                    }
                    
                    let hash = Self::hash_signs_u32(vec_slice, &table_projections, bits32);
                    table_buckets.entry(hash).or_insert_with(SafeBucket::new).push(i);
                });

                let table_map = table_buckets.into_iter().collect::<HashMap<_, _>>();
                (table_projections, table_map)
            }).collect();

        let (multi_projections, multi_tables32): (Vec<_>, Vec<_>) = multi_data.into_iter().unzip();

        // Estatísticas de construção
        let total_buckets_16 = buckets_compat.len();
        let total_buckets_64 = main_buckets64.len();
        let total_buckets_32: usize = multi_tables32.iter().map(|t| t.len()).sum();
        
        info!("ANN index built successfully:");
        info!("  Buckets: 16-bit({}), 64-bit({}), 32-bit({})", 
              total_buckets_16, total_buckets_64, total_buckets_32);
        info!("  Average bucket sizes: 16-bit({:.1}), 64-bit({:.1})", 
              if total_buckets_16 > 0 { rows as f64 / total_buckets_16 as f64 } else { 0.0 },
              if total_buckets_64 > 0 { rows as f64 / total_buckets_64 as f64 } else { 0.0 });

        Self {
            dims,
            bits: bits16,
            projections: projections_16,
            buckets: buckets_compat,
            seed,
            config,
            total_vectors: rows,
            metrics: AnnMetrics::default(),
            main_projections,
            main_bits: bits64,
            main_buckets64,
            num_tables,
            multi_projections,
            // CORRIGIDO: em build usamos diretamente bits32
            multi_bits32: bits32,
            multi_tables32,
        }
    }

    // ========== QUERY CANDIDATES COM LIMITES DE SEGURANÇA ==========
    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        if query.len() != self.dims {
            error!("Query dimension mismatch: expected {}, got {}", self.dims, query.len());
            return Vec::new();
        }

        // Validação de entrada
        if query.iter().any(|&x| !x.is_finite()) {
            error!("Query vector contains invalid values (NaN/Inf)");
            return Vec::new();
        }

        let mut candidates_set: HashSet<usize> = HashSet::new();
        let mut used_fallback = false;

        // 1) TABELA PRINCIPAL 64-bit + multi-probe LIMITADO
        if self.main_bits > 0 && !self.main_projections.is_empty() {
            let hash64 = Self::hash_signs_u64(query, &self.main_projections, self.main_bits);
            
            // Hash exato
            if let Some(bucket) = self.main_buckets64.get(&hash64) {
                for &candidate in bucket.iter() {
                    if candidate < self.total_vectors {
                        candidates_set.insert(candidate);
                    }
                }
                debug!("64-bit exact: {} candidates", bucket.len());
            }
            
            // Multi-probe LIMITADO para evitar explosão
            if candidates_set.len() < self.config.min_candidates {
                let max_flips = self.main_bits.min(self.config.max_hamming_64).min(MAX_HAMMING_RADIUS);
                
                for bit in 0..max_flips {
                    if candidates_set.len() >= self.config.max_candidates {
                        break;
                    }
                    
                    let flipped_hash = hash64 ^ (1u64 << bit);
                    if let Some(bucket) = self.main_buckets64.get(&flipped_hash) {
                        for &candidate in bucket.iter() {
                            if candidate < self.total_vectors {
                                candidates_set.insert(candidate);
                                if candidates_set.len() >= self.config.max_candidates {
                                    break;
                                }
                            }
                        }
                    }
                }
                debug!("64-bit multi-probe: {} total candidates", candidates_set.len());
            }
        }

        // 2) MULTI-TABELAS 32-bit COM LIMITES
        if self.multi_bits32 > 0 && !self.multi_projections.is_empty() {
            for (table_idx, table) in self.multi_tables32.iter().enumerate() {
                if candidates_set.len() >= self.config.max_candidates {
                    break;
                }

                if table_idx >= self.multi_projections.len() {
                    continue;
                }

                let hash32 = Self::hash_signs_u32(query, &self.multi_projections[table_idx], self.multi_bits32);
                
                // Hash exato
                if let Some(bucket) = table.get(&hash32) {
                    for &candidate in bucket.iter() {
                        if candidate < self.total_vectors {
                            candidates_set.insert(candidate);
                            if candidates_set.len() >= self.config.max_candidates {
                                break;
                            }
                        }
                    }
                }
                
                // Multi-probe limitado
                if candidates_set.len() < self.config.min_candidates {
                    let max_flips = self.multi_bits32.min(self.config.max_hamming_32).min(MAX_HAMMING_RADIUS);
                    
                    for bit in 0..max_flips {
                        if candidates_set.len() >= self.config.max_candidates {
                            break;
                        }
                        
                        let flipped_hash = hash32 ^ (1u32 << bit);
                        if let Some(bucket) = table.get(&flipped_hash) {
                            for &candidate in bucket.iter() {
                                if candidate < self.total_vectors {
                                    candidates_set.insert(candidate);
                                    if candidates_set.len() >= self.config.max_candidates {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3) COMPATIBILIDADE: Tabela 16-bit original COM LIMITES
        if candidates_set.len() < self.config.min_candidates && !self.projections.is_empty() {
            let hash16 = Self::hash_signs_u16(query, &self.projections, self.bits);
            
            if let Some(bucket) = self.buckets.get(&hash16) {
                for &candidate in bucket.iter() {
                    if candidate < self.total_vectors {
                        candidates_set.insert(candidate);
                        if candidates_set.len() >= self.config.max_candidates {
                            break;
                        }
                    }
                }
            }

            // Multi-probe 16-bit MUITO LIMITADO
            if candidates_set.len() < self.config.min_candidates {
                let max_radius = self.config.max_hamming_16.min(3); // Máximo 3 para evitar explosão
                
                for i in 0..self.bits.min(16).min(max_radius) {
                    if candidates_set.len() >= self.config.max_candidates {
                        break;
                    }
                    
                    let hash = hash16 ^ (1u16 << i);
                    if let Some(bucket) = self.buckets.get(&hash) {
                        for &candidate in bucket.iter() {
                            if candidate < self.total_vectors {
                                candidates_set.insert(candidate);
                                if candidates_set.len() >= self.config.max_candidates {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4) FALLBACK LIMITADO se ainda insuficiente
        if candidates_set.len() < self.config.min_candidates {
            self.add_safe_random_fallback(&mut candidates_set);
            used_fallback = true;
        }

        let mut result: Vec<usize> = candidates_set.into_iter().collect();
        
        // Validação final e limpeza
        result.retain(|&idx| idx < self.total_vectors);
        result.sort_unstable();
        result.dedup();
        
        // Aplicar limite final absoluto
        if result.len() > self.config.max_candidates {
            result.truncate(self.config.max_candidates);
            warn!("Truncated candidates to max limit: {}", self.config.max_candidates);
        }

        // Registrar métricas
        if self.config.enable_metrics {
            self.metrics.record_query(result.len(), used_fallback);
        }
        
        debug!("Total candidates returned: {} (fallback: {})", result.len(), used_fallback);
        result
    }

    fn add_safe_random_fallback(&self, candidates: &mut HashSet<usize>) {
        use rand::seq::SliceRandom;
        let mut rng = StdRng::seed_from_u64(self.seed + 99999);
        
        // Amostra limitada e segura
        let sample_size = self.config.random_sample_size.min(self.total_vectors).min(1000);
        let needed = self.config.min_candidates.saturating_sub(candidates.len()).min(sample_size);
        
        if needed == 0 {
            return;
        }

        let mut indices: Vec<usize> = (0..self.total_vectors).collect();
        indices.shuffle(&mut rng);
        
        let mut added = 0;
        for idx in indices.into_iter().take(sample_size) {
            if added >= needed {
                break;
            }
            if candidates.insert(idx) {
                added += 1;
            }
        }
        
        debug!("Added {} random fallback candidates (needed: {})", added, needed);
    }

    // ========== QUERY COM VALIDAÇÕES ==========
    pub fn query(&self, query: &[f32], top_k: usize, vectors: &[f32]) -> Vec<(usize, f32)> {
        if query.len() != self.dims {
            error!("Query dimension mismatch: expected {}, got {}", self.dims, query.len());
            return Vec::new();
        }

        if vectors.len() != self.dims * self.total_vectors {
            error!("Vector data size mismatch: expected {}, got {}", 
                   self.dims * self.total_vectors, vectors.len());
            return Vec::new();
        }

        let candidates = self.query_candidates(query);
        
        if candidates.is_empty() {
            warn!("No candidates found for query");
            return Vec::new();
        }

        debug!("Processing {} candidates for top-{}", candidates.len(), top_k);

        let mut results: Vec<(usize, f32)> = candidates
            .into_par_iter()
            .filter_map(|i| {
                if i >= self.total_vectors {
                    warn!("Invalid candidate index: {} >= {}", i, self.total_vectors);
                    return None;
                }
                
                let offset = i * self.dims;
                if offset + self.dims > vectors.len() {
                    warn!("Vector offset out of bounds: {} + {} > {}", offset, self.dims, vectors.len());
                    return None;
                }
                
                let vec_slice = &vectors[offset..offset + self.dims];
                let score = dot_product_safe(query, vec_slice);
                
                if !score.is_finite() {
                    warn!("Invalid score computed for index {}: {}", i, score);
                    return None;
                }
                
                Some((i, score))
            })
            .collect();

        // Ordenação otimizada e segura
        if results.len() > top_k {
            results.select_nth_unstable_by(top_k, |a, b| b.1.total_cmp(&a.1));
            results.truncate(top_k);
        }

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        debug!("Returning {} results", results.len());
        results
    }

    // ========== MÉTODO PARA OBTER MÉTRICAS ==========
    pub fn get_metrics(&self) -> (usize, usize, usize, f64) {
        self.metrics.get_stats()
    }

    // ========== HEALTH CHECK ==========
    pub fn health_check(&self) -> Result<(), String> {
        // Verificar consistência interna
        if self.dims == 0 {
            return Err("Invalid dimensions: 0".into());
        }
        
        if self.total_vectors == 0 {
            return Err("Invalid vector count: 0".into());
        }

        // Verificar tamanhos de projeções
        for (i, proj) in self.projections.iter().enumerate() {
            if proj.len() != self.dims {
                return Err(format!("16-bit projection {} has wrong size: {} vs {}", i, proj.len(), self.dims));
            }
        }

        for (i, proj) in self.main_projections.iter().enumerate() {
            if proj.len() != self.dims {
                return Err(format!("64-bit projection {} has wrong size: {} vs {}", i, proj.len(), self.dims));
            }
        }

        // Verificar buckets
        let mut total_items = 0;
        for bucket in self.main_buckets64.values() {
            total_items += bucket.len();
            for &idx in bucket.iter() {
                if idx >= self.total_vectors {
                    return Err(format!("Invalid index in bucket: {} >= {}", idx, self.total_vectors));
                }
            }
        }

        info!("Health check passed: {} dims, {} vectors, {} total bucket items", 
              self.dims, self.total_vectors, total_items);
        Ok(())
    }

    // ========== PERSISTÊNCIA COM VERSIONAMENTO E VALIDAÇÃO ==========
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Health check antes de salvar
        if let Err(e) = self.health_check() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
        }

        let mut file = BufWriter::new(File::create(path)?);

        // Header com magic number e versão
        file.write_all(b"NSEEKANN")?;
        file.write_all(&4u32.to_le_bytes())?; // Version 4 com melhorias de segurança

        // Metadados básicos
        file.write_all(&(self.dims as u32).to_le_bytes())?;
        file.write_all(&(self.bits as u32).to_le_bytes())?;
        file.write_all(&(self.main_bits as u32).to_le_bytes())?;
        file.write_all(&(self.multi_bits32 as u32).to_le_bytes())?;
        file.write_all(&(self.num_tables as u32).to_le_bytes())?;
        file.write_all(&(self.total_vectors as u32).to_le_bytes())?; // NOVO
        file.write_all(&self.seed.to_le_bytes())?;

        // Config
        file.write_all(&(self.config.target_candidates as u32).to_le_bytes())?;
        file.write_all(&(self.config.min_candidates as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_candidates as u32).to_le_bytes())?; // NOVO
        file.write_all(&(self.config.max_hamming_64 as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_hamming_32 as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_hamming_16 as u32).to_le_bytes())?;
        file.write_all(&(self.config.random_sample_size as u32).to_le_bytes())?;
        file.write_all(&[if self.config.enable_metrics { 1u8 } else { 0u8 }])?; // NOVO

        // 1. Projeções 16-bit (compatibilidade)
        file.write_all(&(self.projections.len() as u32).to_le_bytes())?;
        for proj in &self.projections {
            for &val in proj {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // 2. Buckets 16-bit (compatibilidade) - converter SafeBucket para compatibilidade
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

        // 4. Buckets 64-bit (tabela principal) - SafeBucket format
        file.write_all(&(self.main_buckets64.len() as u32).to_le_bytes())?;
        for (&hash, bucket) in &self.main_buckets64 {
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&(bucket.len() as u32).to_le_bytes())?;
            file.write_all(&[if bucket.is_full { 1u8 } else { 0u8 }])?; // NOVO: flag de bucket cheio
            for &id in bucket.iter() {
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

        // 6. Multi-tabelas 32-bit - SafeBucket format
        file.write_all(&(self.multi_tables32.len() as u32).to_le_bytes())?;
        for table in &self.multi_tables32 {
            file.write_all(&(table.len() as u32).to_le_bytes())?;
            for (&hash, bucket) in table {
                file.write_all(&hash.to_le_bytes())?;
                file.write_all(&(bucket.len() as u32).to_le_bytes())?;
                file.write_all(&[if bucket.is_full { 1u8 } else { 0u8 }])?; // NOVO
                for &id in bucket.iter() {
                    file.write_all(&(id as u32).to_le_bytes())?;
                }
            }
        }
        
        info!("ANN index v4 saved successfully with safety improvements");
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
                4 => Self::load_version_4(file),
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

    fn load_version_4<R: Read>(mut file: R) -> std::io::Result<Self> {
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        let mut u8_buf = [0u8; 1];

        // Metadados básicos
        file.read_exact(&mut u32_buf)?; let dims = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let bits = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let main_bits = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let multi_bits32_val = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let num_tables = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let total_vectors = u32::from_le_bytes(u32_buf) as usize; // NOVO
        file.read_exact(&mut u64_buf)?; let seed = u64::from_le_bytes(u64_buf);

        // Validações de entrada
        if dims == 0 || dims > 10000 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Invalid dimensions: {}", dims)));
        }
        if total_vectors == 0 || total_vectors > 100_000_000 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Invalid vector count: {}", total_vectors)));
        }

        // Config
        file.read_exact(&mut u32_buf)?; let target_candidates = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let min_candidates = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_candidates = u32::from_le_bytes(u32_buf) as usize; // NOVO
        file.read_exact(&mut u32_buf)?; let max_hamming_64 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_hamming_32 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let max_hamming_16 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?; let random_sample_size = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u8_buf)?; let enable_metrics = u8_buf[0] != 0; // NOVO

        let config = AnnConfig {
            target_candidates,
            min_candidates,
            max_candidates,
            max_hamming_64,
            max_hamming_32,
            max_hamming_16,
            random_sample_size,
            enable_metrics,
        };

        // Validações de config
        if min_candidates > target_candidates || target_candidates > max_candidates {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                "Invalid candidate configuration"));
        }

        // 1. Projeções 16-bit
        file.read_exact(&mut u32_buf)?; let num_proj16 = u32::from_le_bytes(u32_buf) as usize;
        if num_proj16 > 100 { // Validação de limite razoável
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Too many 16-bit projections: {}", num_proj16)));
        }
        
        let mut projections = vec![vec![0f32; dims]; num_proj16];
        for proj in &mut projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
                if !val.is_finite() {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                        "Invalid projection value"));
                }
            }
        }

        // 2. Buckets 16-bit
        file.read_exact(&mut u32_buf)?; let num_buckets16 = u32::from_le_bytes(u32_buf) as usize;
        let mut buckets = HashMap::with_capacity(num_buckets16);
        for _ in 0..num_buckets16 {
            let hash = read_u16(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
            
            if len > MAX_BUCKET_SIZE {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                    format!("Bucket too large: {}", len)));
            }
            
            let mut ids = SmallVec::with_capacity(len);
            for _ in 0..len {
                let id = read_u32(&mut file)? as usize;
                if id >= total_vectors {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                        format!("Invalid bucket index: {} >= {}", id, total_vectors)));
                }
                ids.push(id);
            }
            buckets.insert(hash, ids);
        }

        // 3. Projeções 64-bit
        file.read_exact(&mut u32_buf)?; let num_proj64 = u32::from_le_bytes(u32_buf) as usize;
        if num_proj64 > 100 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Too many 64-bit projections: {}", num_proj64)));
        }
        
        let mut main_projections = vec![vec![0f32; dims]; num_proj64];
        for proj in &mut main_projections {
            for val in proj.iter_mut() {
                *val = read_f32(&mut file)?;
                if !val.is_finite() {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                        "Invalid main projection value"));
                }
            }
        }

        // 4. Buckets 64-bit - SafeBucket format
        file.read_exact(&mut u32_buf)?; let num_buckets64 = u32::from_le_bytes(u32_buf) as usize;
        let mut main_buckets64 = HashMap::with_capacity(num_buckets64);
        for _ in 0..num_buckets64 {
            let hash = read_u64(&mut file)?;
            file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u8_buf)?; let is_full = u8_buf[0] != 0;
            
            if len > MAX_BUCKET_SIZE {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                    format!("Main bucket too large: {}", len)));
            }
            
            let mut bucket = SafeBucket::new();
            bucket.is_full = is_full;
            for _ in 0..len {
                let id = read_u32(&mut file)? as usize;
                if id >= total_vectors {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                        format!("Invalid main bucket index: {} >= {}", id, total_vectors)));
                }
                bucket.items.push(id);
            }
            main_buckets64.insert(hash, bucket);
        }

        // 5. Multi-projeções 32-bit
        file.read_exact(&mut u32_buf)?; let num_multi_proj = u32::from_le_bytes(u32_buf) as usize;
        if num_multi_proj != num_tables {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                "Multi-projection count mismatch"));
        }
        
        let mut multi_projections = Vec::with_capacity(num_multi_proj);
        for _ in 0..num_multi_proj {
            file.read_exact(&mut u32_buf)?; let table_proj_count = u32::from_le_bytes(u32_buf) as usize;
            if table_proj_count > 100 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                    format!("Too many table projections: {}", table_proj_count)));
            }
            
            let mut table_projections = vec![vec![0f32; dims]; table_proj_count];
            for proj in &mut table_projections {
                for val in proj.iter_mut() {
                    *val = read_f32(&mut file)?;
                    if !val.is_finite() {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                            "Invalid multi projection value"));
                    }
                }
            }
            multi_projections.push(table_projections);
        }

        // 6. Multi-tabelas 32-bit - SafeBucket format
        file.read_exact(&mut u32_buf)?; let num_multi_tables = u32::from_le_bytes(u32_buf) as usize;
        if num_multi_tables != num_tables {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                "Multi-table count mismatch"));
        }
        
        let mut multi_tables32 = Vec::with_capacity(num_multi_tables);
        for _ in 0..num_multi_tables {
            file.read_exact(&mut u32_buf)?; let table_size = u32::from_le_bytes(u32_buf) as usize;
            let mut table = HashMap::with_capacity(table_size);
            for _ in 0..table_size {
                let hash = read_u32(&mut file)?;
                file.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf) as usize;
                file.read_exact(&mut u8_buf)?; let is_full = u8_buf[0] != 0;
                
                if len > MAX_BUCKET_SIZE {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                        format!("Multi bucket too large: {}", len)));
                }
                
                let mut bucket = SafeBucket::new();
                bucket.is_full = is_full;
                for _ in 0..len {
                    let id = read_u32(&mut file)? as usize;
                    if id >= total_vectors {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                            format!("Invalid multi bucket index: {} >= {}", id, total_vectors)));
                    }
                    bucket.items.push(id);
                }
                table.insert(hash, bucket);
            }
            multi_tables32.push(table);
        }

        let index = Self {
            dims,
            bits,
            projections,
            buckets,
            seed,
            config,
            total_vectors,
            metrics: AnnMetrics::default(),
            main_projections,
            main_bits,
            main_buckets64,
            num_tables,
            multi_projections,
            // CORRIGIDO: em load usamos o valor lido (multi_bits32_val)
            multi_bits32: multi_bits32_val,
            multi_tables32,
        };

        // Health check pós-carregamento
        if let Err(e) = index.health_check() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Health check failed: {}", e)));
        }

        info!("ANN index v4 loaded successfully: {} dims, {} vectors, {} tables", 
              dims, total_vectors, num_tables);

        Ok(index)
    }

    // Manter métodos de compatibilidade para versões antigas
    fn load_version_3<R: Read>(_file: R) -> std::io::Result<Self> {  // CORRIGIDO: _file
        warn!("Loading ANN v3 format - migrating to v4 safety features");
        Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
            "Please rebuild ANN index for v4 safety features"))
    }

    fn load_version_2<R: Read>(_file: R) -> std::io::Result<Self> {  // CORRIGIDO: _file
        warn!("ANN v2 format deprecated - please rebuild index");
        Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
            "Please rebuild ANN index for v4 safety features"))
    }

    fn load_legacy_format<R: Read>(_file: R) -> std::io::Result<Self> {  // CORRIGIDO: _file
        warn!("Legacy ANN format deprecated - please rebuild index");
        Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
            "Please rebuild ANN index for v4 safety features"))
    }
}

// ========== HELPERS SEGUROS ==========
#[inline]
fn dot_product_safe(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        error!("Dot product dimension mismatch: {} vs {}", a.len(), b.len());
        return 0.0;
    }
    
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    
    // Loop unrolling seguro
    for i in 0..chunks {
        let base = i * 4;
        let chunk_sum = a[base] * b[base] 
                      + a[base + 1] * b[base + 1]
                      + a[base + 2] * b[base + 2] 
                      + a[base + 3] * b[base + 3];
        
        if !chunk_sum.is_finite() {
            error!("Invalid chunk sum at index {}", i);
            continue;
        }
        sum += chunk_sum;
    }
    
    // Remainder
    for i in (chunks * 4)..a.len() {
        let product = a[i] * b[i];
        if product.is_finite() {
            sum += product;
        }
    }
    
    if !sum.is_finite() {
        warn!("Dot product resulted in non-finite value, returning 0.0");
        return 0.0;
    }
    
    sum
}

// Helper functions para leitura de dados com validação
#[inline] 
fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4]; 
    r.read_exact(&mut buf)?; 
    let val = f32::from_le_bytes(buf);
    if !val.is_finite() && val != 0.0 { // Permitir zero mas não NaN/Inf
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
            format!("Invalid f32 value: {}", val)));
    }
    Ok(val)
}

#[inline] 
fn read_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2]; 
    r.read_exact(&mut buf)?; 
    Ok(u16::from_le_bytes(buf))
}

#[inline] 
fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4]; 
    r.read_exact(&mut buf)?; 
    Ok(u32::from_le_bytes(buf))
}

#[inline] 
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8]; 
    r.read_exact(&mut buf)?; 
    Ok(u64::from_le_bytes(buf))
}
