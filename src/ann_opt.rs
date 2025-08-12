use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, BufReader, Write, Read};
use std::path::Path;
use std::collections::{HashMap, HashSet};
use smallvec::SmallVec;
use dashmap::DashMap;

// ---------- Compatibilidade: mantém os campos públicos originais ----------
#[derive(Clone)]
pub struct AnnIndex {
    pub dims: usize,
    pub bits: usize, // continua a representar o nº de bits "base"
    pub projections: Vec<Vec<f32>>, // usado como 1ª tabela 16-bit (compat)
    pub buckets: HashMap<u16, SmallVec<[usize; 16]>>, // compat: 1ª tabela

    // ---------- Novos campos (internos) ----------
    seed: u64,
    num_tables: usize, // nº de multi-tabelas 32-bit
    // Tabela principal 64-bit
    main_projections: Vec<Vec<f32>>,                 // [bits64][dims]
    main_bits: usize,                                // nº de bits efetivos na 64-bit
    main_buckets64: HashMap<u64, SmallVec<[usize; 32]>>,
    // Multi-tabelas 32-bit
    multi_projections: Vec<Vec<Vec<f32>>>,           // [table][bits32][dims]
    multi_bits32: usize,
    multi_tables32: Vec<HashMap<u32, SmallVec<[usize; 16]>>>,
}

impl AnnIndex {
    // ---------- Parâmetros adaptativos ----------
    fn calculate_optimal_config(rows: usize, _dims: usize, req_bits: usize) -> (usize, usize, usize) {
        // devolve: (bits16_compat, bits32_multi, num_tables)
        // bits16_compat: para manter o espaço u16 (até 16)
        let bits16 = req_bits.min(16);
        let (bits32, tables) = match rows {
            0..=1_000           => (12, 2),
            1_001..=10_000      => (16, 3),
            10_001..=100_000    => (20, 4),
            100_001..=1_000_000 => (24, 5),
            _                   => (28, 6),
        };
        (bits16, bits32, tables)
    }

    // ---------- Hash helpers ----------
    #[inline]
    fn hash_signs_u16(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u16 {
        let mut h = 0u16;
        for (j, proj) in projections.iter().enumerate().take(bits.min(16)) {
            let dot = dot_product(vec, proj);
            if dot >= 0.0 { h |= 1 << j; }
        }
        h
    }
    #[inline]
    fn hash_signs_u32(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u32 {
        let mut h = 0u32;
        for (j, proj) in projections.iter().enumerate().take(bits.min(32)) {
            let dot = dot_product(vec, proj);
            if dot >= 0.0 { h |= 1 << j; }
        }
        h
    }
    #[inline]
    fn hash_signs_u64(vec: &[f32], projections: &[Vec<f32>], bits: usize) -> u64 {
        let mut h = 0u64;
        for (j, proj) in projections.iter().enumerate().take(bits.min(64)) {
            let dot = dot_product(vec, proj);
            if dot >= 0.0 { h |= 1 << j; }
        }
        h
    }

    // ---------- Build ----------
    pub fn build(vectors: &[f32], dims: usize, rows: usize, bits: usize, seed: u64) -> Self {
        assert_eq!(vectors.len(), dims * rows, "Invalid vector length");

        let (bits16, bits32, num_tables) = Self::calculate_optimal_config(rows, dims, bits);

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // 1) Compat: 1ª tabela 16-bit (mantém campos públicos existentes)
        let projections_16: Vec<Vec<f32>> = (0..bits16)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let buckets_u16 = DashMap::<u16, SmallVec<[usize; 16]>>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let v = &vectors[i * dims..(i + 1) * dims];
            let h = Self::hash_signs_u16(v, &projections_16, bits16);
            buckets_u16.entry(h).or_default().push(i);
        });
        let buckets_compat = buckets_u16.into_iter().collect::<HashMap<_, _>>();

        // 2) Tabela principal 64-bit (mais projeções)
        let bits64 = (bits32 * 2).min(64).max(24); // regra simples: mais resolução na principal
        let main_projections: Vec<Vec<f32>> = (0..bits64)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        let main_buckets = DashMap::<u64, SmallVec<[usize; 32]>>::new();
        (0..rows).into_par_iter().for_each(|i| {
            let v = &vectors[i * dims..(i + 1) * dims];
            let h = Self::hash_signs_u64(v, &main_projections, bits64);
            main_buckets.entry(h).or_default().push(i);
        });
        let main_buckets64 = main_buckets.into_iter().collect::<HashMap<_, _>>();

        // 3) Multi-tabelas 32-bit (com seeds determinísticos por tabela)
        let mut multi_projections: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_tables);
        let mut multi_tables32: Vec<HashMap<u32, SmallVec<[usize; 16]>>> = Vec::with_capacity(num_tables);

        (0..num_tables).into_par_iter().map(|tidx| {
            let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add(1_000 * (tidx as u64)));
            let proj_table: Vec<Vec<f32>> = (0..bits32)
                .map(|_| (0..dims).map(|_| normal.sample(&mut local_rng) as f32).collect())
                .collect();

            let buckets32 = DashMap::<u32, SmallVec<[usize; 16]>>::new();
            (0..rows).into_par_iter().for_each(|i| {
                let v = &vectors[i * dims..(i + 1) * dims];
                let h = Self::hash_signs_u32(v, &proj_table, bits32);
                buckets32.entry(h).or_default().push(i);
            });
            let table_map = buckets32.into_iter().collect::<HashMap<_, _>>();
            (proj_table, table_map)
        }).collect::<Vec<_>>()
        .into_iter()
        .for_each(|(p, m)| {
            multi_projections.push(p);
            multi_tables32.push(m);
        });

        Self {
            dims,
            bits: bits16,
            projections: projections_16,
            buckets: buckets_compat,

            seed,
            num_tables,
            main_projections,
            main_bits: bits64,
            main_buckets64,
            multi_projections,
            multi_bits32: bits32,
            multi_tables32,
        }
    }

    // ---------- Query ----------
    pub fn query_candidates(&self, query: &[f32]) -> Vec<usize> {
        assert_eq!(query.len(), self.dims);

        // parâmetros de multi-probe (seguro e rápido por omissão)
        const MIN_CANDS: usize = 64;     // alvo mínimo de candidatos
        const MAX_FLIPS_64: usize = 20;  // raio parcial para 64-bit
        const MAX_FLIPS_32: usize = 12;  // raio parcial para 32-bit
        const MAX_RADIUS16: usize = 2;   // raio completo (1..=2) para a 16-bit

        let mut set: HashSet<usize> = HashSet::new();

        // 1) Tabela principal 64-bit
        if self.main_bits > 0 && !self.main_projections.is_empty() {
            let h64 = Self::hash_signs_u64(query, &self.main_projections, self.main_bits);
            if let Some(cands) = self.main_buckets64.get(&h64) {
                set.extend(cands.iter().cloned());
            }
            // Multi-probe Hamming(1) no 64-bit até atingir o mínimo
            if set.len() < MIN_CANDS {
                let limit = self.main_bits.min(MAX_FLIPS_64);
                for bit in 0..limit {
                    let fh = h64 ^ (1u64 << bit);
                    if let Some(cands) = self.main_buckets64.get(&fh) {
                        set.extend(cands.iter().cloned());
                        if set.len() >= MIN_CANDS { break; }
                    }
                }
            }
        }

        // 2) Multi-tabelas 32-bit (hash exato + Hamming(1) leve se necessário)
        if self.multi_bits32 > 0 && !self.multi_projections.is_empty() {
            for (tidx, table) in self.multi_tables32.iter().enumerate() {
                let h32 = Self::hash_signs_u32(query, &self.multi_projections[tidx], self.multi_bits32);
                if let Some(cands) = table.get(&h32) {
                    set.extend(cands.iter().cloned());
                }
                if set.len() < MIN_CANDS {
                    let limit = self.multi_bits32.min(MAX_FLIPS_32);
                    for bit in 0..limit {
                        let fh = h32 ^ (1u32 << bit);
                        if let Some(cands) = table.get(&fh) {
                            set.extend(cands.iter().cloned());
                            if set.len() >= MIN_CANDS { break; }
                        }
                    }
                }
                if set.len() >= MIN_CANDS { break; }
            }
        }

        // 3) Compat: 16-bit (tabela histórica)
        let h16 = Self::hash_signs_u16(query, &self.projections, self.bits);
        if let Some(cands) = self.buckets.get(&h16) {
            set.extend(cands.iter().cloned());
        }

        // 4) Multi-probe na 16-bit se ainda for pouco (raios 1 e 2)
        if set.len() < MIN_CANDS && self.bits > 0 {
            let bmax = self.bits.min(16);

            // raio 1
            for i in 0..bmax {
                let h = h16 ^ (1u16 << i);
                if let Some(v) = self.buckets.get(&h) {
                    set.extend(v.iter().cloned());
                    if set.len() >= MIN_CANDS { break; }
                }
            }

            // raio 2 (cuidado com custo quadrático; limitado por bmax)
            if set.len() < MIN_CANDS && MAX_RADIUS16 >= 2 {
                for i in 0..bmax {
                    for j in (i + 1)..bmax {
                        let h = h16 ^ (1u16 << i) ^ (1u16 << j);
                        if let Some(v) = self.buckets.get(&h) {
                            set.extend(v.iter().cloned());
                            if set.len() >= MIN_CANDS { break; }
                        }
                    }
                    if set.len() >= MIN_CANDS { break; }
                }
            }
        }

        let mut out: Vec<usize> = set.into_iter().collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn query(&self, query: &[f32], top_k: usize, vectors: &[f32]) -> Vec<(usize, f32)> {
        let candidates = self.query_candidates(query);
        let mut results: Vec<(usize, f32)> = candidates
            .par_iter()
            .map(|&i| {
                let off = i * self.dims;
                let vi = &vectors[off..off + self.dims];
                let score = dot_product(query, vi);
                (i, score)
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);
        results
    }

    // ---------- Persistência ----------
    // Novo formato (V2): "NSEEKANN" + u32 version(2)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);

        // header
        f.write_all(b"NSEEKANN")?;
        f.write_all(&2u32.to_le_bytes())?;

        // básicos
        f.write_all(&(self.dims as u32).to_le_bytes())?;
        f.write_all(&(self.bits as u32).to_le_bytes())?;      // bits16 compat
        f.write_all(&(self.main_bits as u32).to_le_bytes())?; // bits64 principal
        f.write_all(&(self.multi_bits32 as u32).to_le_bytes())?;
        f.write_all(&(self.num_tables as u32).to_le_bytes())?;
        f.write_all(&self.seed.to_le_bytes())?;

        // projections 16-bit (compat)
        f.write_all(&(self.projections.len() as u32).to_le_bytes())?;
        for p in &self.projections {
            for &x in p { f.write_all(&x.to_le_bytes())?; }
        }

        // buckets 16-bit (compat)
        f.write_all(&(self.buckets.len() as u32).to_le_bytes())?;
        for (&h, ids) in &self.buckets {
            f.write_all(&h.to_le_bytes())?;
            f.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids {
                f.write_all(&(id as u32).to_le_bytes())?;
            }
        }

        // main projections 64-bit
        f.write_all(&(self.main_projections.len() as u32).to_le_bytes())?;
        for p in &self.main_projections {
            for &x in p { f.write_all(&x.to_le_bytes())?; }
        }

        // main buckets 64-bit
        f.write_all(&(self.main_buckets64.len() as u32).to_le_bytes())?;
        for (&h, ids) in &self.main_buckets64 {
            f.write_all(&h.to_le_bytes())?;
            f.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids {
                f.write_all(&(id as u32).to_le_bytes())?;
            }
        }

        // multi projections 32-bit + tabelas
        f.write_all(&(self.multi_projections.len() as u32).to_le_bytes())?; // num_tables
        for table_proj in &self.multi_projections {
            f.write_all(&(table_proj.len() as u32).to_le_bytes())?; // bits32
            for p in table_proj {
                for &x in p { f.write_all(&x.to_le_bytes())?; }
            }
        }
        f.write_all(&(self.multi_tables32.len() as u32).to_le_bytes())?;
        for table in &self.multi_tables32 {
            f.write_all(&(table.len() as u32).to_le_bytes())?;
            for (&h, ids) in table {
                f.write_all(&h.to_le_bytes())?;
                f.write_all(&(ids.len() as u32).to_le_bytes())?;
                for &id in ids {
                    f.write_all(&(id as u32).to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, _vectors: &[f32]) -> std::io::Result<Self> {
        let mut f = BufReader::new(File::open(path)?);

        // tenta ler magic
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic == b"NSEEKANN" {
            // novo formato (V2)
            let mut u32b = [0u8; 4];
            let mut u64b = [0u8; 8];

            f.read_exact(&mut u32b)?; let _ver = u32::from_le_bytes(u32b); // 2
            f.read_exact(&mut u32b)?; let dims = u32::from_le_bytes(u32b) as usize;
            f.read_exact(&mut u32b)?; let bits16 = u32::from_le_bytes(u32b) as usize;
            f.read_exact(&mut u32b)?; let bits64 = u32::from_le_bytes(u32b) as usize;
            f.read_exact(&mut u32b)?; let bits32 = u32::from_le_bytes(u32b) as usize;
            f.read_exact(&mut u32b)?; let num_tables = u32::from_le_bytes(u32b) as usize;
            f.read_exact(&mut u64b)?; let seed = u64::from_le_bytes(u64b);

            // projections 16
            f.read_exact(&mut u32b)?; let p16n = u32::from_le_bytes(u32b) as usize;
            let mut projections = vec![vec![0f32; dims]; p16n];
            for p in projections.iter_mut() {
                for x in p.iter_mut() { *x = read_f32(&mut f)?; }
            }

            // buckets 16
            f.read_exact(&mut u32b)?; let b16n = u32::from_le_bytes(u32b) as usize;
            let mut buckets: HashMap<u16, SmallVec<[usize; 16]>> = HashMap::with_capacity(b16n);
            for _ in 0..b16n {
                let h = read_u16(&mut f)?;
                f.read_exact(&mut u32b)?; let len = u32::from_le_bytes(u32b) as usize;
                let mut ids = SmallVec::with_capacity(len);
                for _ in 0..len { ids.push(read_u32(&mut f)? as usize); }
                buckets.insert(h, ids);
            }

            // main projections 64
            f.read_exact(&mut u32b)?; let p64n = u32::from_le_bytes(u32b) as usize;
            let mut main_projections = vec![vec![0f32; dims]; p64n];
            for p in main_projections.iter_mut() {
                for x in p.iter_mut() { *x = read_f32(&mut f)?; }
            }

            // main buckets 64
            f.read_exact(&mut u32b)?; let b64n = u32::from_le_bytes(u32b) as usize;
            let mut main_buckets64: HashMap<u64, SmallVec<[usize; 32]>> = HashMap::with_capacity(b64n);
            for _ in 0..b64n {
                let h = read_u64(&mut f)?;
                f.read_exact(&mut u32b)?; let len = u32::from_le_bytes(u32b) as usize;
                let mut ids = SmallVec::with_capacity(len);
                for _ in 0..len { ids.push(read_u32(&mut f)? as usize); }
                main_buckets64.insert(h, ids);
            }

            // multi projections 32
            f.read_exact(&mut u32b)?; let mt = u32::from_le_bytes(u32b) as usize;
            let mut multi_projections: Vec<Vec<Vec<f32>>> = Vec::with_capacity(mt);
            for _ in 0..mt {
                f.read_exact(&mut u32b)?; let b32 = u32::from_le_bytes(u32b) as usize;
                let mut table_proj = vec![vec![0f32; dims]; b32];
                for p in table_proj.iter_mut() {
                    for x in p.iter_mut() { *x = read_f32(&mut f)?; }
                }
                multi_projections.push(table_proj);
            }

            // multi tables 32
            f.read_exact(&mut u32b)?; let mt2 = u32::from_le_bytes(u32b) as usize;
            let mut multi_tables32: Vec<HashMap<u32, SmallVec<[usize; 16]>>> = Vec::with_capacity(mt2);
            for _ in 0..mt2 {
                f.read_exact(&mut u32b)?; let nb = u32::from_le_bytes(u32b) as usize;
                let mut table: HashMap<u32, SmallVec<[usize; 16]>> = HashMap::with_capacity(nb);
                for _ in 0..nb {
                    let h = read_u32(&mut f)?;
                    f.read_exact(&mut u32b)?; let len = u32::from_le_bytes(u32b) as usize;
                    let mut ids = SmallVec::with_capacity(len);
                    for _ in 0..len { ids.push(read_u32(&mut f)? as usize); }
                    table.insert(h, ids);
                }
                multi_tables32.push(table);
            }

            Ok(Self {
                dims,
                bits: bits16,
                projections,
                buckets,
                seed,
                num_tables,
                main_projections,
                main_bits: bits64,
                main_buckets64,
                multi_projections,
                multi_bits32: bits32,
                multi_tables32,
            })
        } else {
            // formato LEGADO (compat): já lemos 8 bytes que não eram "NSEEKANN"
            // Recuar file pointer e usar o leitor antigo
            use std::io::Seek;
            use std::io::SeekFrom;
            let mut f2 = f;
            f2.seek(SeekFrom::Start(0))?;

            // --- leitor legado (igual ao teu AnnIndex anterior) ---
            let mut u32_buf = [0u8; 4];

            f2.read_exact(&mut u32_buf)?; let dims = u32::from_le_bytes(u32_buf) as usize;
            f2.read_exact(&mut u32_buf)?; let bits = u32::from_le_bytes(u32_buf) as usize;

            let mut projections = vec![vec![0f32; dims]; bits];
            for proj in &mut projections {
                for val in proj.iter_mut() {
                    *val = read_f32(&mut f2)?;
                }
            }

            f2.read_exact(&mut u32_buf)?; let num_buckets = u32::from_le_bytes(u32_buf);
            let mut buckets = HashMap::new();
            for _ in 0..num_buckets {
                let h = read_u16(&mut f2)?;
                f2.read_exact(&mut u32_buf)?; let len = u32::from_le_bytes(u32_buf);
                let mut ids = SmallVec::with_capacity(len as usize);
                for _ in 0..len {
                    ids.push(read_u32(&mut f2)? as usize);
                }
                buckets.insert(h, ids);
            }

            // Para o legado, inicializamos os novos campos de forma mínima
            Ok(Self {
                dims,
                bits,
                projections,
                buckets,
                seed: 42,
                num_tables: 0,
                main_projections: Vec::new(),
                main_bits: 0,
                main_buckets64: HashMap::new(),
                multi_projections: Vec::new(),
                multi_bits32: 0,
                multi_tables32: Vec::new(),
            })
        }
    }
}

// ---------- helpers ----------
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // versão simples; podes trocar por SIMD depois
    a.iter().zip(b).map(|(x,y)| x * y).sum::<f32>()
}
#[inline]
fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8;4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}
#[inline]
fn read_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut b = [0u8;2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}
#[inline]
fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8;4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
#[inline]
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8;8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
