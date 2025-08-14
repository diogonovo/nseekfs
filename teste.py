#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-of-the-Art IVF-PQ Benchmark for NSeekFS
==============================================

Benchmark científico para validar performance de produção:
• IVF-PQ vs Exact search 
• Datasets realisticamente clusterizados
• Métricas padrão da indústria
• Memory footprint analysis
• Scaling behavior validation

Expected Results (IVF-PQ):
- Recall@10: 93-97%
- Speedup: 50-300x
- Memory: 6-8 bytes/vector (256x compression)
- Build time: 2-5x slower (one-time cost)
"""

import time
import psutil
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from nseekfs.highlevel import VectorSearch

# ============= CONFIGURATION =============
SIZES = [2_000, 5_000]              # Smaller sizes for faster testing
DIMS = 384                          # Standard embedding dimension
N_CLUSTERS = 16                     # Reduced clusters for speed
QUERIES_BATCH = 50                  # Fewer queries for speed
GT_SAMPLE = 30                      # Smaller GT sample
KS = [1, 5, 10]                    # Reduced k values
LEVEL = "f32"                       # Full precision
NORMALIZED = True                   # Cosine similarity
SEED = 42                          # Reproducibility
MEMORY_LIMIT_GB = 8                # Safety limit for your PC
# =========================================

@dataclass
class BenchmarkResult:
    size: int
    k: int
    # Performance
    ann_latency_ms: float
    exact_latency_ms: float
    speedup: float
    recall: float
    # Memory
    ann_memory_mb: float
    exact_memory_mb: float
    compression_ratio: float
    # Build
    ann_build_time_s: float
    exact_build_time_s: float
    build_overhead: float

def memory_usage_mb() -> float:
    """Current process memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 * 1024)

def check_memory_safety(vectors_count: int, dims: int) -> bool:
    """Check if dataset fits safely in memory"""
    vector_size_gb = (vectors_count * dims * 4) / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    needed_gb = vector_size_gb * 3  # Vectors + index + working memory
    
    if needed_gb > MEMORY_LIMIT_GB:
        print(f"⚠️  Skipping {vectors_count:,} vectors: needs {needed_gb:.1f}GB > limit {MEMORY_LIMIT_GB}GB")
        return False
    
    if needed_gb > available_gb * 0.8:
        print(f"⚠️  Skipping {vectors_count:,} vectors: needs {needed_gb:.1f}GB > available {available_gb:.1f}GB")
        return False
    
    return True

def create_clustered_dataset(n_vectors: int, dims: int, n_clusters: int, seed: int) -> np.ndarray:
    """Create realistic clustered embeddings similar to SBERT/OpenAI"""
    rng = np.random.default_rng(seed)
    
    # Create cluster centers with some sparsity (realistic for neural embeddings)
    centers = rng.normal(0, 1, size=(n_clusters, dims)).astype(np.float32)
    sparsity_mask = rng.random(size=(n_clusters, dims)) < 0.1  # 10% sparsity
    centers = centers * (~sparsity_mask)
    
    # Assign vectors to clusters with some noise
    cluster_assignments = rng.integers(0, n_clusters, size=n_vectors)
    vectors = centers[cluster_assignments] + rng.normal(0, 0.25, size=(n_vectors, dims))
    vectors = vectors.astype(np.float32)
    
    # Normalize for cosine similarity (standard for embeddings)
    if NORMALIZED:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)
    
    return vectors

def benchmark_index_type(vectors: np.ndarray, queries: np.ndarray, tmpdir: Path, 
                        use_ann: bool, size: int) -> Tuple[float, float, float, str]:
    """Benchmark either ANN or exact search"""
    print(f"  {'ANN' if use_ann else 'Exact'} search...")
    
    # Build index
    build_start = time.perf_counter()
    bin_path = VectorSearch.create_index(
        embeddings=vectors,
        level=LEVEL,
        normalized=NORMALIZED,
        ann=use_ann,
        base_name=f"bench_{size}_{'ann' if use_ann else 'exact'}",
        output_dir=tmpdir,
        seed=SEED,
    )
    build_time = time.perf_counter() - build_start
    
    # Load engine
    engine = VectorSearch.load_index(str(bin_path), normalized=NORMALIZED, ann=use_ann, level=LEVEL)
    
    # Measure memory usage
    memory_mb = memory_usage_mb()
    
    # Measure query latency
    method = "ann" if use_ann else "exact"
    latencies = []
    
    for i in range(len(queries)):
        query_start = time.perf_counter()
        results = engine.query(queries[i], 10, method=method, return_scores=False)
        query_time = time.perf_counter() - query_start
        latencies.append(query_time * 1000)  # Convert to ms
        
        # Sanity check
        if len(results) == 0:
            print(f"⚠️  Warning: No results for query {i}")
    
    avg_latency = np.mean(latencies) if latencies else float('inf')
    
    return build_time, avg_latency, memory_mb, str(bin_path)

def calculate_recall(ann_engine, exact_engine, queries: np.ndarray, k: int, sample_size: int) -> float:
    """Calculate recall@k using ground truth from exact search"""
    if len(queries) == 0:
        return 0.0
    
    sample_indices = np.linspace(0, len(queries)-1, num=min(sample_size, len(queries)), dtype=int)
    recalls = []
    
    for i in sample_indices:
        # Get ground truth (exact)
        gt_results = exact_engine.query(queries[i], k, method="exact", return_scores=False)
        gt_ids = {r["idx"] for r in gt_results}
        
        # Get ANN results
        ann_results = ann_engine.query(queries[i], k, method="ann", return_scores=False)
        ann_ids = {r["idx"] for r in ann_results}
        
        # Calculate recall
        if len(gt_ids) > 0:
            recall = len(gt_ids.intersection(ann_ids)) / len(gt_ids)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0

def benchmark_dataset_size(size: int, tmpdir: Path) -> List[BenchmarkResult]:
    """Comprehensive benchmark for a given dataset size"""
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARKING {size:,} vectors × {DIMS} dimensions")
    print(f"{'='*60}")
    
    if not check_memory_safety(size, DIMS):
        return []
    
    # Create dataset
    print("🔧 Creating clustered dataset...")
    vectors = create_clustered_dataset(size, DIMS, N_CLUSTERS, SEED)
    queries = create_clustered_dataset(QUERIES_BATCH, DIMS, N_CLUSTERS, SEED + 1)
    
    baseline_memory = memory_usage_mb()
    print(f"📏 Dataset: {vectors.nbytes / (1024*1024):.1f}MB, Baseline memory: {baseline_memory:.1f}MB")
    
    # Benchmark exact search
    exact_build_time, exact_latency, exact_memory, exact_path = benchmark_index_type(
        vectors, queries, tmpdir, use_ann=False, size=size
    )
    
    # Benchmark ANN search  
    ann_build_time, ann_latency, ann_memory, ann_path = benchmark_index_type(
        vectors, queries, tmpdir, use_ann=True, size=size
    )
    
    # Load both engines for recall calculation
    print("🎯 Calculating recall metrics...")
    ann_engine = VectorSearch.load_index(ann_path, normalized=NORMALIZED, ann=True, level=LEVEL)
    exact_engine = VectorSearch.load_index(exact_path, normalized=NORMALIZED, ann=False, level=LEVEL)
    
    results = []
    
    for k in KS:
        recall = calculate_recall(ann_engine, exact_engine, queries, k, GT_SAMPLE)
        
        # Calculate metrics
        speedup = exact_latency / ann_latency if ann_latency > 0 else float('inf')
        exact_mem_adj = exact_memory - baseline_memory
        ann_mem_adj = ann_memory - baseline_memory
        compression_ratio = exact_mem_adj / ann_mem_adj if ann_mem_adj > 0 else 1.0
        build_overhead = ann_build_time / exact_build_time if exact_build_time > 0 else 1.0
        
        result = BenchmarkResult(
            size=size,
            k=k,
            ann_latency_ms=ann_latency,
            exact_latency_ms=exact_latency,
            speedup=speedup,
            recall=recall,
            ann_memory_mb=ann_mem_adj,
            exact_memory_mb=exact_mem_adj,
            compression_ratio=compression_ratio,
            ann_build_time_s=ann_build_time,
            exact_build_time_s=exact_build_time,
            build_overhead=build_overhead
        )
        
        results.append(result)
        
        # Print immediate results
        print(f"  k={k:2d} | Recall: {recall*100:5.1f}% | Speedup: {speedup:6.1f}× | "
              f"ANN: {ann_latency:.2f}ms | Exact: {exact_latency:.2f}ms")
    
    return results

def print_summary_report(all_results: List[BenchmarkResult]):
    """Print comprehensive summary report"""
    print(f"\n{'='*80}")
    print("🏆 STATE-OF-THE-ART PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    # Group by size
    size_groups: Dict[int, List[BenchmarkResult]] = {}
    for result in all_results:
        size_groups.setdefault(result.size, []).append(result)
    
    # Summary table
    print(f"\n{'Size':>8} {'k':>3} {'Recall':>8} {'Speedup':>9} {'ANN ms':>8} {'Compression':>12} {'Build×':>7}")
    print("-" * 70)
    
    for size in sorted(size_groups.keys()):
        results = size_groups[size]
        for r in results:
            print(f"{r.size:>8,} {r.k:>3} {r.recall*100:>7.1f}% "
                  f"{r.speedup:>8.1f}× {r.ann_latency_ms:>7.2f} "
                  f"{r.compression_ratio:>11.1f}× {r.build_overhead:>6.1f}×")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT")
    print("-" * 40)
    
    for size in sorted(size_groups.keys()):
        results = size_groups[size]
        r10 = next((r for r in results if r.k == 10), None)
        if r10:
            status = "✅ EXCELLENT" if r10.recall >= 0.90 else \
                    "🟡 GOOD" if r10.recall >= 0.75 else \
                    "🔴 POOR"
            
            print(f"{size:>8,} vectors: {status} "
                  f"(recall@10: {r10.recall*100:.1f}%, speedup: {r10.speedup:.0f}×)")
    
    # Memory efficiency
    print(f"\n💾 MEMORY EFFICIENCY")
    print("-" * 40)
    
    for size in sorted(size_groups.keys()):
        results = size_groups[size]
        if results:
            r = results[0]  # All k values have same memory
            bytes_per_vector = (r.ann_memory_mb * 1024 * 1024) / r.size
            print(f"{r.size:>8,} vectors: {r.ann_memory_mb:>6.1f}MB "
                  f"({bytes_per_vector:>4.1f} bytes/vector, {r.compression_ratio:.0f}× compression)")

def main():
    """Main benchmark execution"""
    print("🚀 NSeekFS State-of-the-Art IVF-PQ Benchmark")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    print(f"🎯 Memory limit: {MEMORY_LIMIT_GB}GB")
    
    tmpdir = Path(tempfile.mkdtemp(prefix="nseekfs_state_of_art_"))
    all_results: List[BenchmarkResult] = []
    
    try:
        start_time = time.time()
        
        for size in SIZES:
            results = benchmark_dataset_size(size, tmpdir)
            all_results.extend(results)
            
            # Memory cleanup between sizes
            import gc
            gc.collect()
        
        total_time = time.time() - start_time
        
        if all_results:
            print_summary_report(all_results)
            
            print(f"\n⏱️  Total benchmark time: {total_time:.1f}s")
            print(f"🗂️  Temporary files: {tmpdir}")
            
            # Final verdict
            excellent_results = sum(1 for r in all_results if r.k == 10 and r.recall >= 0.90)
            total_k10_results = sum(1 for r in all_results if r.k == 10)
            
            if excellent_results == total_k10_results:
                print(f"\n🏆 VERDICT: STATE-OF-THE-ART PERFORMANCE ACHIEVED!")
                print(f"    All {total_k10_results} test sizes achieved >90% recall@10")
            elif excellent_results > 0:
                print(f"\n🟡 VERDICT: GOOD PERFORMANCE")
                print(f"    {excellent_results}/{total_k10_results} test sizes achieved >90% recall@10")
            else:
                print(f"\n🔴 VERDICT: NEEDS IMPROVEMENT")
                print(f"    No test sizes achieved >90% recall@10")
        else:
            print("\n❌ No benchmark results generated")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass

if __name__ == "__main__":
    main()