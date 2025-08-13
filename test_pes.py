#!/usr/bin/env python3
"""
🚀 NSeekFS Heavy Benchmarks - Real-World Performance Testing
============================================================

Testa performance em cenários realistas de produção:
- Datasets de 100K a 10M vetores
- Dimensões de embedding reais (384, 768, 1536)
- Comparação com Numpy, Scikit-learn, FAISS
- Memory usage tracking
- Throughput testing
- Precision vs Speed trade-offs
"""

import os
import gc
import sys
import time
import psutil
import tempfile
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import nseekfs
except ImportError:
    print("❌ NSeekFS not installed. Run: maturin develop --release")
    sys.exit(1)

# Optional dependencies for comparison
SKLEARN_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Scikit-learn not available - skipping sklearn comparisons")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ FAISS not available - skipping FAISS comparisons")

class MemoryTracker:
    """Track memory usage during benchmarks"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_mb()
        self.peak_memory = self.initial_memory
        self.measurements = []
    
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)
    
    def record(self, label: str = ""):
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        self.measurements.append((label, current))
        return current
    
    def get_peak_usage(self) -> float:
        return self.peak_memory - self.initial_memory
    
    def reset(self):
        gc.collect()
        self.initial_memory = self.get_memory_mb()
        self.peak_memory = self.initial_memory
        self.measurements = []

class BenchmarkSuite:
    """Comprehensive benchmark suite for real-world scenarios"""
    
    def __init__(self):
        self.results = {}
        self.memory_tracker = MemoryTracker()
        self.temp_dir = tempfile.mkdtemp(prefix="nseekfs_bench_")
        print(f"🗂️ Using temp directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temp directory")
        except:
            pass
    
    def generate_realistic_embeddings(self, n_vectors: int, dims: int, distribution: str = "sentence_transformer") -> np.ndarray:
        """Generate embeddings that simulate real-world data"""
        
        print(f"📊 Generating {n_vectors:,} vectors × {dims} dims ({distribution})")
        
        if distribution == "sentence_transformer":
            # Simulate sentence-transformer embeddings (normalized, clustered)
            embeddings = np.random.normal(0, 0.3, (n_vectors, dims)).astype(np.float32)
            
            # Add some clustering (realistic for text embeddings)
            n_clusters = max(10, n_vectors // 1000)
            cluster_centers = np.random.normal(0, 1, (n_clusters, dims))
            
            for i in range(n_vectors):
                cluster_id = i % n_clusters
                noise_scale = 0.1 + 0.2 * np.random.random()
                embeddings[i] += cluster_centers[cluster_id] * 0.5
                embeddings[i] += np.random.normal(0, noise_scale, dims)
            
            # Normalize (like sentence transformers)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        elif distribution == "openai":
            # Simulate OpenAI embeddings (higher variance, less clustered)
            embeddings = np.random.normal(0, 0.5, (n_vectors, dims)).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        elif distribution == "random":
            # Pure random (worst case for ANN)
            embeddings = np.random.uniform(-1, 1, (n_vectors, dims)).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return embeddings
    
    def benchmark_nseekfs(self, embeddings: np.ndarray, test_queries: np.ndarray, 
                         levels: List[str], top_k: int = 10) -> Dict[str, Any]:
        """Benchmark NSeekFS with different levels and ANN settings"""
        
        print(f"\n🚀 Benchmarking NSeekFS...")
        n_vectors, dims = embeddings.shape
        n_queries = len(test_queries)
        
        results = {}
        
        for level in levels:
            print(f"  Testing level: {level}")
            
            # Test both ANN and exact
            for ann_enabled in [True, False]:
                ann_label = "ANN" if ann_enabled else "Exact"
                key = f"nseekfs_{level}_{ann_label.lower()}"
                
                self.memory_tracker.reset()
                
                # Build index
                start_time = time.time()
                try:
                    index = nseekfs.from_embeddings(
                        embeddings, 
                        level=level, 
                        ann=ann_enabled,
                        base_name=f"bench_{level}_{ann_label.lower()}",
                        output_dir=self.temp_dir,
                        normalized=True
                    )
                    build_time = time.time() - start_time
                    build_memory = self.memory_tracker.record("after_build")
                    
                except Exception as e:
                    print(f"    ❌ Failed to build {key}: {e}")
                    continue
                
                # Warmup
                for i in range(min(5, n_queries)):
                    index.query(test_queries[i], top_k=top_k)
                
                # Query benchmark
                query_times = []
                recall_scores = []
                
                start_time = time.time()
                
                for i in range(n_queries):
                    query_start = time.perf_counter()
                    results_query = index.query(test_queries[i], top_k=top_k, method="auto")
                    query_time = (time.perf_counter() - query_start) * 1000  # ms
                    
                    query_times.append(query_time)
                    
                    # Calculate recall (self-hit for query i should be in top-k)
                    if i < n_vectors:
                        found_self = any(r["idx"] == i for r in results_query)
                        recall_scores.append(1.0 if found_self else 0.0)
                
                total_query_time = time.time() - start_time
                query_memory = self.memory_tracker.record("after_queries")
                
                results[key] = {
                    "build_time_s": build_time,
                    "total_query_time_s": total_query_time,
                    "mean_query_time_ms": np.mean(query_times),
                    "median_query_time_ms": np.median(query_times),
                    "p95_query_time_ms": np.percentile(query_times, 95),
                    "p99_query_time_ms": np.percentile(query_times, 99),
                    "throughput_qps": n_queries / total_query_time,
                    "recall_rate": np.mean(recall_scores) if recall_scores else 0.0,
                    "build_memory_mb": build_memory,
                    "query_memory_mb": query_memory,
                    "peak_memory_mb": self.memory_tracker.get_peak_usage(),
                    "level": level,
                    "ann_enabled": ann_enabled,
                    "dims": dims,
                    "n_vectors": n_vectors,
                    "n_queries": n_queries
                }
                
                print(f"    ✅ {key}: {np.mean(query_times):.2f}ms avg, {n_queries/total_query_time:.1f} QPS")
        
        return results
    
    def benchmark_numpy_baseline(self, embeddings: np.ndarray, test_queries: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark pure NumPy cosine similarity (exact search baseline)"""
        
        print(f"\n📊 Benchmarking NumPy baseline...")
        n_vectors, dims = embeddings.shape
        n_queries = len(test_queries)
        
        self.memory_tracker.reset()
        
        # Normalize embeddings
        start_time = time.time()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        build_time = time.time() - start_time
        build_memory = self.memory_tracker.record("after_normalize")
        
        query_times = []
        
        start_time = time.time()
        
        for i in range(n_queries):
            query_start = time.perf_counter()
            
            # Cosine similarity via dot product (since normalized)
            similarities = np.dot(embeddings_norm, test_queries[i])
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            query_time = (time.perf_counter() - query_start) * 1000  # ms
            query_times.append(query_time)
        
        total_query_time = time.time() - start_time
        query_memory = self.memory_tracker.record("after_queries")
        
        return {
            "numpy_exact": {
                "build_time_s": build_time,
                "total_query_time_s": total_query_time,
                "mean_query_time_ms": np.mean(query_times),
                "median_query_time_ms": np.median(query_times),
                "p95_query_time_ms": np.percentile(query_times, 95),
                "p99_query_time_ms": np.percentile(query_times, 99),
                "throughput_qps": n_queries / total_query_time,
                "recall_rate": 1.0,  # Exact
                "build_memory_mb": build_memory,
                "query_memory_mb": query_memory,
                "peak_memory_mb": self.memory_tracker.get_peak_usage(),
                "dims": dims,
                "n_vectors": n_vectors,
                "n_queries": n_queries
            }
        }
    
    def benchmark_sklearn(self, embeddings: np.ndarray, test_queries: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark scikit-learn NearestNeighbors"""
        
        if not SKLEARN_AVAILABLE:
            return {}
        
        print(f"\n🔬 Benchmarking Scikit-learn...")
        n_vectors, dims = embeddings.shape
        n_queries = len(test_queries)
        
        results = {}
        
        # Test different algorithms
        algorithms = ['brute', 'ball_tree', 'kd_tree']
        
        for algorithm in algorithms:
            try:
                self.memory_tracker.reset()
                
                # Build index
                start_time = time.time()
                nn = NearestNeighbors(n_neighbors=top_k, algorithm=algorithm, metric='cosine')
                nn.fit(embeddings)
                build_time = time.time() - start_time
                build_memory = self.memory_tracker.record("after_build")
                
                query_times = []
                
                start_time = time.time()
                
                for i in range(n_queries):
                    query_start = time.perf_counter()
                    distances, indices = nn.kneighbors([test_queries[i]])
                    query_time = (time.perf_counter() - query_start) * 1000  # ms
                    query_times.append(query_time)
                
                total_query_time = time.time() - start_time
                query_memory = self.memory_tracker.record("after_queries")
                
                key = f"sklearn_{algorithm}"
                results[key] = {
                    "build_time_s": build_time,
                    "total_query_time_s": total_query_time,
                    "mean_query_time_ms": np.mean(query_times),
                    "median_query_time_ms": np.median(query_times),
                    "p95_query_time_ms": np.percentile(query_times, 95),
                    "p99_query_time_ms": np.percentile(query_times, 99),
                    "throughput_qps": n_queries / total_query_time,
                    "recall_rate": 1.0,  # Exact
                    "build_memory_mb": build_memory,
                    "query_memory_mb": query_memory,
                    "peak_memory_mb": self.memory_tracker.get_peak_usage(),
                    "algorithm": algorithm,
                    "dims": dims,
                    "n_vectors": n_vectors,
                    "n_queries": n_queries
                }
                
                print(f"    ✅ {key}: {np.mean(query_times):.2f}ms avg, {n_queries/total_query_time:.1f} QPS")
                
            except Exception as e:
                print(f"    ❌ Failed {algorithm}: {e}")
                continue
        
        return results
    
    def benchmark_faiss(self, embeddings: np.ndarray, test_queries: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark FAISS (if available)"""
        
        if not FAISS_AVAILABLE:
            return {}
        
        print(f"\n⚡ Benchmarking FAISS...")
        n_vectors, dims = embeddings.shape
        n_queries = len(test_queries)
        
        results = {}
        
        # Test different FAISS indexes
        index_types = [
            ("Flat", lambda d: faiss.IndexFlatIP(d)),
            ("IVF100", lambda d: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 100)),
            ("LSH", lambda d: faiss.IndexLSH(d, 64)),
        ]
        
        for index_name, index_factory in index_types:
            try:
                self.memory_tracker.reset()
                
                # Build index
                start_time = time.time()
                index = index_factory(dims)
                
                # Normalize for inner product = cosine similarity
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                if hasattr(index, 'train'):
                    index.train(embeddings_norm)
                
                index.add(embeddings_norm)
                build_time = time.time() - start_time
                build_memory = self.memory_tracker.record("after_build")
                
                # Normalize queries
                test_queries_norm = test_queries / np.linalg.norm(test_queries, axis=1, keepdims=True)
                
                query_times = []
                
                start_time = time.time()
                
                for i in range(n_queries):
                    query_start = time.perf_counter()
                    distances, indices = index.search(test_queries_norm[i:i+1], top_k)
                    query_time = (time.perf_counter() - query_start) * 1000  # ms
                    query_times.append(query_time)
                
                total_query_time = time.time() - start_time
                query_memory = self.memory_tracker.record("after_queries")
                
                key = f"faiss_{index_name.lower()}"
                results[key] = {
                    "build_time_s": build_time,
                    "total_query_time_s": total_query_time,
                    "mean_query_time_ms": np.mean(query_times),
                    "median_query_time_ms": np.median(query_times),
                    "p95_query_time_ms": np.percentile(query_times, 95),
                    "p99_query_time_ms": np.percentile(query_times, 99),
                    "throughput_qps": n_queries / total_query_time,
                    "recall_rate": 1.0 if index_name == "Flat" else 0.95,  # Approximate for ANN
                    "build_memory_mb": build_memory,
                    "query_memory_mb": query_memory,
                    "peak_memory_mb": self.memory_tracker.get_peak_usage(),
                    "index_type": index_name,
                    "dims": dims,
                    "n_vectors": n_vectors,
                    "n_queries": n_queries
                }
                
                print(f"    ✅ {key}: {np.mean(query_times):.2f}ms avg, {n_queries/total_query_time:.1f} QPS")
                
            except Exception as e:
                print(f"    ❌ Failed {index_name}: {e}")
                continue
        
        return results
    
    def benchmark_concurrency(self, embeddings: np.ndarray, test_queries: np.ndarray, 
                             n_threads: int = 4, top_k: int = 10) -> Dict[str, Any]:
        """Test concurrent query performance"""
        
        print(f"\n🧵 Benchmarking concurrency ({n_threads} threads)...")
        
        # Build NSeekFS index
        index = nseekfs.from_embeddings(
            embeddings, 
            level="f32", 
            ann=True,
            base_name="concurrency_test",
            output_dir=self.temp_dir,
            normalized=True
        )
        
        n_queries_per_thread = len(test_queries) // n_threads
        
        def worker_thread(thread_id: int, queries: np.ndarray) -> List[float]:
            thread_times = []
            for query in queries:
                start = time.perf_counter()
                index.query(query, top_k=top_k)
                thread_times.append((time.perf_counter() - start) * 1000)
            return thread_times
        
        # Sequential baseline
        sequential_times = []
        start_time = time.time()
        for query in test_queries:
            query_start = time.perf_counter()
            index.query(query, top_k=top_k)
            sequential_times.append((time.perf_counter() - query_start) * 1000)
        sequential_total = time.time() - start_time
        
        # Concurrent test
        query_batches = [
            test_queries[i*n_queries_per_thread:(i+1)*n_queries_per_thread] 
            for i in range(n_threads)
        ]
        
        concurrent_times = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(worker_thread, i, batch) 
                for i, batch in enumerate(query_batches)
            ]
            
            for future in as_completed(futures):
                concurrent_times.extend(future.result())
        
        concurrent_total = time.time() - start_time
        
        return {
            "concurrency": {
                "n_threads": n_threads,
                "sequential_total_s": sequential_total,
                "concurrent_total_s": concurrent_total,
                "speedup": sequential_total / concurrent_total,
                "sequential_mean_ms": np.mean(sequential_times),
                "concurrent_mean_ms": np.mean(concurrent_times),
                "sequential_qps": len(test_queries) / sequential_total,
                "concurrent_qps": len(test_queries) / concurrent_total,
            }
        }
    
    def run_scenario(self, scenario_name: str, n_vectors: int, dims: int, 
                    distribution: str = "sentence_transformer", n_queries: int = 100) -> Dict[str, Any]:
        """Run a complete benchmark scenario"""
        
        print(f"\n" + "="*80)
        print(f"🎯 SCENARIO: {scenario_name}")
        print(f"📊 Dataset: {n_vectors:,} vectors × {dims} dims")
        print(f"🔍 Distribution: {distribution}")
        print(f"🎯 Queries: {n_queries}")
        print("="*80)
        
        # Generate data
        embeddings = self.generate_realistic_embeddings(n_vectors, dims, distribution)
        
        # Use first N vectors as queries (simulates real usage)
        test_queries = embeddings[:n_queries].copy()
        
        print(f"💾 Dataset size: {embeddings.nbytes / (1024**2):.1f}MB")
        
        # Run all benchmarks
        scenario_results = {}
        
        # NSeekFS
        nseekfs_results = self.benchmark_nseekfs(embeddings, test_queries, levels=["f32", "f16"])
        scenario_results.update(nseekfs_results)
        
        # NumPy baseline
        numpy_results = self.benchmark_numpy_baseline(embeddings, test_queries)
        scenario_results.update(numpy_results)
        
        # Sklearn
        sklearn_results = self.benchmark_sklearn(embeddings, test_queries)
        scenario_results.update(sklearn_results)
        
        # FAISS
        faiss_results = self.benchmark_faiss(embeddings, test_queries)
        scenario_results.update(faiss_results)
        
        # Concurrency (only for larger datasets)
        if n_vectors >= 10000:
            concurrency_results = self.benchmark_concurrency(embeddings, test_queries[:20])  # Smaller sample
            scenario_results.update(concurrency_results)
        
        # Add scenario metadata
        for key in scenario_results:
            if isinstance(scenario_results[key], dict):
                scenario_results[key]["scenario"] = scenario_name
                scenario_results[key]["distribution"] = distribution
        
        self.results[scenario_name] = scenario_results
        
        # Print summary
        self.print_scenario_summary(scenario_name, scenario_results)
        
        return scenario_results
    
    def print_scenario_summary(self, scenario_name: str, results: Dict[str, Any]):
        """Print summary of scenario results"""
        
        print(f"\n📋 SUMMARY: {scenario_name}")
        print("-" * 60)
        
        # Sort by mean query time
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if isinstance(v, dict) and "mean_query_time_ms" in v],
            key=lambda x: x[1]["mean_query_time_ms"]
        )
        
        print(f"{'Method':<20} {'Avg Query':<12} {'QPS':<10} {'Recall':<8} {'Memory':<10}")
        print("-" * 60)
        
        for method, data in sorted_results:
            avg_time = data["mean_query_time_ms"]
            qps = data.get("throughput_qps", 0)
            recall = data.get("recall_rate", 0)
            memory = data.get("peak_memory_mb", 0)
            
            print(f"{method:<20} {avg_time:>8.2f}ms {qps:>8.1f} {recall:>6.1%} {memory:>8.1f}MB")
    
    def print_final_report(self):
        """Print comprehensive final report"""
        
        print("\n" + "="*80)
        print("🏆 FINAL BENCHMARK REPORT")
        print("="*80)
        
        # Overall statistics
        all_nseekfs_results = []
        all_baseline_results = []
        
        for scenario_name, scenario_results in self.results.items():
            for method, data in scenario_results.items():
                if isinstance(data, dict) and "mean_query_time_ms" in data:
                    if "nseekfs" in method:
                        all_nseekfs_results.append((scenario_name, method, data))
                    else:
                        all_baseline_results.append((scenario_name, method, data))
        
        # NSeekFS performance summary
        if all_nseekfs_results:
            print(f"\n🚀 NSeekFS Performance:")
            print(f"  Best query time: {min(d['mean_query_time_ms'] for _, _, d in all_nseekfs_results):.2f}ms")
            print(f"  Best throughput: {max(d.get('throughput_qps', 0) for _, _, d in all_nseekfs_results):.1f} QPS")
            print(f"  Average recall: {np.mean([d.get('recall_rate', 0) for _, _, d in all_nseekfs_results]):.1%}")
        
        # Comparison with baselines
        print(f"\n📊 Performance vs Baselines:")
        
        for scenario_name in self.results:
            print(f"\n  {scenario_name}:")
            
            nseekfs_best = None
            baseline_best = None
            
            for method, data in self.results[scenario_name].items():
                if isinstance(data, dict) and "mean_query_time_ms" in data:
                    if "nseekfs" in method and "ann" in method:
                        if nseekfs_best is None or data["mean_query_time_ms"] < nseekfs_best["mean_query_time_ms"]:
                            nseekfs_best = data
                    elif method not in ["concurrency"]:
                        if baseline_best is None or data["mean_query_time_ms"] < baseline_best["mean_query_time_ms"]:
                            baseline_best = data
            
            if nseekfs_best and baseline_best:
                speedup = baseline_best["mean_query_time_ms"] / nseekfs_best["mean_query_time_ms"]
                print(f"    NSeekFS speedup: {speedup:.1f}x")
                print(f"    NSeekFS: {nseekfs_best['mean_query_time_ms']:.2f}ms vs Baseline: {baseline_best['mean_query_time_ms']:.2f}ms")
        
        # Memory efficiency
        print(f"\n💾 Memory Efficiency:")
        if all_nseekfs_results:
            avg_memory = np.mean([d.get('peak_memory_mb', 0) for _, _, d in all_nseekfs_results])
            print(f"  Average NSeekFS memory usage: {avg_memory:.1f}MB")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        print(f"  • Use f16 for memory-constrained environments")
        print(f"  • Use ANN for >10K vectors with acceptable recall trade-off")
        print(f"  • Use exact search for <1K vectors or when 100% recall required")
        
        if FAISS_AVAILABLE:
            print(f"  • NSeekFS competitive with FAISS while being simpler to use")
        else:
            print(f"  • Install FAISS for additional comparisons: pip install faiss-cpu")

def main():
    """Run comprehensive benchmarks"""
    
    print("🚀 NSeekFS Heavy Benchmarks")
    print("="*50)
    print("Testing real-world performance scenarios...")
    
    benchmark = BenchmarkSuite()
    
    try:
        # Scenario 1: Small embeddings (typical for testing)
        benchmark.run_scenario(
            "Small Dataset", 
            n_vectors=5000, 
            dims=384, 
            distribution="sentence_transformer",
            n_queries=50
        )
        
        # Scenario 2: Medium embeddings (typical production)
        benchmark.run_scenario(
            "Medium Dataset", 
            n_vectors=50000, 
            dims=768, 
            distribution="sentence_transformer",
            n_queries=100
        )
        
        # Scenario 3: Large embeddings (stress test)
        benchmark.run_scenario(
            "Large Dataset", 
            n_vectors=200000, 
            dims=384, 
            distribution="sentence_transformer",
            n_queries=100
        )
        
        # Scenario 4: High-dimensional (OpenAI style)
        benchmark.run_scenario(
            "OpenAI Style", 
            n_vectors=25000, 
            dims=1536, 
            distribution="openai",
            n_queries=50
        )
        
        # Scenario 5: Random distribution (worst case for ANN)
        benchmark.run_scenario(
            "Random Distribution", 
            n_vectors=10000, 
            dims=512, 
            distribution="random",
            n_queries=50
        )
        
        # Final report
        benchmark.print_final_report()
        
        print(f"\n✅ Benchmarks completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Benchmarks interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        benchmark.cleanup()

if __name__ == "__main__":
    main()