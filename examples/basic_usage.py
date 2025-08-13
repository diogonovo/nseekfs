#!/usr/bin/env python3
"""
NSeekFS - Professional Usage Examples
Demonstrates production-ready exact search and experimental ANN capabilities
"""

import numpy as np
import nseekfs
import time
import tempfile
from pathlib import Path

def example_1_production_exact_search():
    """🏆 Production-Ready Exact Search - Recommended for Production"""
    print("=" * 70)
    print("🏆 Example 1: Production Exact Search (Recommended)")
    print("=" * 70)
    
    # Simulate real-world embeddings (e.g., from sentence-transformers)
    print("Creating realistic embeddings (50,000 x 384)...")
    embeddings = np.random.randn(50000, 384).astype(np.float32)
    
    # Normalize for cosine similarity (common in ML)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    
    # Production-ready exact search
    print("Building production index...")
    start_time = time.time()
    
    index = nseekfs.from_embeddings(
        embeddings=embeddings,
        ann=False,              # Exact search for production
        level="f32",            # Full precision
        normalized=True,        # Pre-normalized embeddings
        method="auto"           # Auto-detect SIMD capabilities
    )
    
    build_time = time.time() - start_time
    print(f"✅ Index built in {build_time:.2f}s")
    print(f"   Dimensions: {index.dims}")
    print(f"   Vectors: {index.rows}")
    print(f"   Precision: {index.level}")
    
    # Fast similarity search with guaranteed 100% recall
    query_vector = embeddings[42]  # Use known vector
    
    start_time = time.time()
    results = index.query(query_vector, top_k=10)
    query_time = (time.time() - start_time) * 1000
    
    print(f"\n🔍 Query Results (100% recall guaranteed):")
    print(f"   Query time: {query_time:.2f}ms")
    print(f"   Results found: {len(results)}")
    print(f"   Top result: Index {results[0]['idx']}, Score {results[0]['score']:.6f}")
    
    # Verify exact search quality
    assert results[0]['idx'] == 42, "Exact search should find itself as top result"
    assert results[0]['score'] > 0.999, "Self-similarity should be ~1.0"
    
    print("✅ Production exact search: PASSED")

def example_2_memory_optimization():
    """💾 Memory-Efficient Deployment with Quantization"""
    print("\n" + "=" * 70)
    print("💾 Example 2: Memory Optimization with Quantization")
    print("=" * 70)
    
    embeddings = np.random.randn(10000, 512).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print("Comparing memory usage across precision levels...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_results = {}
        
        for level in ["f64", "f32", "f16", "f8"]:
            print(f"\n📊 Testing {level} precision...")
            
            # Create index with specific precision
            index = nseekfs.from_embeddings(
                embeddings, 
                level=level,
                ann=False,
                output_dir=temp_dir,
                base_name=f"memory_test_{level}"
            )
            
            # Check file size (memory usage indicator)
            bin_file = Path(temp_dir) / f"memory_test_{level}_{level}.bin"
            if bin_file.exists():
                size_mb = bin_file.stat().st_size / (1024 * 1024)
                memory_results[level] = size_mb
            
            # Test accuracy
            query = embeddings[0]
            results = index.query(query, top_k=5)
            accuracy = results[0]['score']
            
            print(f"   Memory usage: {memory_results[level]:.2f} MB")
            print(f"   Accuracy: {accuracy:.6f}")
            
            # Quality assessment
            if accuracy > 0.999:
                quality = "🟢 Excellent"
            elif accuracy > 0.995:
                quality = "🟡 Good"
            else:
                quality = "🟠 Reduced"
            
            print(f"   Quality: {quality}")
    
    # Summary
    print(f"\n📈 Memory Usage Summary:")
    f32_size = memory_results['f32']
    for level, size in memory_results.items():
        reduction = ((f32_size - size) / f32_size) * 100
        print(f"   {level}: {size:.2f} MB ({reduction:+.0f}% vs f32)")
    
    print("✅ Memory optimization: DEMONSTRATED")

def example_3_simd_performance():
    """⚡ SIMD Performance Validation"""
    print("\n" + "=" * 70)
    print("⚡ Example 3: SIMD Performance Comparison")
    print("=" * 70)
    
    # Test different dimension sizes for SIMD effectiveness
    test_cases = [
        (32, "Small (no SIMD benefit)"),
        (128, "Medium (SIMD benefit)"), 
        (384, "Large (optimal SIMD)"),
        (768, "Very Large (maximum SIMD)")
    ]
    
    n_vectors = 5000
    n_queries = 50
    
    for dims, description in test_cases:
        print(f"\n🧪 Testing {dims}D vectors - {description}")
        
        embeddings = np.random.randn(n_vectors, dims).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create index
        index = nseekfs.from_embeddings(embeddings, ann=False)
        
        # Test different methods
        query_vectors = embeddings[:n_queries]
        
        # Auto method (should use SIMD for 64+ dims)
        start_time = time.time()
        for query in query_vectors:
            results = index.query(query, top_k=5, method="auto")
        auto_time = (time.time() - start_time) * 1000 / n_queries
        
        # Scalar method (baseline)
        start_time = time.time()
        for query in query_vectors:
            results = index.query(query, top_k=5, method="scalar")
        scalar_time = (time.time() - start_time) * 1000 / n_queries
        
        # Calculate speedup
        speedup = scalar_time / auto_time if auto_time > 0 else 1.0
        
        print(f"   Auto method: {auto_time:.2f}ms avg")
        print(f"   Scalar method: {scalar_time:.2f}ms avg")
        print(f"   Speedup: {speedup:.1f}x")
        
        # Expected performance
        if dims >= 64:
            expected = "🚀 SIMD acceleration expected"
            assert speedup > 1.2, f"SIMD should provide speedup for {dims}D"
        else:
            expected = "📏 Scalar performance (dimensions < 64)"
        
        print(f"   Status: {expected}")
    
    print("✅ SIMD performance: VALIDATED")

def example_4_experimental_ann():
    """🧪 Experimental ANN Testing - Research & Prototyping"""
    print("\n" + "=" * 70)
    print("🧪 Example 4: Experimental ANN (Research/Prototyping Only)")
    print("=" * 70)
    
    print("⚠️  WARNING: ANN is experimental (10-15% recall@10)")
    print("   Use exact search for production workloads")
    print("   Consider Faiss for production ANN requirements\n")
    
    embeddings = np.random.randn(20000, 256).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build both exact and ANN indexes
        print("Building indexes for comparison...")
        
        start_time = time.time()
        index_exact = nseekfs.from_embeddings(
            embeddings, 
            ann=False,
            output_dir=temp_dir,
            base_name="exact"
        )
        exact_build_time = time.time() - start_time
        
        start_time = time.time()
        index_ann = nseekfs.from_embeddings(
            embeddings, 
            ann=True,
            output_dir=temp_dir,
            base_name="ann"
        )
        ann_build_time = time.time() - start_time
        
        print(f"   Exact build: {exact_build_time:.2f}s")
        print(f"   ANN build: {ann_build_time:.2f}s")
        
        # Compare search performance and quality
        test_queries = embeddings[:10]
        
        exact_times = []
        ann_times = []
        recalls = []
        
        for i, query in enumerate(test_queries):
            # Exact search
            start_time = time.time()
            exact_results = index_exact.query(query, top_k=20)
            exact_times.append((time.time() - start_time) * 1000)
            
            # ANN search
            start_time = time.time()
            ann_results = index_ann.query(query, top_k=20)
            ann_times.append((time.time() - start_time) * 1000)
            
            # Calculate recall@10
            if len(exact_results) >= 10 and len(ann_results) >= 10:
                exact_top10 = set(r['idx'] for r in exact_results[:10])
                ann_top10 = set(r['idx'] for r in ann_results[:10])
                recall = len(exact_top10.intersection(ann_top10)) / len(exact_top10)
                recalls.append(recall)
            
            # Check self-hit (should find query vector itself)
            exact_self_hit = exact_results[0]['idx'] == i if exact_results else False
            ann_self_hit = ann_results[0]['idx'] == i if ann_results else False
            
            if i < 3:  # Show details for first few
                print(f"   Query {i}: Exact self-hit: {exact_self_hit}, ANN self-hit: {ann_self_hit}")
        
        # Summary statistics
        avg_exact_time = np.mean(exact_times)
        avg_ann_time = np.mean(ann_times)
        avg_recall = np.mean(recalls) if recalls else 0
        
        print(f"\n📊 Performance Comparison:")
        print(f"   Exact search: {avg_exact_time:.2f}ms avg")
        print(f"   ANN search: {avg_ann_time:.2f}ms avg")
        print(f"   ANN recall@10: {avg_recall:.1%}")
        
        # Honest assessment
        if avg_recall > 0.5:
            assessment = "🟢 Good for prototyping"
        elif avg_recall > 0.2:
            assessment = "🟡 Experimental quality"
        else:
            assessment = "🟠 Research only"
        
        print(f"   Quality: {assessment}")
        print(f"\n💡 Recommendation: Use exact search for production")
    
    print("✅ Experimental ANN: DEMONSTRATED")

def example_5_professional_persistence():
    """💾 Professional Persistence & Loading"""
    print("\n" + "=" * 70)
    print("💾 Example 5: Professional Persistence & Binary Format")
    print("=" * 70)
    
    embeddings = np.random.randn(25000, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create index with persistence
        print("Creating persistent index...")
        
        index_path = nseekfs.from_embeddings(
            embeddings,
            ann=False,
            level="f32", 
            output_dir=temp_dir,
            base_name="production_v1"
        )
        
        # Check created files
        index_dir = Path(temp_dir)
        bin_files = list(index_dir.glob("*.bin"))
        
        print(f"   Created files: {len(bin_files)}")
        for bin_file in bin_files:
            size_mb = bin_file.stat().st_size / (1024 * 1024)
            print(f"   - {bin_file.name}: {size_mb:.2f} MB")
        
        # Test loading
        print("\nTesting index loading...")
        
        for bin_file in bin_files:
            start_time = time.time()
            loaded_index = nseekfs.load_index(str(bin_file), ann=False)
            load_time = (time.time() - start_time) * 1000
            
            print(f"   Loaded {bin_file.name} in {load_time:.1f}ms")
            
            # Verify loaded index works
            query = embeddings[100]
            results = loaded_index.query(query, top_k=5)
            
            assert len(results) == 5, "Loaded index should return results"
            assert results[0]['idx'] == 100, "Should find self as top result"
            
            print(f"   Verification: ✅ Working correctly")
        
        print("✅ Professional persistence: VALIDATED")

def example_6_api_variations():
    """🔧 API Usage Patterns"""
    print("\n" + "=" * 70)
    print("🔧 Example 6: Different API Usage Patterns")
    print("=" * 70)
    
    embeddings = np.random.randn(5000, 128).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Demonstrating different API styles...")
        
        # Style 1: Simple convenience function
        print("\n1️⃣ Convenience function (recommended for most users):")
        index1 = nseekfs.from_embeddings(embeddings)
        print(f"   ✅ Created index: {index1.dims}D, {index1.rows} vectors")
        
        # Style 2: Class-based approach
        print("\n2️⃣ Class-based approach (full control):")
        index2 = nseekfs.NSeek.from_embeddings(
            embeddings,
            ann=False,
            level="f32",
            output_dir=temp_dir,
            base_name="class_based"
        )
        print(f"   ✅ Created index: {index2.dims}D, {index2.rows} vectors")
        
        # Style 3: Explicit import
        print("\n3️⃣ Explicit import:")
        from nseekfs import from_embeddings
        index3 = from_embeddings(
            embeddings,
            output_dir=temp_dir,
            base_name="explicit"
        )
        print(f"   ✅ Created index: {index3.dims}D, {index3.rows} vectors")
        
        # Test all work the same
        query = embeddings[42]
        for i, idx in enumerate([index1, index2, index3], 1):
            results = idx.query(query, top_k=3)
            print(f"   API style {i}: {len(results)} results, top score: {results[0]['score']:.6f}")
    
    print("✅ API variations: DEMONSTRATED")

def main():
    """Run all professional examples"""
    print("🚀 NSeekFS - Professional Usage Examples")
    print("Demonstrating production-ready exact search with experimental ANN")
    print("=" * 70)
    
    try:
        example_1_production_exact_search()
        example_2_memory_optimization()
        example_3_simd_performance()
        example_4_experimental_ann()
        example_5_professional_persistence()
        example_6_api_variations()
        
        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("=" * 70)
        
        print("\n💡 Key Takeaways:")
        print("   🏆 Use exact search for production (100% recall guaranteed)")
        print("   ⚡ SIMD acceleration works automatically for 64+ dimensions")
        print("   💾 Multiple precision levels available for memory optimization")
        print("   🧪 ANN is experimental - use for research/prototyping only")
        print("   💾 Professional binary format with versioning support")
        print("   🔧 Multiple API styles for different preferences")
        
        print("\n🎯 Next Steps:")
        print("   • Use nseekfs.from_embeddings(embeddings, ann=False) for production")
        print("   • Experiment with quantization levels (f16, f8) for memory savings")  
        print("   • Consider Faiss for production ANN requirements")
        print("   • Monitor performance with your specific embedding dimensions")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter issues, please check:")
        print("   • NumPy is properly installed")
        print("   • Sufficient memory available")
        print("   • Platform compatibility (x86_64 required)")

if __name__ == "__main__":
    main()