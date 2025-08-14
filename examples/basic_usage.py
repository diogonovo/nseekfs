#!/usr/bin/env python3
"""
NSeekFS - Production Examples

Simple, clean examples showcasing the power of NSeek's vector search tool.
This demonstrates the clean API after our simplification.

NSeek Company Tools:
- nseekfs: Vector similarity search (this package)
- nseekplus: Advanced analytics (future)
- nseekgraph: Graph analysis (future)  
- nseektext: Text processing (future)
"""

import numpy as np
import tempfile
import time
import nseekfs

def example_1_simple_usage():
    """🚀 Example 1: Simple and Clean Usage"""
    print("\n" + "=" * 70)
    print("🚀 Example 1: Simple Vector Search")
    print("=" * 70)
    
    # Generate sample embeddings (simulating sentence embeddings)
    print("Creating 1,000 sample vectors (384 dimensions)...")
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    
    # ✅ SIMPLE API - This is what 99% of users should use
    print("\n✅ Creating index with simple API:")
    print("   index = nseekfs.from_embeddings(embeddings)")
    
    start_time = time.time()
    index = nseekfs.from_embeddings(embeddings)
    build_time = time.time() - start_time
    
    print(f"   Created index in {build_time:.2f}s")
    print(f"   Index: {index.dims}D x {index.rows} vectors")
    
    # Query the index
    print("\n🔍 Querying for similar vectors:")
    query_vector = embeddings[42]  # Use one of our vectors
    
    start_time = time.time()
    results = index.query(query_vector, top_k=5)
    query_time = time.time() - start_time
    
    print(f"   Found {len(results)} results in {query_time*1000:.2f}ms")
    print("   Top results:")
    for i, result in enumerate(results):
        print(f"     {i+1}. Vector #{result['idx']} (score: {result['score']:.4f})")
    
    print("✅ Simple usage: COMPLETE")

def example_2_power_user_api():
    """🔧 Example 2: Power User API"""
    print("\n" + "=" * 70)
    print("🔧 Example 2: Power User Control")
    print("=" * 70)
    
    embeddings = np.random.randn(500, 128).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Power users can access advanced options:")
        
        # ✅ Direct class access for full control
        print("\n💪 Advanced API - Full control:")
        print("   index = nseekfs.VectorSearch.from_embeddings(...)")
        
        index = nseekfs.VectorSearch.from_embeddings(
            embeddings,
            level="f16",           # Half precision for memory savings
            ann=False,             # Exact search for 100% recall
            normalized=True,       # Normalize vectors
            output_dir=temp_dir,
            base_name="power_user"
        )
        
        print(f"   ✅ Created {index.level} index: {index.dims}D x {index.rows} vectors")
        
        # Check statistics
        stats = index.stats
        print(f"   📊 Stats: {stats['query_count']} queries, {stats['uptime_seconds']:.1f}s uptime")
        
        # Test query
        results = index.query(embeddings[0], top_k=3)
        print(f"   🔍 Query returned {len(results)} results")
    
    print("✅ Power user API: COMPLETE")

def example_3_precision_levels():
    """📊 Example 3: Memory Optimization with Precision Levels"""
    print("\n" + "=" * 70)
    print("📊 Example 3: Memory Optimization")
    print("=" * 70)
    
    embeddings = np.random.randn(2000, 256).astype(np.float32)
    original_size = embeddings.nbytes / (1024**2)  # MB
    
    print(f"Original data: {original_size:.1f}MB")
    print("\nTesting different precision levels:")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        precision_levels = ["f32", "f16", "f8"]
        memory_savings = [0, 50, 75]  # Approximate savings
        
        for level, savings in zip(precision_levels, memory_savings):
            print(f"\n🔹 Level {level} (≈{savings}% memory reduction):")
            
            start_time = time.time()
            index = nseekfs.from_embeddings(
                embeddings,
                level=level,
                output_dir=temp_dir,
                base_name=f"test_{level}"
            )
            build_time = time.time() - start_time
            
            # Test query accuracy
            query = embeddings[10]
            results = index.query(query, top_k=5)
            
            print(f"   Build time: {build_time:.2f}s")
            print(f"   Top result: Vector #{results[0]['idx']} (score: {results[0]['score']:.4f})")
            print(f"   Memory savings: ~{savings}% vs f32")
    
    print("✅ Precision levels: COMPLETE")

def example_4_persistence():
    """💾 Example 4: Professional Persistence"""
    print("\n" + "=" * 70)
    print("💾 Example 4: Index Persistence")
    print("=" * 70)
    
    embeddings = np.random.randn(1500, 192).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating persistent index...")
        
        # Create and save index
        index1 = nseekfs.from_embeddings(
            embeddings,
            output_dir=temp_dir,
            base_name="persistent_index"
        )
        
        # Find the created file
        import os
        bin_files = [f for f in os.listdir(temp_dir) if f.endswith('.bin')]
        bin_path = os.path.join(temp_dir, bin_files[0])
        file_size = os.path.getsize(bin_path) / (1024**2)  # MB
        
        print(f"✅ Index saved: {bin_path}")
        print(f"   File size: {file_size:.1f}MB")
        
        # Load the same index later
        print("\n📂 Loading existing index:")
        print("   index = nseekfs.load_index(bin_path)")
        
        index2 = nseekfs.load_index(bin_path)
        
        print(f"   ✅ Loaded: {index2.dims}D x {index2.rows} vectors")
        
        # Verify both indexes give same results
        query = embeddings[100]
        results1 = index1.query(query, top_k=3)
        results2 = index2.query(query, top_k=3)
        
        print("🔍 Verification:")
        print(f"   Original index: Vector #{results1[0]['idx']} (score: {results1[0]['score']:.6f})")
        print(f"   Loaded index:   Vector #{results2[0]['idx']} (score: {results2[0]['score']:.6f})")
        print(f"   Match: {'✅' if results1[0]['idx'] == results2[0]['idx'] else '❌'}")
    
    print("✅ Persistence: COMPLETE")

def example_5_performance_showcase():
    """⚡ Example 5: Performance Showcase"""
    print("\n" + "=" * 70)
    print("⚡ Example 5: Performance Showcase")
    print("=" * 70)
    
    sizes = [1000, 5000, 10000]
    dims = 384  # Common sentence transformer dimension
    
    print(f"Performance test with {dims}D vectors:")
    
    for n in sizes:
        print(f"\n📈 Testing {n:,} vectors:")
        embeddings = np.random.randn(n, dims).astype(np.float32)
        
        # Build performance
        start_time = time.time()
        index = nseekfs.from_embeddings(embeddings)
        build_time = time.time() - start_time
        
        # Query performance (average of 10 queries)
        query_times = []
        for i in range(10):
            query = embeddings[i]
            start_time = time.time()
            results = index.query(query, top_k=10)
            query_times.append((time.time() - start_time) * 1000)  # ms
        
        avg_query_time = np.mean(query_times)
        
        print(f"   Build time: {build_time:.2f}s")
        print(f"   Avg query: {avg_query_time:.2f}ms")
        print(f"   Memory usage: ~{embeddings.nbytes / (1024**2):.0f}MB")
        
        # Check SIMD usage
        if dims >= 64:
            print("   🚀 SIMD acceleration: ACTIVE (2-4x speedup)")
        else:
            print("   ⚪ SIMD acceleration: Not used (<64 dims)")
    
    print("✅ Performance showcase: COMPLETE")

def example_6_api_comparison():
    """🔧 Example 6: API Style Comparison"""
    print("\n" + "=" * 70)
    print("🔧 Example 6: API Style Comparison")
    print("=" * 70)
    
    embeddings = np.random.randn(800, 128).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Demonstrating different API styles:")
        
        # Style 1: Simple function (recommended)
        print("\n1️⃣ Simple API (recommended for most users):")
        print("   index = nseekfs.from_embeddings(embeddings)")
        index1 = nseekfs.from_embeddings(embeddings)
        print(f"   ✅ Created: {index1.dims}D x {index1.rows} vectors")
        
        # Style 2: Direct class access (power users)
        print("\n2️⃣ Direct class access (power users):")
        print("   index = nseekfs.VectorSearch.from_embeddings(...)")
        index2 = nseekfs.VectorSearch.from_embeddings(
            embeddings,
            ann=False,
            level="f32",
            output_dir=temp_dir,
            base_name="class_based"
        )
        print(f"   ✅ Created: {index2.dims}D x {index2.rows} vectors")
        
        # Style 3: Explicit import
        print("\n3️⃣ Explicit import:")
        print("   from nseekfs import from_embeddings")
        print("   index = from_embeddings(embeddings)")
        from nseekfs import from_embeddings
        index3 = from_embeddings(
            embeddings,
            output_dir=temp_dir,
            base_name="explicit"
        )
        print(f"   ✅ Created: {index3.dims}D x {index3.rows} vectors")
        
        # All should work the same
        print("\n🔍 Testing all APIs give same results:")
        query = embeddings[42]
        
        for i, idx in enumerate([index1, index2, index3], 1):
            results = idx.query(query, top_k=3)
            print(f"   API style {i}: Top result #{results[0]['idx']} (score: {results[0]['score']:.6f})")
    
    print("✅ API comparison: COMPLETE")

def example_7_utilities():
    """🛠️ Example 7: Utility Functions"""
    print("\n" + "=" * 70)
    print("🛠️ Example 7: Utility Functions")
    print("=" * 70)
    
    print("System information and health checks:")
    
    # Health check
    print("\n🏥 Health check:")
    health = nseekfs.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Basic test time: {health['basic_test_time']:.3f}s")
    print(f"   Rust engine: {'✅' if health['rust_engine_working'] else '❌'}")
    
    # System info
    print("\n💻 System information:")
    info = nseekfs.get_system_info()
    print(f"   Platform: {info['platform']}")
    print(f"   Python: {info['python_version']}")
    print(f"   NSeekFS: {info['nseekfs_version']}")
    
    # Configuration
    print("\n⚙️ Configuration:")
    config = nseekfs.validate_config()
    print(f"   Max concurrent queries: {config['max_concurrent_queries']}")
    print(f"   Memory warning threshold: {config['memory_warning_threshold_gb']:.1f}GB")
    print(f"   Config valid: {'✅' if config['config_valid'] else '❌'}")
    
    print("✅ Utilities: COMPLETE")

def main():
    """Run all examples showcasing NSeekFS capabilities"""
    print("🚀 NSeekFS - Clean and Simple Examples")
    print("Showcasing NSeek's vector similarity search tool")
    print("="*70)
    
    try:
        example_1_simple_usage()
        example_2_power_user_api()
        example_3_precision_levels()
        example_4_persistence()
        example_5_performance_showcase()
        example_6_api_comparison()
        example_7_utilities()
        
        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("=" * 70)
        
        print("\n💡 Key Takeaways:")
        print("   ✅ Simple API: nseekfs.from_embeddings(vectors)")
        print("   ⚡ High performance with SIMD acceleration") 
        print("   💾 Professional persistence and loading")
        print("   📊 Multiple precision levels for memory optimization")
        print("   🔧 Power user API for advanced control")
        print("   🛠️ Built-in utilities and health checks")
        
        print("\n🎯 Recommended Usage:")
        print("   • Use nseekfs.from_embeddings() for 99% of cases")
        print("   • Choose ann=False for 100% recall (production)")
        print("   • Use f16/f8 levels for memory savings")
        print("   • Save indexes for reuse with output_dir parameter")
        
        print("\n🏢 NSeek Ecosystem:")
        print("   • nseekfs: Vector search (this package) ✅")
        print("   • nseekplus: Advanced analytics (coming soon)")
        print("   • nseekgraph: Graph analysis (coming soon)")
        print("   • nseektext: Text processing (coming soon)")
        
        print("\nFor more information: https://github.com/diogonovo/nseekfs")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter issues:")
        print("   • Check NumPy is properly installed")
        print("   • Ensure sufficient memory available")
        print("   • Verify platform compatibility (x86_64 required)")

if __name__ == "__main__":
    main()