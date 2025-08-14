import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import nseekfs

class TestNSeekFSBasic:
    """Basic tests essential for PyPI release"""
    
    def test_simple_api_imports(self):
        """Test that the simple API works correctly"""
        # Test data
        embeddings = np.random.randn(20, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # ✅ Simple API (recommended)
            index1 = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="simple_test"
            )
            assert index1.dims == 32
            assert index1.rows == 20
            
            # ✅ Direct import also works
            from nseekfs import from_embeddings
            index2 = from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="direct_test"
            )
            assert index2.dims == 32
            assert index2.rows == 20
    
    def test_advanced_api_access(self):
        """Test that advanced API access works for power users"""
        embeddings = np.random.randn(20, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # ✅ Advanced API (power users)
            index = nseekfs.VectorSearch.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="advanced_test",
                ann=False,
                level="f32"
            )
            assert index.dims == 32
            assert index.rows == 20
    
    def test_create_and_load_index(self):
        """Test basic index creation and loading"""
        # Small test data
        embeddings = np.random.randn(100, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create index
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                level="f32",
                ann=True,
                base_name="test",
                output_dir=temp_dir
            )
            
            # Verify properties
            assert index.dims == 64
            assert index.rows == 100
            
            # Test query
            query_vector = embeddings[0]
            results = index.query(query_vector, top_k=5)
            
            assert len(results) == 5
            assert all('idx' in result for result in results)
            assert all('score' in result for result in results)
            
            # First result should be the query vector itself
            assert results[0]['idx'] == 0
            assert results[0]['score'] > 0.99  # Should be very similar
    
    def test_load_existing_index(self):
        """Test loading an existing index"""
        embeddings = np.random.randn(50, 128).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create index
            index1 = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="persistent"
            )
            
            # Find the created .bin file
            bin_files = list(Path(temp_dir).glob("*.bin"))
            assert len(bin_files) == 1
            bin_path = bin_files[0]
            
            # Load the same index
            index2 = nseekfs.load_index(bin_path)
            
            # Should have same properties
            assert index2.dims == index1.dims
            assert index2.rows == index1.rows
            
            # Should give same results
            query = embeddings[10]
            results1 = index1.query(query, top_k=3)
            results2 = index2.query(query, top_k=3)
            
            assert len(results1) == len(results2)
            assert results1[0]['idx'] == results2[0]['idx']
    
    def test_different_precision_levels(self):
        """Test different precision levels"""
        embeddings = np.random.randn(30, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            levels = ["f32", "f16", "f8"]
            
            for level in levels:
                index = nseekfs.from_embeddings(
                    embeddings=embeddings,
                    level=level,
                    output_dir=temp_dir,
                    base_name=f"test_{level}"
                )
                
                assert index.dims == 64
                assert index.rows == 30
                assert index.level == level
                
                # Test query works
                results = index.query(embeddings[0], top_k=3)
                assert len(results) == 3
                assert results[0]['idx'] == 0
    
    def test_utilities(self):
        """Test utility functions"""
        # Health check
        health = nseekfs.health_check()
        assert health['status'] == 'healthy'
        assert 'basic_test_time' in health
        
        # System info
        info = nseekfs.get_system_info()
        assert 'platform' in info
        assert 'nseekfs_version' in info
        
        # Config validation
        config = nseekfs.validate_config()
        assert config['config_valid'] == True
        assert 'max_concurrent_queries' in config
    
    def test_error_handling(self):
        """Test proper error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid precision level
            embeddings = np.random.randn(10, 32).astype(np.float32)
            with pytest.raises(Exception):  # Should raise ValidationError
                nseekfs.from_embeddings(
                    embeddings=embeddings,
                    level="invalid_level",
                    output_dir=temp_dir
                )
    
    def test_concurrent_queries(self):
        """Test that concurrent queries work safely"""
        import threading
        import time
        
        embeddings = np.random.randn(200, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="concurrent_test"
            )
            
            results = []
            errors = []
            
            def worker(worker_id):
                try:
                    query = embeddings[worker_id % len(embeddings)]
                    result = index.query(query, top_k=5)
                    results.append((worker_id, len(result)))
                except Exception as e:
                    errors.append((worker_id, str(e)))
            
            # Launch multiple threads
            threads = []
            for i in range(8):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for all to complete
            for t in threads:
                t.join()
            
            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 8
            assert all(count == 5 for _, count in results)
    
    def test_context_manager(self):
        """Test context manager functionality"""
        embeddings = np.random.randn(50, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with nseekfs.from_embeddings(embeddings, output_dir=temp_dir) as index:
                assert index.dims == 32
                assert index.rows == 50
                
                results = index.query(embeddings[0], top_k=3)
                assert len(results) == 3
    
    def test_stats_and_properties(self):
        """Test index statistics and properties"""
        embeddings = np.random.randn(75, 96).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="stats_test"
            )
            
            # Test properties
            assert index.dims == 96
            assert index.rows == 75
            assert len(index) == 75  # __len__ method
            
            # Test stats
            stats = index.stats
            assert stats['dims'] == 96
            assert stats['rows'] == 75
            assert stats['query_count'] == 0
            assert 'uptime_seconds' in stats
            
            # Run a query and check stats update
            index.query(embeddings[0], top_k=3)
            updated_stats = index.stats
            assert updated_stats['query_count'] == 1
    
    def test_string_representation(self):
        """Test string representation of index"""
        embeddings = np.random.randn(25, 48).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                level="f16",
                output_dir=temp_dir
            )
            
            repr_str = repr(index)
            assert "VectorSearch" in repr_str
            assert "level='f16'" in repr_str
            assert "dims=48" in repr_str
            assert "rows=25" in repr_str

class TestNSeekFSCompatibility:
    """Test compatibility and edge cases"""
    
    def test_different_input_formats(self):
        """Test different input formats for embeddings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # NumPy array (recommended)
            np_embeddings = np.random.randn(20, 32).astype(np.float32)
            index1 = nseekfs.from_embeddings(np_embeddings, output_dir=temp_dir, base_name="numpy")
            assert index1.dims == 32
            
            # Python list of lists
            list_embeddings = np_embeddings.tolist()
            index2 = nseekfs.from_embeddings(list_embeddings, output_dir=temp_dir, base_name="list")
            assert index2.dims == 32
    
    def test_normalization_behavior(self):
        """Test vector normalization behavior"""
        # Create unnormalized vectors
        embeddings = np.random.randn(30, 64).astype(np.float32) * 10  # Large magnitude
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with normalization (default)
            index_norm = nseekfs.from_embeddings(
                embeddings=embeddings,
                normalized=True,
                output_dir=temp_dir,
                base_name="normalized"
            )
            
            # Test without normalization
            index_unnorm = nseekfs.from_embeddings(
                embeddings=embeddings,
                normalized=False,
                output_dir=temp_dir,
                base_name="unnormalized"
            )
            
            # Both should work but give different results
            query = embeddings[0]
            results_norm = index_norm.query(query, top_k=3)
            results_unnorm = index_unnorm.query(query, top_k=3)
            
            assert len(results_norm) == 3
            assert len(results_unnorm) == 3
            # Results may differ due to normalization
    
    def test_memory_efficiency(self):
        """Test with larger datasets to check memory efficiency"""
        # Create moderately large dataset
        embeddings = np.random.randn(5000, 128).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle 5K vectors without issues
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="large_test"
            )
            
            assert index.dims == 128
            assert index.rows == 5000
            
            # Test queries work efficiently
            import time
            start_time = time.time()
            
            for i in range(10):
                results = index.query(embeddings[i], top_k=10)
                assert len(results) == 10
            
            elapsed = time.time() - start_time
            # Should complete 10 queries in reasonable time (< 1 second)
            assert elapsed < 1.0, f"Queries took too long: {elapsed:.2f}s"

if __name__ == "__main__":
    # Run basic smoke test
    print("🔧 Running basic smoke test...")
    
    # Simple functionality test
    embeddings = np.random.randn(100, 64).astype(np.float32)
    index = nseekfs.from_embeddings(embeddings)
    results = index.query(embeddings[0], top_k=5)
    
    assert len(results) == 5
    assert results[0]['idx'] == 0
    
    print("✅ Basic smoke test passed!")
    print(f"   Created index: {index.dims}D x {index.rows} vectors")
    print(f"   Query returned {len(results)} results")
    print(f"   Top result: idx={results[0]['idx']}, score={results[0]['score']:.4f}")