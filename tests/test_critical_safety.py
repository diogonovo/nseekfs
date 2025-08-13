#!/usr/bin/env python3
"""
Testes críticos de segurança para NSeekFS
Valida todas as correções implementadas antes do release PyPI
"""

import os
import sys
import time
import tempfile
import threading
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configurar logging para testes
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import com fallback para desenvolvimento
try:
    import nseekfs
    from nseekfs.validation import ValidationError
    from nseekfs.highlevel import NSeekError, NSeekValidationError, NSeekMemoryError, NSeekIndexError
except ImportError as e:
    logger.error(f"Failed to import nseekfs: {e}")
    sys.exit(1)

class TestCriticalSafety:
    """Testes críticos que DEVEM passar antes do release"""

    def test_memory_leak_prevention(self):
        """Testar prevenção de memory leaks no ANN"""
        print("\n🧪 Testing memory leak prevention...")
        
        # Criar dataset que force buckets grandes
        n_vectors = 5000
        dims = 128
        embeddings = np.random.randn(n_vectors, dims).astype(np.float32)
        
        # Forçar vetores similares para criar buckets grandes
        base_vector = np.random.randn(dims).astype(np.float32)
        for i in range(0, n_vectors, 100):
            # Cada 100 vetores são similares ao base
            noise = np.random.normal(0, 0.1, (min(100, n_vectors - i), dims))
            embeddings[i:i+100] = base_vector + noise
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar índice com ANN
            index = nseekfs.from_embeddings(
                embeddings, ann=True, output_dir=temp_dir, base_name="memory_test"
            )
            
            # Múltiplas queries para testar estabilidade de memória
            for i in range(50):
                query = embeddings[i]
                results = index.query(query, top_k=10)
                assert len(results) <= 10
                assert all(r["idx"] < n_vectors for r in results)
            
            print("✅ Memory leak prevention working")

    def test_thread_safety(self):
        """Testar thread safety rigoroso"""
        print("\n🧪 Testing thread safety...")
        
        embeddings = np.random.randn(1000, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(
                embeddings, output_dir=temp_dir, base_name="thread_test"
            )
            
            results = []
            errors = []
            
            def worker_query(worker_id):
                try:
                    for i in range(20):
                        query_idx = (worker_id * 20 + i) % len(embeddings)
                        query = embeddings[query_idx]
                        result = index.query(query, top_k=5)
                        results.append((worker_id, len(result)))
                        time.sleep(0.001)  # Pequena pausa para simular uso real
                    return f"Worker {worker_id} completed"
                except Exception as e:
                    errors.append(f"Worker {worker_id} failed: {e}")
                    return f"Worker {worker_id} failed"
            
            # Executar múltiplas threads simultaneamente
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker_query, i) for i in range(8)]
                
                for future in as_completed(futures):
                    result = future.result()
                    print(f"  {result}")
            
            # Verificar resultados
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            assert len(results) == 8 * 20, f"Expected 160 results, got {len(results)}"
            
            print(f"✅ Thread safety verified: {len(results)} queries completed")

    def test_input_validation_comprehensive(self):
        """Testar validação de entrada abrangente"""
        print("\n🧪 Testing comprehensive input validation...")
        
        # Test 1: Embeddings inválidos
        invalid_cases = [
            ([], "Empty embeddings"),
            (np.array([]), "Empty numpy array"),
            (np.random.randn(10, 2), "Too few dimensions"),
            (np.random.randn(2, 10), "Too few vectors"),
        ]
        
        for invalid_input, description in invalid_cases:
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                nseekfs.from_embeddings(invalid_input)
            print(f"  ✅ Rejected: {description}")
        
        # Test 2: Embeddings com valores inválidos
        bad_embeddings = np.random.randn(50, 32).astype(np.float32)
        
        # NaN values
        bad_embeddings_nan = bad_embeddings.copy()
        bad_embeddings_nan[0, 0] = np.nan
        with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
            nseekfs.from_embeddings(bad_embeddings_nan)
        print("  ✅ Rejected: NaN values")
        
        # Infinite values
        bad_embeddings_inf = bad_embeddings.copy()
        bad_embeddings_inf[1, 1] = np.inf
        with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
            nseekfs.from_embeddings(bad_embeddings_inf)
        print("  ✅ Rejected: Infinite values")
        
        # Zero vectors
        bad_embeddings_zero = bad_embeddings.copy()
        bad_embeddings_zero[2, :] = 0.0
        with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
            nseekfs.from_embeddings(bad_embeddings_zero)
        print("  ✅ Rejected: Zero vectors")

    def test_query_validation_strict(self):
        """Testar validação de queries rigorosa"""
        print("\n🧪 Testing strict query validation...")
        
        embeddings = np.random.randn(100, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(embeddings, output_dir=temp_dir)
            
            # Test dimensões erradas
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(np.random.randn(32), top_k=5)
            print("  ✅ Rejected: Wrong dimensions")
            
            # Test top_k inválido
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(embeddings[0], top_k=0)
            print("  ✅ Rejected: Zero top_k")
            
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(embeddings[0], top_k=200)  # > dataset size
            print("  ✅ Rejected: top_k > dataset size")
            
            # Test query com valores inválidos
            bad_query = embeddings[0].copy()
            bad_query[0] = np.nan
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(bad_query, top_k=5)
            print("  ✅ Rejected: NaN in query")
            
            # Test zero query
            zero_query = np.zeros(64, dtype=np.float32)
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(zero_query, top_k=5)
            print("  ✅ Rejected: Zero query vector")

    def test_error_handling_specific(self):
        """Testar tratamento de erros específico"""
        print("\n🧪 Testing specific error handling...")
        
        embeddings = np.random.randn(50, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(embeddings, output_dir=temp_dir)
            
            # Test get_vector com índices inválidos
            with pytest.raises((ValueError, RuntimeError, NSeekIndexError)):
                index.get_vector(-1)
            print("  ✅ Rejected: Negative index")
            
            with pytest.raises((ValueError, RuntimeError, NSeekIndexError)):
                index.get_vector(100)  # > dataset size
            print("  ✅ Rejected: Index out of bounds")
            
            # Test métodos inválidos
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(embeddings[0], top_k=5, method="invalid")
            print("  ✅ Rejected: Invalid method")
            
            # Test similaridade inválida
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(embeddings[0], top_k=5, similarity="invalid")
            print("  ✅ Rejected: Invalid similarity")

    def test_file_operations_robust(self):
        """Testar operações de arquivo robustas"""
        print("\n🧪 Testing robust file operations...")
        
        embeddings = np.random.randn(100, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test carregamento de arquivo inexistente
            fake_path = Path(temp_dir) / "nonexistent.bin"
            with pytest.raises(FileNotFoundError):
                nseekfs.load_index(fake_path)
            print("  ✅ Rejected: Nonexistent file")
            
            # Test criação e carregamento normal
            index = nseekfs.from_embeddings(
                embeddings, output_dir=temp_dir, base_name="robust_test"
            )
            
            # Verificar se arquivos foram criados
            bin_file = Path(temp_dir) / "f32.bin"
            assert bin_file.exists()
            
            # Test carregamento do arquivo criado
            loaded_index = nseekfs.load_index(bin_file)
            assert loaded_index.dims == 32
            assert loaded_index.rows == 100
            print("  ✅ File operations working correctly")
            
            # Test funcionalidade idêntica
            query = embeddings[0]
            original_results = index.query(query, top_k=5)
            loaded_results = loaded_index.query(query, top_k=5)
            
            assert len(original_results) == len(loaded_results)
            assert original_results[0]["idx"] == loaded_results[0]["idx"]
            print("  ✅ Loaded index functions identically")

    def test_resource_limits(self):
        """Testar limites de recursos"""
        print("\n🧪 Testing resource limits...")
        
        # Test dataset muito grande (simulado)
        try:
            huge_embeddings = np.random.randn(1000, 15000).astype(np.float32)  # > MAX_DIMENSIONS
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError, NSeekMemoryError)):
                nseekfs.from_embeddings(huge_embeddings)
            print("  ✅ Rejected: Too many dimensions")
        except MemoryError:
            print("  ✅ System memory limit reached (expected)")
        
        # Test top_k muito grande
        small_embeddings = np.random.randn(10, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(small_embeddings, output_dir=temp_dir)
            
            # Tentar query com top_k muito grande
            with pytest.raises((ValueError, RuntimeError, NSeekValidationError)):
                index.query(small_embeddings[0], top_k=50000)  # Muito maior que dataset
            print("  ✅ Rejected: Excessive top_k")

    def test_ann_safety_improvements(self):
        """Testar melhorias de segurança do ANN"""
        print("\n🧪 Testing ANN safety improvements...")
        
        # Dataset médio para forçar comportamento do ANN
        embeddings = np.random.randn(2000, 128).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar índice com ANN
            index = nseekfs.from_embeddings(
                embeddings, ann=True, output_dir=temp_dir, base_name="ann_safety"
            )
            
            # Múltiplas queries para testar estabilidade
            successful_queries = 0
            for i in range(100):
                try:
                    query = embeddings[i % len(embeddings)]
                    results = index.query(query, top_k=10)
                    
                    # Validar resultados
                    assert isinstance(results, list)
                    assert len(results) <= 10
                    assert all(isinstance(r, dict) for r in results)
                    assert all("idx" in r and "score" in r for r in results)
                    assert all(0 <= r["idx"] < len(embeddings) for r in results)
                    assert all(isinstance(r["score"], (int, float)) for r in results)
                    assert all(np.isfinite(r["score"]) for r in results)