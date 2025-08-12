import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import nseekfs

class TestNSeekBasic:
    """Testes básicos essenciais para release no PyPI"""
    
    def test_api_imports(self):
        """Testar se todas as APIs de import funcionam"""
        # Test 1: Convenience function
        embeddings = np.random.randn(20, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Via nseekfs.from_embeddings()
            index1 = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="api_test1"
            )
            assert index1.dims == 32
            assert index1.rows == 20
            
            # Via NSeek class
            index2 = nseekfs.NSeek.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="api_test2"
            )
            assert index2.dims == 32
            assert index2.rows == 20
            
            # Via direct import
            from nseekfs import from_embeddings
            index3 = from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                base_name="api_test3"
            )
            assert index3.dims == 32
            assert index3.rows == 20
    
    def test_create_and_load_index(self):
        """Teste básico de criação e carregamento de índice"""
        # Dados de teste pequenos
        embeddings = np.random.randn(100, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar índice
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                level="f32",
                ann=True,
                base_name="test",
                output_dir=temp_dir
            )
            
            # Verificar propriedades
            assert index.dims == 64
            assert index.rows == 100
            assert len(index) == 100  # Test __len__
            
            # Testar query
            query_vec = embeddings[0]
            results = index.query(query_vec, top_k=5)
            
            assert len(results) == 5
            assert results[0]["idx"] == 0  # Deve retornar o próprio vetor como mais similar
            assert 0.99 <= results[0]["score"] <= 1.01  # Cosine similarity ~1.0
            
            # Verificar que scores estão em ordem decrescente
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_different_levels(self):
        """Testar diferentes níveis de precisão"""
        embeddings = np.random.randn(50, 32).astype(np.float32)
        
        for level in ["f32", "f16", "f8"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                index = nseekfs.from_embeddings(
                    embeddings=embeddings,
                    level=level,
                    ann=False,  # Sem ANN para teste determinístico
                    output_dir=temp_dir,
                    base_name=f"test_{level}"
                )
                
                results = index.query(embeddings[0], top_k=3)
                assert len(results) == 3
                assert results[0]["idx"] == 0
                assert results[0]["score"] > 0.8  # High similarity
    
    def test_search_methods(self):
        """Testar diferentes métodos de busca"""
        embeddings = np.random.randn(100, 128).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(
                embeddings=embeddings,
                output_dir=temp_dir,
                ann=False  # Exact search for consistency
            )
            
            query = embeddings[0]
            
            # Test all methods
            for method in ["scalar", "simd", "auto"]:
                results = index.query(query, top_k=5, method=method)
                assert len(results) == 5
                assert results[0]["idx"] == 0
                assert results[0]["score"] > 0.99
    
    def test_input_validation(self):
        """Testar validação de inputs"""
        # Embeddings vazios
        with pytest.raises((ValueError, RuntimeError)):
            nseekfs.from_embeddings(np.array([]))
        
        # Dimensões muito pequenas
        with pytest.raises((ValueError, RuntimeError)):
            nseekfs.from_embeddings(np.random.randn(10, 2))  # < 8 dims
        
        # Level inválido
        with pytest.raises((ValueError, RuntimeError)):
            nseekfs.from_embeddings(np.random.randn(10, 64), level="invalid")
        
        # Embeddings com NaN
        bad_embeddings = np.random.randn(10, 32).astype(np.float32)
        bad_embeddings[0, 0] = np.nan
        with pytest.raises((ValueError, RuntimeError)):
            nseekfs.from_embeddings(bad_embeddings)
    
    def test_query_validation(self):
        """Testar validação de queries"""
        embeddings = np.random.randn(50, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(embeddings, output_dir=temp_dir)
            
            # Query com dimensão errada
            with pytest.raises((ValueError, RuntimeError)):
                index.query(np.random.randn(32), top_k=5)  # 32 != 64
            
            # Top-k inválido
            with pytest.raises((ValueError, RuntimeError)):
                index.query(embeddings[0], top_k=0)
            
            # Top-k maior que dataset
            with pytest.raises((ValueError, RuntimeError)):
                index.query(embeddings[0], top_k=1000)  # > 50
            
            # Query com NaN
            bad_query = embeddings[0].copy()
            bad_query[0] = np.nan
            with pytest.raises((ValueError, RuntimeError)):
                index.query(bad_query, top_k=5)
    
    def test_normalization_modes(self):
        """Testar diferentes modos de normalização"""
        embeddings = np.random.randn(20, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Testar normalized=True (dados já normalizados)
            index1 = nseekfs.from_embeddings(
                embeddings, normalized=True, output_dir=temp_dir, base_name="norm_true"
            )
            
            # Testar normalized=False (normalizar no Rust)
            index2 = nseekfs.from_embeddings(
                embeddings, normalized=False, output_dir=temp_dir, base_name="norm_false"
            )
            
            # Ambos devem funcionar
            results1 = index1.query(embeddings[0], top_k=3)
            results2 = index2.query(embeddings[0], top_k=3)
            
            assert len(results1) == 3
            assert len(results2) == 3
            assert results1[0]["idx"] == 0
            assert results2[0]["idx"] == 0
    
    def test_file_persistence(self):
        """Testar persistência de arquivos"""
        embeddings = np.random.randn(30, 32).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar índice
            bin_path = nseekfs.create_index(
                embeddings, output_dir=temp_dir, base_name="persist_test"
            )
            
            # Verificar se arquivo foi criado
            assert os.path.exists(bin_path)
            
            # Carregar índice existente
            index = nseekfs.load_index(bin_path)
            
            # Verificar se funciona
            results = index.query(embeddings[0], top_k=3)
            assert len(results) == 3
            assert results[0]["idx"] == 0
    
    def test_get_vector(self):
        """Testar recuperação de vetores"""
        embeddings = np.random.randn(20, 16).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index = nseekfs.from_embeddings(embeddings, output_dir=temp_dir)
            
            # Testar get_vector válido
            vec = index.get_vector(0)
            assert len(vec) == 16
            assert isinstance(vec, np.ndarray)
            
            # Verificar similaridade com original
            similarity = np.dot(vec, embeddings[0]) / (np.linalg.norm(vec) * np.linalg.norm(embeddings[0]))
            assert similarity > 0.99
            
            # Testar índice inválido
            with pytest.raises((ValueError, RuntimeError)):
                index.get_vector(1000)  # Fora de bounds
            
            with pytest.raises((ValueError, RuntimeError)):
                index.get_vector(-1)  # Índice negativo
    
    def test_ann_vs_exact(self):
        """Testar ANN vs busca exata"""
        embeddings = np.random.randn(200, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Índice com ANN
            index_ann = nseekfs.from_embeddings(
                embeddings, ann=True, output_dir=temp_dir, base_name="ann_test"
            )
            
            # Índice exato
            index_exact = nseekfs.from_embeddings(
                embeddings, ann=False, output_dir=temp_dir, base_name="exact_test"
            )
            
            query = embeddings[0]
            
            # Ambos devem retornar o próprio vetor como top result
            results_ann = index_ann.query(query, top_k=5)
            results_exact = index_exact.query(query, top_k=5)
            
            assert results_ann[0]["idx"] == 0
            assert results_exact[0]["idx"] == 0
            assert len(results_ann) == 5
            assert len(results_exact) == 5

# Teste de integração completo
def test_integration_workflow():
    """Teste de workflow completo"""
    # Simular embeddings de frases
    n_docs = 100
    dims = 384  # Tamanho típico de sentence transformers
    embeddings = np.random.randn(n_docs, dims).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Criar índice
        index = nseekfs.from_embeddings(
            embeddings=embeddings,
            level="f32",
            ann=True,
            normalized=True,
            base_name="integration_test",
            output_dir=temp_dir
        )
        
        # 2. Verificar propriedades
        assert index.dims == dims
        assert index.rows == n_docs
        
        # 3. Fazer várias queries
        for i in range(5):
            query = embeddings[i]
            results = index.query(query, top_k=10)
            
            # Verificações básicas
            assert len(results) == 10
            assert results[0]["idx"] == i  # Mais similar deve ser ele mesmo
            assert results[0]["score"] > 0.9  # Alta similaridade
            
            # Scores devem estar em ordem decrescente
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
        
        # 4. Testar métodos diferentes
        query_vec = embeddings[0]
        results_simd = index.query(query_vec, top_k=5, method="simd")
        results_scalar = index.query(query_vec, top_k=5, method="scalar")
        results_auto = index.query(query_vec, top_k=5, method="auto")
        
        # Todos devem retornar resultados válidos
        for results in [results_simd, results_scalar, results_auto]:
            assert len(results) == 5
            assert results[0]["idx"] == 0
            assert results[0]["score"] > 0.9

def test_edge_cases():
    """Testar casos extremos"""
    
    # Dataset muito pequeno
    tiny_embeddings = np.random.randn(5, 8).astype(np.float32)  # Mínimo: 8 dims
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index = nseekfs.from_embeddings(tiny_embeddings, output_dir=temp_dir)
        
        # Query deve funcionar mesmo com dataset pequeno
        results = index.query(tiny_embeddings[0], top_k=3)
        assert len(results) == 3
        
        # Top-k maior que dataset
        results_all = index.query(tiny_embeddings[0], top_k=10)
        assert len(results_all) == 5  # Deve retornar apenas os disponíveis

if __name__ == "__main__":
    pytest.main([__file__, "-v"])