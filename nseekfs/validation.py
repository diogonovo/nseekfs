"""Input validation utilities for nseekfs"""

import numpy as np
from typing import Union, List
from pathlib import Path

def validate_embeddings(embeddings: Union[np.ndarray, List[List[float]], str]) -> np.ndarray:
    """Validate and convert embeddings to proper format"""
    
    if isinstance(embeddings, str):
        path = Path(embeddings)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings}")
        
        if embeddings.endswith(".npy"):
            embeddings = np.load(embeddings)
        elif embeddings.endswith(".csv"):
            embeddings = np.loadtxt(embeddings, delimiter=",")
        else:
            raise ValueError("Unsupported file format. Only .npy and .csv are supported.")
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    # Validações básicas
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_samples, dimensions)")
    
    n_samples, dims = embeddings.shape
    
    if n_samples < 1:
        raise ValueError("At least one embedding is required")
    
    if dims < 8:
        raise ValueError("Embedding dimension must be at least 8")
    
    if dims > 4096:
        raise ValueError("Embedding dimension must be at most 4096")
    
    # Verificar valores inválidos
    if np.any(np.isnan(embeddings)):
        raise ValueError("NaN values detected in embeddings")
    
    if np.any(np.isinf(embeddings)):
        raise ValueError("Infinite values detected in embeddings")
    
    # Verificar se todos os vetores são zero
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        zero_indices = np.where(norms == 0)[0]
        raise ValueError(f"Zero vectors found at indices: {zero_indices[:5].tolist()}")
    
    return embeddings

def validate_query_vector(query_vector: Union[np.ndarray, List[float]], expected_dims: int) -> np.ndarray:
    """Validate query vector format and dimensions"""
    
    if not isinstance(query_vector, (list, np.ndarray)):
        raise TypeError("Query vector must be a list or numpy array")
    
    query_vector = np.asarray(query_vector, dtype=np.float32)
    
    if query_vector.ndim != 1:
        raise ValueError("Query vector must be one-dimensional")
    
    if len(query_vector) != expected_dims:
        raise ValueError(f"Query vector dimension mismatch: expected {expected_dims}, got {len(query_vector)}")
    
    if np.any(np.isnan(query_vector)):
        raise ValueError("NaN values detected in query vector")
    
    if np.any(np.isinf(query_vector)):
        raise ValueError("Infinite values detected in query vector")
    
    if np.linalg.norm(query_vector) == 0:
        raise ValueError("Query vector cannot be zero")
    
    return query_vector

def validate_level(level: str) -> str:
    """Validate precision level"""
    valid_levels = {"f8", "f16", "f32", "f64"}
    if level not in valid_levels:
        raise ValueError(f"Invalid level '{level}'. Must be one of: {', '.join(sorted(valid_levels))}")
    return level

def validate_top_k(top_k: int, max_rows: int) -> int:
    """Validate top_k parameter"""
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    
    if top_k > max_rows:
        raise ValueError(f"top_k ({top_k}) cannot be greater than number of vectors ({max_rows})")
    
    return top_k

def validate_method(method: str) -> str:
    """Validate search method"""
    valid_methods = {"simd", "scalar", "auto"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {', '.join(sorted(valid_methods))}")
    return method

def validate_similarity(similarity: str) -> str:
    """Validate similarity metric"""
    valid_similarities = {"cosine", "euclidean", "dot_product", "dot"}
    if similarity not in valid_similarities:
        raise ValueError(f"Invalid similarity '{similarity}'. Must be one of: {', '.join(sorted(valid_similarities))}")
    return similarity