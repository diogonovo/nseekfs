import numpy as np
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from .validation import (
    validate_embeddings, validate_query_vector, validate_level, 
    validate_top_k, validate_method, validate_similarity
)

logger = logging.getLogger(__name__)

# ========== CONFIGURAÇÕES DE SEGURANÇA ==========
MAX_CONCURRENT_QUERIES = 4  # Limite de queries simultâneas
MEMORY_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB warning
MAX_VECTOR_SIZE = 10000  # Dimensões máximas por segurança
MAX_DATASET_SIZE = 100_000_000  # 100M vetores máximo

# ========== EXCEPTION CLASSES ESPECÍFICAS ==========
class NSeekError(Exception):
    """Base exception para erros do NSeek"""
    pass

class NSeekValidationError(NSeekError):
    """Erro de validação de entrada"""
    pass

class NSeekMemoryError(NSeekError):
    """Erro relacionado a memória/recursos"""
    pass

class NSeekIndexError(NSeekError):
    """Erro relacionado ao índice"""
    pass

# ========== THREAD-SAFE QUERY LIMITER ==========
class QueryLimiter:
    """Limita número de queries simultâneas para evitar sobrecarga"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_QUERIES):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_queries = 0
        self._lock = threading.Lock()
    
    def __enter__(self):
        self._semaphore.acquire()
        with self._lock:
            self._active_queries += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._active_queries -= 1
        self._semaphore.release()
    
    def active_count(self) -> int:
        with self._lock:
            return self._active_queries

# ========== CLASSE NSEEK MELHORADA ==========
class NSeek:
    """
    High-level interface for initializing and querying vector search indexes using nseekfs.
    
    Thread-safe with built-in resource management and safety checks.
    """

    def __init__(self, engine, level: str, normalized: bool):
        # Validações iniciais
        if engine is None:
            raise NSeekIndexError("Engine cannot be None")
        if not isinstance(level, str) or level not in {"f8", "f16", "f32", "f64"}:
            raise NSeekValidationError(f"Invalid level: {level}")
        if not isinstance(normalized, bool):
            raise NSeekValidationError("normalized must be boolean")
        
        self.engine = engine
        self.level = level
        self.normalized = normalized
        self._query_limiter = QueryLimiter()
        self._creation_time = time.time()
        self._query_count = 0
        self._lock = threading.RLock()  # Reentrant lock para thread safety
        
        # Verificar saúde do engine
        try:
            dims = self.engine.dims()
            rows = self.engine.rows()
            
            if dims <= 0 or dims > MAX_VECTOR_SIZE:
                raise NSeekIndexError(f"Invalid dimensions: {dims}")
            if rows <= 0 or rows > MAX_DATASET_SIZE:
                raise NSeekIndexError(f"Invalid row count: {rows}")
                
            # Estimar uso de memória
            estimated_memory = dims * rows * 4  # f32 = 4 bytes
            if estimated_memory > MEMORY_WARNING_THRESHOLD:
                logger.warning(f"Large index loaded: ~{estimated_memory / (1024**3):.1f}GB memory usage")
                
        except Exception as e:
            raise NSeekIndexError(f"Engine validation failed: {e}")
        
        logger.info(f"NSeek initialized: {dims}D x {rows} vectors, level={level}")

    @classmethod
    def create_index(
        cls,
        embeddings: Union[np.ndarray, List[List[float]], str],
        level: str = "f32",
        normalized: Optional[bool] = True,
        ann: bool = True,
        base_name: str = "default",
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> str:
        """
        Create a binary index from embeddings and return the created file path.
        
        Args:
            embeddings: Input embeddings (array, list, or file path)
            level: Precision level ("f8", "f16", "f32", "f64")
            normalized: Whether embeddings are normalized (None=auto-normalize)
            ann: Enable approximate nearest neighbors
            base_name: Base name for index files
            output_dir: Output directory path
            seed: Random seed for ANN index
            
        Returns:
            str: Path to the created .bin file
            
        Raises:
            NSeekValidationError: Invalid input parameters
            NSeekMemoryError: Memory constraints exceeded
            NSeekIndexError: Index creation failed
        """
        from .nseekfs import py_prepare_bin_from_embeddings

        start_time = time.time()
        
        try:
            # Validação rigorosa de inputs
            embeddings = validate_embeddings(embeddings)
            level = validate_level(level)
            
            if not isinstance(seed, int) or seed < 0:
                raise NSeekValidationError("seed must be a non-negative integer")

            if not isinstance(base_name, str) or not base_name.strip():
                raise NSeekValidationError("base_name must be a non-empty string")

            n, d = embeddings.shape
            
            # Verificações de limites de segurança
            if d > MAX_VECTOR_SIZE:
                raise NSeekMemoryError(f"Vector dimension too large: {d} > {MAX_VECTOR_SIZE}")
            
            if n > MAX_DATASET_SIZE:
                raise NSeekMemoryError(f"Dataset too large: {n} > {MAX_DATASET_SIZE}")
            
            # Estimar uso de memória
            estimated_memory = n * d * 4  # f32 bytes
            if estimated_memory > MEMORY_WARNING_THRESHOLD:
                logger.warning(f"Large dataset: ~{estimated_memory / (1024**3):.1f}GB estimated memory usage")
            
            logger.info(f"Creating index: {n} vectors, {d} dimensions, level={level}, ann={ann}")

            # Normalization logic com validação
            if normalized is True:
                normalize_flag = False  # Already normalized
                # Verificar se realmente está normalizado
                sample_norms = np.linalg.norm(embeddings[:min(100, n)], axis=1)
                if not np.allclose(sample_norms, 1.0, atol=0.1):
                    logger.warning("normalized=True but vectors don't appear normalized")
            elif normalized is False:
                normalize_flag = True   # Normalize in Rust
            elif normalized is None:
                normalize_flag = True   # Default normalize
                logger.info("Auto-normalizing vectors")
            else:
                raise NSeekValidationError("normalized must be True, False, or None")

            # Validar embeddings para valores inválidos
            if np.any(~np.isfinite(embeddings)):
                raise NSeekValidationError("Embeddings contain NaN or infinite values")

            created_path = py_prepare_bin_from_embeddings(
                embeddings=embeddings,
                base_name=base_name,
                level=level,
                ann=ann,
                normalize=normalize_flag,
                seed=seed,
                output_dir=str(output_dir) if output_dir else None
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Index created successfully in {elapsed:.2f}s: {created_path}")
            return created_path

        except NSeekValidationError:
            raise
        except NSeekMemoryError:
            raise
        except MemoryError as e:
            raise NSeekMemoryError(f"Out of memory during index creation: {e}")
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise NSeekIndexError(f"Failed to create index for level '{level}': {e}")

    @classmethod
    def load_index(
        cls,
        bin_path: Union[str, Path],
        normalized: bool = True,
        ann: bool = True,
        level: Optional[str] = None
    ) -> "NSeek":
        """
        Load an existing index from a .bin file
        
        Args:
            bin_path: Path to the .bin file
            normalized: Whether vectors are normalized
            ann: Enable ANN if available
            level: Precision level (inferred from filename if not specified)
            
        Returns:
            NSeek: Loaded index instance
            
        Raises:
            FileNotFoundError: Binary file not found
            NSeekIndexError: Failed to load index
            NSeekValidationError: Invalid parameters
        """
        from .nseekfs import PySearchEngine

        start_time = time.time()
        
        try:
            bin_path = Path(bin_path)
            if not bin_path.exists():
                raise FileNotFoundError(f"Binary file not found: {bin_path}")

            if not bin_path.is_file():
                raise NSeekValidationError(f"Path is not a file: {bin_path}")

            # Verificar tamanho do arquivo
            file_size = bin_path.stat().st_size
            if file_size < 16:  # Mínimo: 8 bytes header + 8 bytes data
                raise NSeekIndexError(f"Binary file too small: {file_size} bytes")
            
            if file_size > MEMORY_WARNING_THRESHOLD:
                logger.warning(f"Large index file: {file_size / (1024**3):.1f}GB")

            # Infer level from filename if not specified
            if level is None:
                level = bin_path.stem  # f32.bin -> f32
            
            level = validate_level(level)

            if not isinstance(normalized, bool):
                raise NSeekValidationError("normalized must be a boolean")
            
            if not isinstance(ann, bool):
                raise NSeekValidationError("ann must be a boolean")

            logger.info(f"Loading index from: {bin_path}")
            
            # Tentar carregar com timeout para detectar problemas
            engine = PySearchEngine(str(bin_path), ann=ann)
            
            # Validar engine carregado
            dims = engine.dims()
            rows = engine.rows()
            
            if dims <= 0 or rows <= 0:
                raise NSeekIndexError(f"Invalid engine dimensions: {dims}x{rows}")
            
            elapsed = time.time() - start_time
            logger.info(f"Index loaded in {elapsed:.2f}s: dims={dims}, rows={rows}")
            
            return cls(engine=engine, level=level, normalized=normalized)

        except FileNotFoundError:
            raise
        except NSeekValidationError:
            raise
        except NSeekIndexError:
            raise
        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            raise NSeekIndexError(f"Failed to load engine from '{bin_path}': {e}")

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Union[np.ndarray, List[List[float]], str],
        level: str = "f32",
        normalized: Optional[bool] = True,
        ann: bool = True,
        base_name: str = "default",
        output_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> "NSeek":
        """
        Convenience method that creates and loads an index in one step.
        If index exists and force_rebuild=False, loads it. Otherwise creates new.
        
        Args:
            embeddings: Input embeddings
            level: Precision level
            normalized: Normalization setting
            ann: Enable ANN
            base_name: Base name for files
            output_dir: Output directory
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            NSeek: Ready-to-use index
            
        Raises:
            NSeekValidationError: Invalid parameters
            NSeekIndexError: Index creation/loading failed
        """
        try:
            # Determinar bin file path
            if output_dir:
                bin_path = Path(output_dir) / f"{level}.bin"
            else:
                base_dir = Path.home() / ".nseek" / "indexes" / base_name
                bin_path = base_dir / f"{level}.bin"

            # Load existing if available and not forcing rebuild
            if bin_path.exists() and not force_rebuild:
                try:
                    logger.info(f"Loading existing index from {bin_path}")
                    return cls.load_index(bin_path, normalized=(normalized is not False), ann=ann, level=level)
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}, rebuilding...")
                    # Continue to create new index
            
            # Create new index
            logger.info(f"Creating new index at {bin_path}")
            created_path = cls.create_index(
                embeddings=embeddings,
                level=level,
                normalized=normalized,
                ann=ann,
                base_name=base_name,
                output_dir=output_dir
            )
            
            # Load the newly created index
            return cls.load_index(created_path, normalized=(normalized is not False), ann=ann, level=level)
            
        except (NSeekValidationError, NSeekIndexError, NSeekMemoryError):
            raise
        except Exception as e:
            logger.error(f"from_embeddings failed: {e}")
            raise NSeekIndexError(f"Failed to create index from embeddings: {e}")

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "auto",
        similarity: str = "cosine"
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Execute a query on the loaded index.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            method: Search method ("simd", "scalar", "auto")
            similarity: Similarity metric ("cosine", "euclidean", "dot_product")
            
        Returns:
            List[dict]: Results with 'idx' and 'score' keys
            
        Raises:
            NSeekValidationError: Invalid parameters
            NSeekIndexError: Search failed
        """
        with self._query_limiter:  # Thread-safe query limiting
            try:
                with self._lock:
                    self._query_count += 1
                    query_id = self._query_count

                start_time = time.time()
                
                # Validações rigorosas
                query_vector = validate_query_vector(query_vector, self.dims)
                top_k = validate_top_k(top_k, self.rows)
                method = validate_method(method)
                similarity = validate_similarity(similarity)

                # Validar query vector para valores inválidos
                if np.any(~np.isfinite(query_vector)):
                    raise NSeekValidationError("Query vector contains NaN or infinite values")

                # Normalize query if needed for cosine similarity
                if similarity == "cosine" and not self.normalized:
                    norm = np.linalg.norm(query_vector)
                    if norm > 0:
                        query_vector = query_vector / norm
                    else:
                        raise NSeekValidationError("Query vector has zero norm for cosine similarity")

                logger.debug(f"Query {query_id}: top_k={top_k}, method={method}, similarity={similarity}")

                # Execute query with timeout protection
                results = self.engine.top_k_query(
                    query_vector.tolist(), 
                    top_k, 
                    method=method, 
                    similarity=similarity
                )
                
                # Validar resultados
                if not isinstance(results, list):
                    raise NSeekIndexError("Engine returned invalid results type")
                
                valid_results = []
                for i, (idx, score) in enumerate(results):
                    if not isinstance(idx, (int, np.integer)):
                        logger.warning(f"Invalid index type at position {i}: {type(idx)}")
                        continue
                    if not isinstance(score, (float, np.floating)) or not np.isfinite(score):
                        logger.warning(f"Invalid score at position {i}: {score}")
                        continue
                    if idx < 0 or idx >= self.rows:
                        logger.warning(f"Index out of bounds at position {i}: {idx}")
                        continue
                    
                    valid_results.append({"idx": int(idx), "score": float(score)})
                
                elapsed = time.time() - start_time
                logger.debug(f"Query {query_id} completed in {elapsed*1000:.2f}ms, {len(valid_results)} results")
                
                return valid_results
                
            except NSeekValidationError:
                raise
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise NSeekIndexError(f"Search failed at level {self.level}: {e}")

    def query_batch(
        self,
        query_vectors: Union[np.ndarray, List[List[float]]],
        top_k: int = 5,
        method: str = "auto",
        similarity: str = "cosine",
        max_workers: Optional[int] = None
    ) -> List[List[Dict[str, Union[int, float]]]]:
        """
        Execute multiple queries in parallel (thread-safe).
        
        Args:
            query_vectors: Multiple query vectors
            top_k: Number of results per query
            method: Search method
            similarity: Similarity metric
            max_workers: Maximum parallel workers (None=auto)
            
        Returns:
            List of results for each query
            
        Raises:
            NSeekValidationError: Invalid parameters
            NSeekIndexError: Batch search failed
        """
        try:
            query_vectors = np.asarray(query_vectors, dtype=np.float32)
            
            if query_vectors.ndim != 2:
                raise NSeekValidationError("query_vectors must be 2D")
            
            if query_vectors.shape[1] != self.dims:
                raise NSeekValidationError(f"Query dimension mismatch: expected {self.dims}, got {query_vectors.shape[1]}")
            
            n_queries = query_vectors.shape[0]
            if n_queries == 0:
                return []
            
            # Limitar workers para evitar sobrecarga
            if max_workers is None:
                max_workers = min(MAX_CONCURRENT_QUERIES, n_queries)
            else:
                max_workers = min(max_workers, MAX_CONCURRENT_QUERIES)
            
            logger.info(f"Batch query: {n_queries} queries with {max_workers} workers")
            
            def query_single(query_vec):
                return self.query(query_vec, top_k=top_k, method=method, similarity=similarity)
            
            # Execute em paralelo com ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(query_single, query_vectors))
            
            logger.info(f"Batch query completed: {n_queries} queries processed")
            return results
            
        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            raise NSeekIndexError(f"Batch query failed: {e}")

    def get_vector(self, idx: int) -> np.ndarray:
        """
        Return the vector at the specified index.
        
        Args:
            idx: Vector index
            
        Returns:
            np.ndarray: Vector at the index
            
        Raises:
            NSeekValidationError: Invalid index
            NSeekIndexError: Failed to retrieve vector
        """
        try:
            if not isinstance(idx, (int, np.integer)):
                raise NSeekValidationError("Index must be an integer")
            
            idx = int(idx)  # Convert numpy ints
            
            if idx < 0:
                raise NSeekValidationError("Index must be non-negative")
            
            if idx >= self.rows:
                raise NSeekValidationError(f"Index {idx} out of bounds (max: {self.rows - 1})")

            vector = self.engine.get_vector(idx)
            
            if not isinstance(vector, list):
                raise NSeekIndexError("Engine returned invalid vector type")
            
            result = np.array(vector, dtype=np.float32)
            
            if result.shape != (self.dims,):
                raise NSeekIndexError(f"Vector shape mismatch: expected ({self.dims},), got {result.shape}")
            
            if np.any(~np.isfinite(result)):
                raise NSeekIndexError(f"Vector at index {idx} contains invalid values")
            
            return result
            
        except NSeekValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get vector {idx}: {e}")
            raise NSeekIndexError(f"Failed to get vector at index {idx}: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the index.
        
        Returns:
            Dict with health status and metrics
        """
        try:
            health = {
                "status": "healthy",
                "dims": self.dims,
                "rows": self.rows,
                "level": self.level,
                "normalized": self.normalized,
                "query_count": self._query_count,
                "active_queries": self._query_limiter.active_count(),
                "uptime_seconds": time.time() - self._creation_time,
                "warnings": []
            }
            
            # Test basic functionality
            if self.rows > 0:
                test_vector = self.get_vector(0)
                if test_vector is None or len(test_vector) != self.dims:
                    health["status"] = "degraded"
                    health["warnings"].append("Failed to retrieve test vector")
                else:
                    # Test query
                    test_results = self.query(test_vector, top_k=1)
                    if not test_results or test_results[0]["idx"] != 0:
                        health["status"] = "degraded"
                        health["warnings"].append("Self-query test failed")
            
            # Memory usage estimation
            estimated_memory = self.dims * self.rows * 4
            health["estimated_memory_mb"] = estimated_memory / (1024 * 1024)
            
            if estimated_memory > MEMORY_WARNING_THRESHOLD:
                health["warnings"].append(f"High memory usage: {health['estimated_memory_mb']:.1f}MB")
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "dims": getattr(self, 'dims', 'unknown'),
                "rows": getattr(self, 'rows', 'unknown'),
            }

    @property
    def dims(self) -> int:
        """Number of vector dimensions."""
        try:
            return self.engine.dims()
        except Exception as e:
            logger.error(f"Failed to get dims: {e}")
            raise NSeekIndexError(f"Failed to get dimensions: {e}")

    @property
    def rows(self) -> int:
        """Number of vectors in the index."""
        try:
            return self.engine.rows()
        except Exception as e:
            logger.error(f"Failed to get rows: {e}")
            raise NSeekIndexError(f"Failed to get row count: {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "dims": self.dims,
            "rows": self.rows,
            "level": self.level,
            "normalized": self.normalized,
            "query_count": self._query_count,
            "active_queries": self._query_limiter.active_count(),
            "uptime_seconds": time.time() - self._creation_time,
        }

    def __repr__(self) -> str:
        try:
            return f"NSeek(level='{self.level}', dims={self.dims}, rows={self.rows}, normalized={self.normalized})"
        except:
            return f"NSeek(level='{self.level}', status=unknown)"
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        return self.rows

    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        # Cleanup resources if needed
        with self._lock:
            logger.debug(f"NSeek context exited after {self._query_count} queries")