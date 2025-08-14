"""
NSeekFS High-Level Python API with Timer Fix

This module provides the high-level, user-friendly API for NSeekFS vector search.
It wraps the low-level Rust engine with convenient Python interfaces.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for validation
MEMORY_WARNING_THRESHOLD = 2 * 1024**3  # 2GB
SUPPORTED_LEVELS = ["f8", "f16", "f32", "f64"]
MAX_DIMENSIONS = 10000
MIN_DIMENSIONS = 1
MAX_VECTORS = 100_000_000


class ValidationError(ValueError):
    """Raised when input validation fails"""
    pass


class IndexError(Exception):
    """Raised when index operations fail"""
    pass


def validate_embeddings(embeddings: Union[np.ndarray, List[List[float]], str]) -> np.ndarray:
    """Validate and convert embeddings to proper numpy format"""
    
    if isinstance(embeddings, str):
        raise ValidationError("String embeddings not supported. Use numpy array or list.")
    
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    
    if not isinstance(embeddings, np.ndarray):
        raise ValidationError(f"Embeddings must be numpy array or list, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise ValidationError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
    
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValidationError(f"Embeddings cannot be empty, got shape {embeddings.shape}")
    
    if embeddings.shape[1] > MAX_DIMENSIONS:
        raise ValidationError(f"Too many dimensions: {embeddings.shape[1]} (max: {MAX_DIMENSIONS})")
    
    if embeddings.shape[0] > MAX_VECTORS:
        raise ValidationError(f"Too many vectors: {embeddings.shape[0]} (max: {MAX_VECTORS})")
    
    # Convert to contiguous f32 array if needed
    if embeddings.dtype != np.float32:
        logger.info(f"Converting embeddings from {embeddings.dtype} to float32")
        embeddings = embeddings.astype(np.float32)
    
    if not embeddings.flags['C_CONTIGUOUS']:
        logger.info("Making embeddings C-contiguous")
        embeddings = np.ascontiguousarray(embeddings)
    
    # Check for invalid values
    invalid_mask = ~np.isfinite(embeddings)
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        total_count = embeddings.size
        percentage = (invalid_count / total_count) * 100
        
        if percentage > 1.0:
            raise ValidationError(f"Too many invalid values: {percentage:.1f}% ({invalid_count}/{total_count})")
        else:
            logger.warning(f"Found {invalid_count} invalid values ({percentage:.2f}%)")
    
    return embeddings


def validate_level(level: str) -> str:
    """Validate precision level"""
    if not isinstance(level, str):
        raise ValidationError(f"Level must be string, got {type(level)}")
    
    if level not in SUPPORTED_LEVELS:
        raise ValidationError(f"Invalid level '{level}'. Supported: {SUPPORTED_LEVELS}")
    
    return level


class VectorSearch:
    """
    High-level vector search interface with timing support
    
    This class provides a user-friendly API around the low-level Rust engine,
    with proper error handling, logging, and timing measurement.
    """
    
    def __init__(self, engine, level: str = "f32", normalized: bool = True):
        """Initialize VectorSearch with engine"""
        self.engine = engine
        self.level = level
        self.normalized = normalized
        self._creation_time = time.time()
        
        logger.info(f"VectorSearch initialized: {self.dims}D x {self.rows} vectors, level={level}")

    def __repr__(self) -> str:
        return f"VectorSearch(dims={self.dims}, rows={self.rows}, level='{self.level}', ann={self.has_ann})"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Engine cleanup is handled automatically by Rust Drop
        pass

    @property
    def dims(self) -> int:
        """Number of dimensions"""
        return self.engine.dims()

    @property
    def rows(self) -> int:
        """Number of vectors"""
        return self.engine.rows()

    @property
    def has_ann(self) -> bool:
        """Whether ANN is available"""
        return self.engine.has_ann()

    @property
    def memory_usage_mb(self) -> float:
        """Memory usage in MB"""
        return self.engine.memory_usage_mb()

    @property
    def stats(self) -> Dict[str, Any]:
        """Engine statistics"""
        queries, avg_time, ann, exact, simd, scalar, uptime = self.engine.get_stats()
        return {
            "total_queries": queries,
            "avg_query_time_ms": avg_time,
            "ann_queries": ann,
            "exact_queries": exact,
            "simd_queries": simd,
            "scalar_queries": scalar,
            "uptime_seconds": uptime,
            "memory_usage_mb": self.memory_usage_mb,
        }

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        method: Optional[str] = None,
        return_scores: bool = True,
        return_timing: bool = False,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Query the index with timing support
        
        Args:
            query_vector: Query vector as numpy array or list
            top_k: Number of results to return
            method: Search method ("auto", "exact", "ann", or None for auto)
            return_scores: Include similarity scores in results
            return_timing: Return timing information
            
        Returns:
            List of result dictionaries, optionally with timing info
            
        Example:
            results = index.query(query_vector, top_k=5)
            results, timing = index.query(query_vector, top_k=5, return_timing=True)
        """
        start_time = time.time()
        
        # Validate inputs
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        elif isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32)
        else:
            raise ValidationError(f"Query vector must be numpy array or list, got {type(query_vector)}")

        if query_vector.ndim != 1:
            raise ValidationError(f"Query vector must be 1D, got {query_vector.ndim}D")

        if len(query_vector) != self.dims:
            raise ValidationError(f"Query vector dimension mismatch: expected {self.dims}, got {len(query_vector)}")

        if top_k <= 0:
            raise ValidationError(f"top_k must be positive, got {top_k}")

        if top_k > self.rows:
            logger.warning(f"top_k ({top_k}) exceeds number of vectors ({self.rows}), limiting to {self.rows}")
            top_k = self.rows

        # Normalize query vector if index is normalized
        if self.normalized:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        try:
            # Use new timing-aware method if available
            if hasattr(self.engine, 'query_with_timing'):
                logger.debug("Using query_with_timing method")
                result = self.engine.query_with_timing(
                    query_vector.tolist(), 
                    top_k, 
                    method
                )
                
                # Convert to standard format
                results = []
                for item in result.results:
                    result_dict = {"idx": item.idx}
                    if return_scores:
                        result_dict["score"] = item.score
                    results.append(result_dict)

                timing_info = {
                    "query_time_ms": result.query_time_ms,
                    "method_used": result.method_used,
                    "candidates_generated": result.candidates_generated,
                    "simd_used": result.simd_used,
                }
                
            else:
                # Fallback to legacy method
                logger.debug("Using legacy query method")
                raw_results = self.engine.query(
                    query_vector.tolist(), 
                    top_k, 
                    method or "auto"
                )
                
                # Convert format
                results = []
                for item in raw_results:
                    result_dict = {"idx": item["idx"]}
                    if return_scores:
                        result_dict["score"] = item["score"]
                    results.append(result_dict)

                python_time = (time.time() - start_time) * 1000
                timing_info = {
                    "query_time_ms": python_time,
                    "method_used": method or "auto",
                    "candidates_generated": 0,
                    "simd_used": len(query_vector) >= 64,
                }

            logger.debug(f"Query completed: {len(results)} results in {timing_info['query_time_ms']:.3f}ms")

            if return_timing:
                return results, timing_info
            else:
                return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise IndexError(f"Query failed: {e}")

    def query_batch(
        self,
        query_vectors: Union[np.ndarray, List[List[float]]],
        top_k: int = 10,
        method: Optional[str] = None,
        return_scores: bool = True,
        return_timing: bool = False,
    ) -> Union[List[List[Dict[str, Any]]], Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]]:
        """
        Query multiple vectors at once
        
        Args:
            query_vectors: Multiple query vectors as 2D array
            top_k: Number of results per query
            method: Search method
            return_scores: Include similarity scores
            return_timing: Return timing information
            
        Returns:
            List of result lists, optionally with timing info
        """
        start_time = time.time()
        
        # Validate inputs
        if isinstance(query_vectors, list):
            query_vectors = np.array(query_vectors, dtype=np.float32)
        elif isinstance(query_vectors, np.ndarray):
            query_vectors = query_vectors.astype(np.float32)
        else:
            raise ValidationError(f"Query vectors must be numpy array or list, got {type(query_vectors)}")

        if query_vectors.ndim != 2:
            raise ValidationError(f"Query vectors must be 2D, got {query_vectors.ndim}D")

        if query_vectors.shape[1] != self.dims:
            raise ValidationError(f"Query vectors dimension mismatch: expected {self.dims}, got {query_vectors.shape[1]}")

        # Process each query
        all_results = []
        total_query_time = 0.0
        
        for i, query_vector in enumerate(query_vectors):
            try:
                result, timing = self.query(
                    query_vector, 
                    top_k=top_k, 
                    method=method, 
                    return_scores=return_scores, 
                    return_timing=True
                )
                all_results.append(result)
                total_query_time += timing["query_time_ms"]
                
            except Exception as e:
                logger.error(f"Batch query {i} failed: {e}")
                all_results.append([])  # Empty result for failed query

        batch_time = (time.time() - start_time) * 1000
        
        timing_info = {
            "total_time_ms": batch_time,
            "avg_query_time_ms": total_query_time / len(query_vectors),
            "queries_processed": len(query_vectors),
            "method_used": method or "auto",
        }

        logger.info(f"Batch query completed: {len(query_vectors)} queries in {batch_time:.2f}ms")

        if return_timing:
            return all_results, timing_info
        else:
            return all_results

    def debug_query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        force_method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Debug query with detailed timing information
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            force_method: Force specific method ("ann", "exact")
            
        Returns:
            Dictionary with results and detailed timing
        """
        # Validate and convert query
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        elif isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32)

        if len(query_vector) != self.dims:
            raise ValidationError(f"Query vector dimension mismatch: expected {self.dims}, got {len(query_vector)}")

        # Normalize if needed
        if self.normalized:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        debug_info = {
            "query_vector_norm": float(np.linalg.norm(query_vector)),
            "index_info": {
                "dims": self.dims,
                "rows": self.rows,
                "has_ann": self.has_ann,
                "level": self.level,
                "normalized": self.normalized,
            }
        }

        try:
            # Try specific debug methods if available
            if force_method == "ann" and hasattr(self.engine, 'query_ann_debug'):
                result = self.engine.query_ann_debug(query_vector.tolist(), top_k)
                debug_info["method_used"] = "ann_debug"
                
            elif force_method == "exact" and hasattr(self.engine, 'query_exact_debug'):
                result = self.engine.query_exact_debug(query_vector.tolist(), top_k)
                debug_info["method_used"] = "exact_debug"
                
            else:
                # Use regular method with timing
                result = self.engine.query_with_timing(query_vector.tolist(), top_k, force_method)
                debug_info["method_used"] = "unified"

            # Convert results
            results = [{"idx": item.idx, "score": item.score} for item in result.results]
            
            debug_info.update({
                "results": results,
                "timing": {
                    "query_time_ms": result.query_time_ms,
                    "method_used": result.method_used,
                    "candidates_generated": result.candidates_generated,
                    "simd_used": result.simd_used,
                },
                "performance": {
                    "results_per_ms": len(results) / max(result.query_time_ms, 0.001),
                    "candidates_per_result": result.candidates_generated / max(len(results), 1) if result.candidates_generated > 0 else 0,
                }
            })

            return debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"Debug query failed: {e}")
            return debug_info

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the index
        
        Returns:
            Dictionary with health status and diagnostics
        """
        health_info = {
            "status": "unknown",
            "engine_healthy": False,
            "basic_test_passed": False,
            "basic_test_time_ms": 0.0,
            "error": None,
        }

        try:
            # Check engine health
            engine_healthy = self.engine.health_check()
            health_info["engine_healthy"] = engine_healthy

            if not engine_healthy:
                health_info["status"] = "unhealthy"
                health_info["error"] = "Engine health check failed"
                return health_info

            # Basic query test
            if self.rows > 0:
                test_vector = np.random.randn(self.dims).astype(np.float32)
                if self.normalized:
                    test_vector = test_vector / np.linalg.norm(test_vector)

                start_time = time.time()
                results = self.query(test_vector, top_k=1, return_timing=False)
                test_time = (time.time() - start_time) * 1000

                health_info["basic_test_time_ms"] = test_time
                health_info["basic_test_passed"] = len(results) > 0

            # Overall status
            if health_info["engine_healthy"] and health_info["basic_test_passed"]:
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "degraded"

        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return health_info

    @classmethod
    def create_index(
        cls,
        embeddings: Union[np.ndarray, List[List[float]]],
        level: str = "f32",
        normalized: bool = True,
        ann: bool = True,
        base_name: str = "default",
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> str:
        """
        Create binary index file from embeddings
        
        Args:
            embeddings: Vector data as numpy array or list
            level: Precision level ("f8", "f16", "f32", "f64")
            normalized: Whether vectors are normalized
            ann: Enable approximate nearest neighbor
            base_name: Base filename for output
            output_dir: Output directory
            seed: Random seed for reproducibility
            
        Returns:
            str: Path to created binary index file
            
        Raises:
            ValidationError: Invalid input parameters
            MemoryError: Insufficient memory
            IndexError: Index creation failed
        """
        from .nseekfs import py_prepare_bin_from_embeddings

        start_time = time.time()
        
        try:
            # Validate and convert embeddings
            embeddings = validate_embeddings(embeddings)
            level = validate_level(level)
            
            # Memory check
            estimated_memory = embeddings.nbytes
            if estimated_memory > MEMORY_WARNING_THRESHOLD:
                logger.warning(f"Large embeddings: {estimated_memory / (1024**3):.1f}GB")
            
            # Setup output directory
            if output_dir is None:
                output_dir = Path.cwd() / "nseekfs_indexes"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Normalize if requested
            if normalized:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                embeddings = embeddings / norms
            
            logger.info(f"Creating {level} index: {embeddings.shape[0]} vectors x {embeddings.shape[1]} dims")
            
            # Create binary index
            bin_path = py_prepare_bin_from_embeddings(
                embeddings=embeddings,
                dims=embeddings.shape[1],
                rows=embeddings.shape[0],
                base_name=base_name,
                level=level,
                output_dir=str(output_dir),
                ann=ann,
                normalize=False,  # Already normalized above if requested
                seed=seed
            )
            
            elapsed = time.time() - start_time
            file_size = Path(bin_path).stat().st_size / (1024**2)  # MB
            logger.info(f"Index created in {elapsed:.2f}s: {bin_path} ({file_size:.1f}MB)")
            
            return bin_path
            
        except ValidationError:
            raise
        except MemoryError as e:
            logger.error(f"Memory error during index creation: {e}")
            raise MemoryError(f"Out of memory during index creation: {e}")
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise IndexError(f"Failed to create index for level '{level}': {e}")

    @classmethod
    def load_index(
        cls,
        bin_path: Union[str, Path],
        normalized: bool = True,
        ann: bool = True,
        level: Optional[str] = None
    ) -> "VectorSearch":
        """
        Load an existing index from a .bin file
        
        Args:
            bin_path: Path to binary index file  
            normalized: Whether vectors are normalized
            ann: Enable approximate nearest neighbor
            level: Precision level (auto-detected if None)
            
        Returns:
            VectorSearch: Loaded vector search index
            
        Raises:
            FileNotFoundError: Index file not found
            IndexError: Failed to load index
        """
        from .nseekfs import PySearchEngine

        start_time = time.time()
        
        try:
            bin_path = Path(bin_path)
            
            if not bin_path.exists():
                raise FileNotFoundError(f"Index file not found: {bin_path}")
          
            if not isinstance(ann, bool):
                raise ValidationError("ann must be a boolean")

            # Auto-detect level from filename if not provided
            if level is None:
                for supported_level in SUPPORTED_LEVELS:
                    if f"_{supported_level}." in str(bin_path):
                        level = supported_level
                        break
                else:
                    level = "f32"  # Default fallback

            logger.info(f"Loading index from: {bin_path}")
            
            # Try to load with timeout to detect issues
            engine = PySearchEngine(str(bin_path), ann=ann)
            
            # Validate loaded engine
            dims = engine.dims()
            rows = engine.rows()
            
            if dims <= 0 or rows <= 0:
                raise IndexError(f"Invalid engine dimensions: {dims}x{rows}")
            
            elapsed = time.time() - start_time
            logger.info(f"Index loaded in {elapsed:.2f}s: dims={dims}, rows={rows}")
            
            return cls(engine=engine, level=level, normalized=normalized)

        except FileNotFoundError:
            raise
        except ValidationError:
            raise
        except IndexError:
            raise
        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            raise IndexError(f"Failed to load engine from '{bin_path}': {e}")

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
    ) -> "VectorSearch":
        """
        Convenience method that creates and loads an index in one step.
        If index exists and force_rebuild=False, loads it. Otherwise creates new.
        
        This is the main entry point most users should use.
        
        Args:
            embeddings: Vector data as numpy array or list
            level: Precision level ("f8", "f16", "f32", "f64")  
            normalized: Whether to normalize vectors
            ann: Enable approximate nearest neighbor
            base_name: Base filename for index
            output_dir: Directory for index files
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            VectorSearch: Ready-to-query vector search index
            
        Example:
            import nseekfs
            import numpy as np
            
            vectors = np.random.randn(1000, 384).astype(np.float32)
            index = nseekfs.from_embeddings(vectors)
            results = index.query(query_vector, top_k=10)
        """
        try:
            # Setup paths
            if output_dir is None:
                output_dir = Path.cwd() / "nseekfs_indexes"
            else:
                output_dir = Path(output_dir)
            
            bin_path = output_dir / f"{base_name}_{level}.bin"
            
            # Load existing if available and not forcing rebuild
            if bin_path.exists() and not force_rebuild:
                logger.info(f"Loading existing index: {bin_path}")
                return cls.load_index(bin_path, normalized=normalized, ann=ann, level=level)
            
            # Create new index
            logger.info(f"Creating new index: {bin_path}")
            created_path = cls.create_index(
                embeddings=embeddings,
                level=level,
                normalized=normalized,
                ann=ann,
                base_name=base_name,
                output_dir=output_dir
            )
            
            # Load the created index
            return cls.load_index(created_path, normalized=normalized, ann=ann, level=level)
            
        except Exception as e:
            logger.error(f"from_embeddings failed: {e}")
            raise


# Backward compatibility aliases
VectorIndex = VectorSearch  # Legacy name
NSeek = VectorSearch        # Alternative name