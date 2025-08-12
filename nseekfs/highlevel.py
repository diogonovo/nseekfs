import numpy as np
from typing import List, Union, Optional
import logging
from pathlib import Path
import time
from .validation import (
    validate_embeddings, validate_query_vector, validate_level, 
    validate_top_k, validate_method, validate_similarity
)

logger = logging.getLogger(__name__)

class NSeek:
    """
    High-level interface for initializing and querying vector search indexes using nseekfs.
    """

    def __init__(self, engine, level: str, normalized: bool):
        self.engine = engine
        self.level = level
        self.normalized = normalized

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
            ValueError: Invalid input parameters
            RuntimeError: Index creation failed
        """
        from .nseekfs import py_prepare_bin_from_embeddings

        # Validate inputs
        embeddings = validate_embeddings(embeddings)
        level = validate_level(level)
        
        if not isinstance(ann, bool):
            raise TypeError("ann must be a boolean")
        
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        n, d = embeddings.shape
        logger.info(f"Creating index: {n} vectors, {d} dimensions, level={level}, ann={ann}")

        # Normalization logic
        if normalized is True:
            normalize_flag = False  # Already normalized
        elif normalized is False:
            normalize_flag = True   # Normalize in Rust
        elif normalized is None:
            normalize_flag = True   # Default normalize
        else:
            raise ValueError("normalized must be True, False, or None")

        try:
            created_path = py_prepare_bin_from_embeddings(
                embeddings=embeddings,
                base_name=base_name,
                level=level,
                ann=ann,
                normalize=normalize_flag,
                seed=seed,
                output_dir=str(output_dir) if output_dir else None
            )
            
            logger.info(f"Index created successfully at: {created_path}")
            return created_path

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise RuntimeError(f"Failed to create index for level '{level}': {e}")

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
            RuntimeError: Failed to load index
        """
        from .nseekfs import PySearchEngine

        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        # Infer level from filename if not specified
        if level is None:
            level = bin_path.stem  # f32.bin -> f32
        
        level = validate_level(level)

        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        
        if not isinstance(ann, bool):
            raise TypeError("ann must be a boolean")

        try:
            logger.info(f"Loading index from: {bin_path}")
            engine = PySearchEngine(str(bin_path), ann=ann)
            logger.info(f"Index loaded: dims={engine.dims()}, rows={engine.rows()}")
            
            return cls(engine=engine, level=level, normalized=normalized)

        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            raise RuntimeError(f"Failed to load engine from '{bin_path}': {e}")

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
        """
        # Determine bin file path
        if output_dir:
            bin_path = Path(output_dir) / f"{level}.bin"
        else:
            base_dir = Path.home() / ".nseek" / "indexes" / base_name
            bin_path = base_dir / f"{level}.bin"

        # Load existing if available and not forcing rebuild
        if bin_path.exists() and not force_rebuild:
            logger.info(f"Loading existing index from {bin_path}")
            return cls.load_index(bin_path, normalized=(normalized is not False), ann=ann, level=level)
        
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

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "auto",
        similarity: str = "cosine"
    ) -> List[dict]:
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
            ValueError: Invalid parameters
            RuntimeError: Search failed
        """
        # Validate inputs
        query_vector = validate_query_vector(query_vector, self.dims)
        top_k = validate_top_k(top_k, self.rows)
        method = validate_method(method)
        similarity = validate_similarity(similarity)

        # Normalize query if needed for cosine similarity
        if similarity == "cosine" and not self.normalized:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        try:
            results = self.engine.top_k_query(
                query_vector.tolist(), 
                top_k, 
                method=method, 
                similarity=similarity
            )
            
            return [{"idx": int(idx), "score": float(score)} for idx, score in results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed at level {self.level}: {e}")

    def get_vector(self, idx: int) -> np.ndarray:
        """
        Return the vector at the specified index.
        
        Args:
            idx: Vector index
            
        Returns:
            np.ndarray: Vector at the index
            
        Raises:
            ValueError: Invalid index
            RuntimeError: Failed to retrieve vector
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        
        if idx < 0:
            raise ValueError("Index must be non-negative")
        
        if idx >= self.rows:
            raise ValueError(f"Index {idx} out of bounds (max: {self.rows - 1})")

        try:
            vector = self.engine.get_vector(idx)
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to get vector at index {idx}: {e}")

    @property
    def dims(self) -> int:
        """Number of vector dimensions."""
        return self.engine.dims()

    @property
    def rows(self) -> int:
        """Number of vectors in the index."""
        return self.engine.rows()

    def __repr__(self) -> str:
        return f"NSeek(level='{self.level}', dims={self.dims}, rows={self.rows}, normalized={self.normalized})"
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        return self.rows