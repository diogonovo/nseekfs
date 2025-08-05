import os
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class NSeek:
    """
    High-level interface for initializing and querying vector search indexes using nseekfs.
    """
    def __init__(self, engine, level: str):
        self.engine = engine
        self.level = level

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Union[np.ndarray, List[List[float]], str],
        level: str = "f32",
        similarity: str = "cosine",
        use_ann: bool = True,
        base_dir: str = "nseek_indexes",
        base_name: str = "default"
    ) -> "NSeek":
        """
        Initializes the engine from embeddings (array, list, or file path).
        Generates a binary index file if it doesn't exist.
        """
        if similarity != "cosine":
            raise ValueError("Only 'cosine' similarity is supported in this version.")

        if isinstance(embeddings, str):
            if embeddings.endswith(".npy"):
                embeddings = np.load(embeddings)
            elif embeddings.endswith(".csv"):
                embeddings = np.loadtxt(embeddings, delimiter=",")
            else:
                raise ValueError("Unsupported file format. Only .npy and .csv are supported.")

        elif isinstance(embeddings, (list, np.ndarray)):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        else:
            raise TypeError("Embeddings must be a numpy array, list of vectors, or a path to .npy or .csv file.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert isinstance(embeddings, np.ndarray)  # type checker


        if level not in {"f8", "f16", "f32", "f64"}:
            raise ValueError("Level must be one of: 'f8', 'f16', 'f32', 'f64'.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array with shape (n_samples, dimension).")

        n, d = embeddings.shape
        if n < 1:
            raise ValueError("At least one embedding is required.")
        if d < 8:
            raise ValueError("Each embedding must have at least 8 dimensions.")
        if d > 4096:
            raise ValueError("Embedding dimension too large. Must be <= 4096.")

        #norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        #norms[norms == 0] = 1e-12
        #embeddings = embeddings / norms

        index_dir = os.path.abspath(os.path.join(base_dir, base_name))
        os.makedirs(index_dir, exist_ok=True)

        from .nseekfs import py_prepare_bin_from_embeddings, PySearchEngine

        bin_path = os.path.join(index_dir, f"{level}.bin")
        if not os.path.exists(bin_path):
            logger.info(f"Creating binary file for level '{level}' at {bin_path}")
            try:
                path_created = py_prepare_bin_from_embeddings(
                    embeddings.tolist(),  
                    d,
                    bin_path,
                    level,
                    use_ann,
                    normalize=False,
                    seed=42
               )

                engine = PySearchEngine(path_created, use_ann=use_ann)
            except Exception as e:
                raise RuntimeError(f"Failed to create binary for level {level}: {e}")
        else:
            engine = PySearchEngine(bin_path, use_ann=use_ann)

        return cls(engine=engine, level=level)

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "simd"
    ) -> List[dict]:
        """
        Queries the index with a normalized vector and returns the top-k matches.

        Args:
            query_vector: The input vector to search for.
            top_k: Number of nearest neighbors to return.
            method: Search method to use: "simd", "scalar", or "auto". Default is "simd".

        Returns:
            List of dictionaries with keys 'idx' and 'score'.
        """
        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be one-dimensional.")

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query vector cannot be zero.")

        query_vector = query_vector / norm

        try:
            results = self.engine.top_k_query(
                query_vector.tolist(),
                top_k,
                method=method
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed at level {self.level}: {e}")

        return [
            {"idx": int(idx), "score": float(score)}
            for idx, score in results
        ]

