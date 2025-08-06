import numpy as np
from typing import List, Union
import logging
from pathlib import Path

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
        base_name: str = "default"
    ) -> "NSeek":
        if similarity != "cosine":
            raise ValueError("Only 'cosine' similarity is supported.")

        # Carregamento dos embeddings
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
            raise TypeError("Embeddings must be a numpy array, list of vectors, or a valid file path.")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Verificações
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (n_samples, dim).")
        n, d = embeddings.shape
        if n < 1:
            raise ValueError("At least one embedding is required.")
        if d < 8 or d > 4096:
            raise ValueError("Embedding dimension must be between 8 and 4096.")
        if level not in {"f8", "f16", "f32", "f64"}:
            raise ValueError("Invalid level. Must be one of: 'f8', 'f16', 'f32', 'f64'.")

        from .nseekfs import py_prepare_bin_from_embeddings, PySearchEngine

        # Caminho local seguro
        bin_path = Path("nseek_temp") / base_name / f"{level}.bin"
        bin_path.parent.mkdir(parents=True, exist_ok=True)

        if not bin_path.exists():
            logger.info(f"Creating binary index at {bin_path}")
            try:
                print(f"🧪 BIN PATH: {bin_path}")
                created_path = py_prepare_bin_from_embeddings(
                    embeddings.tolist(),
                    d,
                    str(bin_path),
                    level,
                    use_ann,
                    normalize=False,
                    seed=42
                )
                engine = PySearchEngine(created_path, use_ann=use_ann)
            except Exception as e:
                logger.error(f"Binary creation failed: {e}")
                raise RuntimeError(f"Failed to create binary for level '{level}': {e}")
        else:
            engine = PySearchEngine(str(bin_path), use_ann=use_ann)

        return cls(engine=engine, level=level)

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "simd"
    ) -> List[dict]:
        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be one-dimensional.")

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query vector cannot be zero.")
        query_vector /= norm

        try:
            results = self.engine.top_k_query(query_vector.tolist(), top_k, method=method)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed at level {self.level}: {e}")

        return [{"idx": int(idx), "score": float(score)} for idx, score in results]
