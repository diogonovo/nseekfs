import numpy as np
from typing import List, Union, Optional
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)
start = time.time()

class NSeek:
    """
    High-level interface for initializing and querying vector search indexes using nseekfs.
    """

    def __init__(self, engine, level: str, normalized: bool):
        self.engine = engine
        self.level = level
        self.normalized = normalized

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Union[np.ndarray, List[List[float]], str],
        level: str = "f32",
        normalized: Optional[bool] = True,
        ann: bool = True,
        base_name: str = "default",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "NSeek":
        from .nseekfs import py_prepare_bin_from_embeddings, PySearchEngine

        if isinstance(embeddings, str):
            if embeddings.endswith(".npy"):
                embeddings = np.load(embeddings)
            elif embeddings.endswith(".csv"):
                embeddings = np.loadtxt(embeddings, delimiter=",")
            else:
                raise ValueError("Unsupported file format. Only .npy and .csv are supported.")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (n_samples, dim).")

        n, d = embeddings.shape
        if n < 1:
            raise ValueError("At least one embedding is required.")
        if d < 8 or d > 4096:
            raise ValueError("Embedding dimension must be between 8 and 4096.")
        if level not in {"f8", "f16", "f32", "f64"}:
            raise ValueError("Invalid level. Must be one of: 'f8', 'f16', 'f32', 'f64'.")

        # ⚠️ Corrigido: lógica de normalização
        if normalized is True:
            normalize_flag = False  # já vem normalizado
        elif normalized is False:
            normalize_flag = True   # normalizar no Rust
        elif normalized is None:
            normalize_flag = True   # default
        else:
            raise ValueError("Invalid value for 'normalized'. Must be True, False or None.")

        base_dir = Path(output_dir) if output_dir else Path.home() / ".nseek" / "indexes"
        bin_path = base_dir / base_name / f"{level}.bin"
        bin_path.parent.mkdir(parents=True, exist_ok=True)

        if not bin_path.exists():
            logger.info(f"Creating binary index at {bin_path}")
            try:
                print(f"🧪 BIN PATH: {bin_path}")
                print(f"⏱️ Antes do py_prepare_bin {time.time() - start:.2f}s")

                # ✅ Embeddings como numpy array 2D direto
                created_path = py_prepare_bin_from_embeddings(
                    embeddings=embeddings,  # ← não usar tolist()
                    base_name=base_name,
                    level=level,
                    ann=ann,
                    normalize=normalize_flag,
                    seed=42,
                    output_dir=str(output_dir) if output_dir else None
                )

                print(f"⏱️ Antes do pysearchengine: {time.time() - start:.2f}s")
                engine = PySearchEngine(str(created_path), normalize_flag, ann=ann)

            except Exception as e:
                logger.error(f"Binary creation failed: {e}")
                raise RuntimeError(f"Failed to create binary for level '{level}': {e}")
        else:
            engine = PySearchEngine(str(bin_path), normalize_flag, ann=ann)

        print(f"⏱️ Antes do Return: {time.time() - start:.2f}s")
        return cls(engine=engine, level=level, normalized=(normalized is not False))

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "simd",
        similarity: str = "cosine"
    ) -> List[dict]:
        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be one-dimensional.")

        if similarity == "cosine":
            if not self.normalized:
                norm = np.linalg.norm(query_vector)
                if norm == 0:
                    raise ValueError("Query vector cannot be zero.")
                query_vector /= norm
        else:
            raise ValueError(f"Similarity '{similarity}' not supported. Only 'cosine' is available for now.")

        try:
            results = self.engine.top_k_query(query_vector.tolist(), top_k, method=method, similarity=similarity)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed at level {self.level}: {e}")

        return [{"idx": int(idx), "score": float(score)} for idx, score in results]
