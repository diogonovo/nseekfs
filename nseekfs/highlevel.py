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
    def create_index(
        cls,
        embeddings: Union[np.ndarray, List[List[float]], str],
        level: str = "f32",
        normalized: Optional[bool] = True,
        ann: bool = True,
        base_name: str = "default",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Cria um índice binário a partir de embeddings e retorna o caminho do ficheiro criado.
        
        Returns:
            str: Caminho para o ficheiro .bin criado
        """
        from .nseekfs import py_prepare_bin_from_embeddings

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

        # Lógica de normalização
        if normalized is True:
            normalize_flag = False  # já vem normalizado
        elif normalized is False:
            normalize_flag = True   # normalizar no Rust
        elif normalized is None:
            normalize_flag = True   # default
        else:
            raise ValueError("Invalid value for 'normalized'. Must be True, False or None.")

        logger.info(f"Creating binary index with level={level}, ann={ann}, normalize={normalize_flag}")
        
        try:
            created_path = py_prepare_bin_from_embeddings(
                embeddings=embeddings,
                base_name=base_name,
                level=level,
                ann=ann,
                normalize=normalize_flag,
                seed=42,
                output_dir=str(output_dir) if output_dir else None
            )
            logger.info(f"Index created successfully at: {created_path}")
            return created_path

        except Exception as e:
            logger.error(f"Binary creation failed: {e}")
            raise RuntimeError(f"Failed to create binary for level '{level}': {e}")

    @classmethod
    def load_index(
        cls,
        bin_path: Union[str, Path],
        normalized: bool = True,
        ann: bool = True,
        level: Optional[str] = None
    ) -> "NSeek":
        """
        Carrega um índice existente a partir de um ficheiro .bin
        
        Args:
            bin_path: Caminho para o ficheiro .bin
            normalized: Se os vetores estão normalizados
            ann: Se deve usar ANN (se disponível)
            level: Nível de precisão (inferido do nome do ficheiro se não especificado)
        """
        from .nseekfs import PySearchEngine

        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        # Inferir o level do nome do ficheiro se não especificado
        if level is None:
            level = bin_path.stem  # f32.bin -> f32

        # Lógica de normalização para o engine
        if normalized is True:
            normalize_flag = False  # já vem normalizado
        elif normalized is False:
            normalize_flag = True   # normalizar no Rust
        else:
            raise ValueError("Invalid value for 'normalized'. Must be True or False.")

        try:
            logger.info(f"Loading index from: {bin_path}")
            engine = PySearchEngine(str(bin_path), normalize_flag, ann=ann)
            logger.info(f"Index loaded successfully: dims={engine.dims()}, rows={engine.rows()}")
            
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
    ) -> "NSeek":
        """
        Método de conveniência que cria e carrega um índice em um só passo.
        Se o índice já existir, carrega-o. Caso contrário, cria um novo.
        """
        # Determinar o caminho do arquivo bin
        if output_dir:
            bin_path = Path(output_dir) / f"{level}.bin"
        else:
            base_dir = Path.home() / ".nseek" / "indexes" / base_name
            bin_path = base_dir / f"{level}.bin"

        # Se o arquivo já existe, carregar
        if bin_path.exists():
            logger.info(f"Loading existing index from {bin_path}")
            return cls.load_index(bin_path, normalized=(normalized is not False), ann=ann, level=level)
        
        # Caso contrário, criar novo índice
        logger.info(f"Creating new index at {bin_path}")
        created_path = cls.create_index(
            embeddings=embeddings,
            level=level,
            normalized=normalized,
            ann=ann,
            base_name=base_name,
            output_dir=output_dir
        )
        
        # Carregar o índice recém-criado
        return cls.load_index(created_path, normalized=(normalized is not False), ann=ann, level=level)

    def query(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        method: str = "simd",
        similarity: str = "cosine"
    ) -> List[dict]:
        """
        Executa uma query no índice carregado.
        """
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

    def get_vector(self, idx: int) -> np.ndarray:
        """
        Retorna o vetor no índice especificado.
        """
        try:
            vector = self.engine.get_vector(idx)
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to get vector at index {idx}: {e}")

    @property
    def dims(self) -> int:
        """Número de dimensões dos vetores."""
        return self.engine.dims()

    @property
    def rows(self) -> int:
        """Número de vetores no índice."""
        return self.engine.rows()

    def __repr__(self) -> str:
        return f"NSeek(level='{self.level}', dims={self.dims}, rows={self.rows}, normalized={self.normalized})"