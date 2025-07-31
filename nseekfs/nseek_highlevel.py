import os
import time
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from nseekfs import prepare_engine_from_embeddings, PySearchEngine
from nseek_hierarchical_engine import HierarchicalEngine

def search_embeddings(
    embeddings: np.ndarray,
    sentences: List[str],
    query_text: Union[str, None] = None,
    query_vector: Union[np.ndarray, List[float], None] = None,
    levels: List[str] = ["f16", "f32"],
    top_k: int = 5,
    base_path: str = "nseek",
    disable_ann: bool = False
) -> List[dict]:
    """
    Realiza pesquisa hierárquica entre frases, usando embeddings.
    Gera binários caso ainda não existam.

    Args:
        embeddings: np.ndarray (shape: [n, d])
        sentences: Lista de frases (mesmo número que embeddings)
        query_text: Texto para codificar (usa modelo)
        query_vector: Vetor já codificado (opcional)
        levels: Níveis a usar na hierarquia (ex: ["f8", "f16", "f32"])
        top_k: Número de resultados a devolver
        base_path: Prefixo dos ficheiros binários
        disable_ann: Se True, desativa aproximação ANN

    Returns:
        Lista de dicionários: {idx, score, text}
    """
    start_total = time.time()

    if query_vector is None:
        if query_text is None:
            raise ValueError("Provide either 'query_text' or 'query_vector'")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode([query_text], normalize_embeddings=True)[0]

    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.shape[-1] != embeddings.shape[-1]:
        raise ValueError(f"Dimensionality mismatch: query={query_vector.shape[-1]}, embeddings={embeddings.shape[-1]}")

    paths = {}
    for level in levels:
        filename = f"{base_path}_{level}.bin"
        if not os.path.exists(filename):
            print(f"🛠️ A criar {filename}...")
            try:
                path_created = prepare_engine_from_embeddings(
                    embeddings,
                    base_path,
                    level,
                    normalize=True,
                    use_ann=not disable_ann
                )
                paths[level] = path_created
            except Exception as e:
                raise RuntimeError(f"Erro ao criar binário '{filename}': {e}")
        else:
            paths[level] = filename

    try:
        engine = HierarchicalEngine(paths, disable_ann=disable_ann)
    except Exception as e:
        raise RuntimeError(f"Erro ao inicializar HierarchicalEngine: {e}")

    start = time.time()
    try:
        results = engine.search(query_vector, path=levels, top_k=top_k)
    except Exception as e:
        raise RuntimeError(f"Erro na pesquisa: {e}")
    elapsed = time.time() - start

    formatted = []
    for idx, score in results:
        if isinstance(idx, int) and 0 <= idx < len(sentences):
            formatted.append({
                "idx": idx,
                "score": float(score),
                "text": sentences[idx]
            })

    print(f"✅ Pesquisa em {levels} concluída em {elapsed:.4f}s (total {time.time() - start_total:.2f}s)")
    return formatted
