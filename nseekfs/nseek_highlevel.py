import numpy as np
import time
import os
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
    Pesquisa hierárquica com embeddings e frases.
    """
    start_total = time.time()

    # Gerar vetor de consulta se query_text for usado
    if query_vector is None:
        if query_text is None:
            raise ValueError("You must provide either query_text or query_vector")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode([query_text], normalize_embeddings=True)[0]

    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.shape[-1] != embeddings.shape[-1]:
        raise ValueError("Dimensionality mismatch between query vector and embeddings")

    # Garantir binário para cada nível
    paths = {}
    for level in levels:
        expected_path = f"{base_path}_{level}.bin"
        if not os.path.exists(expected_path):
            print(f"🛠️ A criar {expected_path}...")
            try:
                actual_path = prepare_engine_from_embeddings(
                    embeddings,
                    base_path,
                    level,
                    True,
                    not disable_ann
                )
                paths[level] = actual_path  # <- garante o caminho real
            except Exception as e:
                raise RuntimeError(f"Falha ao criar binário para {level}: {e}")
        else:
            paths[level] = expected_path



    # Criar engine hierárquica
    #paths = {lvl: f"{base_path}_{lvl}.bin" for lvl in levels}
    try:
        engine = HierarchicalEngine(paths, disable_ann=disable_ann)
    except Exception as e:
        raise RuntimeError(f"Erro ao inicializar HierarchicalEngine: {e}")

    # Pesquisa
    start = time.time()
    try:
        results = engine.search(query_vector, path=levels, top_k=top_k)
    except Exception as e:
        raise RuntimeError(f"Erro na pesquisa: {e}")
    elapsed = time.time() - start

    if not isinstance(results, list):
        raise RuntimeError("Engine search returned unexpected result type.")

    formatted = []
    for r in results:
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        idx, score = r
        if not isinstance(idx, int) or not isinstance(score, (float, np.floating)):
            continue
        if idx < 0 or idx >= len(sentences):
            continue
        formatted.append({
            "idx": idx,
            "score": float(score),
            "text": sentences[idx]
        })

    print(f"✅ Pesquisa em {levels} concluída em {elapsed:.4f}s (total {time.time() - start_total:.2f}s)")
    return formatted
