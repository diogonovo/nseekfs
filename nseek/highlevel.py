import os
import time
import numpy as np
import logging
from typing import List, Union
#from sentence_transformers import SentenceTransformer
#from nseek import prepare_engine_from_embeddings, PySearchEngine
#from nseek_hierarchical_engine import HierarchicalEngine
from .engine_core import prepare_engine_from_embeddings, PySearchEngine

logger = logging.getLogger(__name__)

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
    Performs hierarchical search over sentence embeddings.

    Args:
        embeddings: np.ndarray with shape (n, d)
        sentences: List of texts (length n)
        query_text: Optional query string (encoded if no query_vector)
        query_vector: Optional precomputed query vector
        levels: Precision levels to search through
        top_k: Number of top results to return
        base_path: Prefix for binary files
        disable_ann: If True, disables approximate search

    Returns:
        List of dicts with keys: idx, score, text
    """
    from . import prepare_engine_from_embeddings, PySearchEngine
    from .hierarchical import HierarchicalEngine

    start_total = time.time()
    logger.info(f"Starting search_embeddings → top_k={top_k}, levels={levels}, base='{base_path}', ANN={'disabled' if disable_ann else 'enabled'}")

    if query_vector is None:
        raise ValueError("Provide 'query_vector'")
        #if query_text is None:
        #    raise ValueError("Provide either 'query_text' or 'query_vector'")
        #logger.info("Encoding query text using sentence-transformers")
        #model = SentenceTransformer("all-MiniLM-L6-v2")
        #query_vector = model.encode([query_text], normalize_embeddings=True)[0]

    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.shape[-1] != embeddings.shape[-1]:
        raise ValueError(f"Dimensionality mismatch: query={query_vector.shape[-1]}, embeddings={embeddings.shape[-1]}")

    paths = {}
    for level in levels:
        filename = f"{base_path}_{level}.bin"
        if not os.path.exists(filename):
            logger.warning(f"Binary not found for level '{level}', creating: {filename}")
            try:
                path_created = prepare_engine_from_embeddings(
                    embeddings,
                    base_path,
                    level,
                    normalize=True,
                    use_ann=not disable_ann
                )
                paths[level] = path_created
                logger.info(f"Created binary file: {path_created}")
            except Exception as e:
                logger.error(f"Failed to create binary for level {level}: {e}")
                raise RuntimeError(f"Error creating binary '{filename}': {e}")
        else:
            paths[level] = filename
            logger.debug(f"Using existing binary file for level {level}: {filename}")

    try:
        engine = HierarchicalEngine(paths, disable_ann=disable_ann)
    except Exception as e:
        logger.error(f"Failed to initialize HierarchicalEngine: {e}")
        raise RuntimeError(f"Error initializing HierarchicalEngine: {e}")

    start = time.time()
    try:
        results = engine.search(query_vector, path=levels, top_k=top_k)
    except Exception as e:
        logger.exception(f"Error during hierarchical search: {e}")
        raise RuntimeError(f"Error during search: {e}")
    
    elapsed = time.time() - start

    formatted = []
    for idx, score in results:
        if isinstance(idx, int) and 0 <= idx < len(sentences):
            formatted.append({
                "idx": idx,
                "score": float(score),
                "text": sentences[idx]
            })

    total_time = time.time() - start_total
    logger.info(f"Search completed in {elapsed:.4f}s (total {total_time:.2f}s)")

    return formatted
