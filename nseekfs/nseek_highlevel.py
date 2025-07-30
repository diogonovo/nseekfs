import numpy as np
import time
from nseekfs import prepare_bin_from_numpy
from nseek_hierarchical_engine import HierarchicalEngine
from sentence_transformers import SentenceTransformer
import os

def search_embeddings(
    embeddings,
    sentences,
    query_text=None,
    query_vector=None,
    levels=["f16", "f32"],
    top_k=5,
    base_path="nseek"
):
    """
    Pesquisa completa e automática com geração de .bin, engine e reranking.

    Args:
        embeddings (np.ndarray): matriz de embeddings (float32)
        sentences (list[str]): lista de textos originais
        query_text (str): texto da query (opcional se query_vector for fornecido)
        query_vector (list[float] ou np.ndarray): vetor da query (opcional se query_text for fornecido)
        levels (list[str]): níveis de precisão (ex: ["f16", "f32"])
        top_k (int): número de resultados finais
        base_path (str): prefixo para os ficheiros binários

    Returns:
        list[dict]: resultados com idx, score e frase
    """
    start_total = time.time()

    # 1. Garantir query válida
    if query_vector is None:
        if query_text is None:
            raise ValueError("You must provide either query_text or query_vector")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode([query_text], normalize_embeddings=True)[0]

    # 2. Gerar binários (se não existirem)
    for level in levels:
        path = f"{base_path}_{level}.bin"
        if not os.path.exists(path):
            prepare_bin_from_numpy(embeddings, level, base_path)

    # 3. Criar engine
    engine = HierarchicalEngine({lvl: f"{base_path}_{lvl}.bin" for lvl in levels})

    # 4. Pesquisa
    start = time.time()
    results = engine.search(query_vector, path=levels, top_k=top_k)
    elapsed = time.time() - start

    # 5. Formatar saída
    formatted = [
        {"idx": idx, "score": score, "text": sentences[idx]}
        for idx, score in results
    ]

    print(f"✅ Pesquisa em {levels} concluída em {elapsed:.4f}s (total {time.time() - start_total:.2f}s)")
    return formatted
