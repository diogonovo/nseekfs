import os
import time
import numpy as np
from nseek_highlevel import search_embeddings

# Configuração geral
NIVEIS_TESTE = [
    ["f16"],
    ["f32"],
    ["f16", "f32"],
    ["f8", "f16", "f32"],
    ["f8", "f16", "f32", "f64"]
]

TOP_K = 5
BASE = "simulado"
QUERY_IDX = 1234

# 1. Carregar embeddings e sentenças
sentencas = np.load("sentencas.npy", allow_pickle=True).tolist()
embeddings = np.load("embeddings.npy")
query_vector = embeddings[QUERY_IDX]
query_text = sentencas[QUERY_IDX]

print(f"🔍 A testar a mesma query: '{query_text}'\n")

# 2. Executar benchmarks
for niveis in NIVEIS_TESTE:
    print(f"⚙️ Teste com caminho: {niveis}")
    start = time.time()
    resultados = search_embeddings(
        embeddings=embeddings,
        sentences=sentencas,
        query_vector=query_vector,
        query_text=None,
        levels=niveis,
        top_k=TOP_K,
        base_path=BASE
    )
    total = time.time() - start

    print(f"⏱️ Tempo total: {total:.3f}s")
    print("Top resultados:")
    for i, r in enumerate(resultados, 1):
        print(f"  #{i}: score={r['score']:.6f}, frase='{r['text']}'")
    print("\n" + "+" * 80 + "\n")
