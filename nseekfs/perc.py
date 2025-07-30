import os
import numpy as np
import time
from nseekfs import prepare_bin_from_numpy
from nseek_highlevel import search_embeddings

# -------------------- CONFIGURAÇÃO -------------------- #
MODOS = [
    ["f32"],
    ["f16", "f32"],
    ["f8", "f16", "f32"],
    ["f8", "f16", "f32", "f64"]
]

N_VETORES = [
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    2_000_000
    # Podes aumentar até 10_000_000 se tiveres memória
]

PERCENTAGENS_TESTE = {
    "f8": [0.01],
    "f16": [0.2, 0.1, 0.05, 0.02, 0.01],
    "f32": [1.0]
}

BASE = "benchmark"

# -------------------- FUNÇÕES AUXILIARES -------------------- #
def normalizar(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def gerar_embeddigs(N, D=384):
    print(f"🔧 A gerar {N} vetores sintéticos...")
    np.random.seed(42)
    E = np.random.randn(N, D).astype(np.float32)
    return normalizar(E)

def gerar_frases(N):
    return [f"frase_{i}" for i in range(N)]

def garantir_bin(level, base_path, embeddings):
    path = f"{base_path}_{level}.bin"
    if not os.path.exists(path):
        print(f"💾 A gerar binário para {level}...")
        prepare_bin_from_numpy(embeddings, level, base_path)
    return path

# -------------------- BENCHMARK -------------------- #
print("⚙️ INÍCIO DO BENCHMARK AUTOMÁTICO\n")

for N in N_VETORES:
    print(f"=================== TESTE COM {N} VETORES ===================")

    # Carregar ou gerar embeddings e frases
    path_embeddings = f"embeddings_{N}.npy"
    path_sentencas = f"sentencas_{N}.npy"

    if os.path.exists(path_embeddings):
        embeddings = np.load(path_embeddings)
        sentencas = np.load(path_sentencas, allow_pickle=True)
        print(f"✅ Embeddings carregados de ficheiro")
    else:
        embeddings = gerar_embeddigs(N)
        sentencas = np.array(gerar_frases(N))
        np.save(path_embeddings, embeddings)
        np.save(path_sentencas, sentencas)
        print(f"✅ Embeddings e frases salvos")

    # Garantir binários para todos os níveis usados em MODOS
    niveis_necessarios = sorted(set(n for modo in MODOS for n in modo))
    for nivel in niveis_necessarios:
        garantir_bin(nivel, f"{BASE}_{N}", embeddings)

    # Usar sempre o mesmo vetor de query (último)
    query_vector = embeddings[-1]
    query_text = sentencas[-1]

    for modo in MODOS:
        print(f"\n⚙️ Caminho: {modo}")
        start = time.time()
        resultados = search_embeddings(
            embeddings,
            sentencas,
            query_text=None,
            query_vector=query_vector,
            levels=modo,
            top_k=5,
            base_path=f"{BASE}_{N}"
        )
        total = time.time() - start
        print(f"⏱️ Tempo total: {total:.3f}s")
        print("Top 3 resultados:")
        for i, r in enumerate(resultados[:3], 1):
            print(f"  #{i}: score={r['score']:.6f}, frase='{r['text']}'")

    print("\n" + "="*70 + "\n")
