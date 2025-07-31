import os
import numpy as np
import time
from nseek_highlevel import search_embeddings

# Configuração
N = 100_000
D = 384
BASE = "simulado"
LEVELS = ["f16", "f32"]
TOP_K = 5

# 1. Gerar frases simuladas
start_sent = time.time()
if os.path.exists("sentencas.npy"):
    sentencas = np.load("sentencas.npy", allow_pickle=True).tolist()
else:
    from random import choices, randint
    import string
    def frase_aleatoria():
        return " ".join(
            ["".join(choices(string.ascii_lowercase, k=randint(3, 8))) for _ in range(randint(5, 12))]
        )
    sentencas = [frase_aleatoria() for _ in range(N)]
    np.save("sentencas.npy", sentencas)
print(f"📝 Frases carregadas em {time.time() - start_sent:.2f}s")

# 2. Gerar embeddings aleatórios e normalizados
start_emb = time.time()
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")
else:
    print("⚙️ A gerar embeddings simulados...")
    embeddings = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save("embeddings.npy", embeddings)
print(f"📦 Embeddings prontos em {time.time() - start_emb:.2f}s")

# 3. Escolher query como vetor diretamente
query_idx = 1234
query_vector = embeddings[query_idx]
query_text = sentencas[query_idx]

print(f"🔎 A pesquisar por vetor da frase #{query_idx}: \"{query_text}\"")

# 4. Pesquisa
start_search = time.time()
resultados = search_embeddings(
    embeddings=embeddings,
    sentences=sentencas,
    query_text=None,
    query_vector=query_vector,
    levels=LEVELS,
    top_k=TOP_K,
    base_path=BASE
)
print(f"⏱️ Pesquisa total em {time.time() - start_search:.2f}s\n")

# 5. Mostrar resultados
for i, r in enumerate(resultados, 1):
    print(f"#{i}: score={r['score']:.6f}, frase='{r['text']}'")
