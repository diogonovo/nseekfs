import os
import numpy as np
import time
from nseek_highlevel import search_embeddings
from nseekfs import PySearchEngine

# ========================
# Configuração
# ========================
num_vectors = 100_000
dims = 384
base_path = "testdata"
csv_path = f"{base_path}.csv"
npy_path = f"{base_path}.npy"
sentences = [f"Frase {i}" for i in range(num_vectors)]
levels = ["f8", "f16", "f32"]

# ========================
# Limpeza do ambiente
# ========================
for ext in ["csv", "npy"] + [f"{lvl}.bin" for lvl in levels]:
    f = f"{base_path}_{ext}" if ext.endswith(".bin") else f"{base_path}.{ext}"
    if os.path.exists(f):
        os.remove(f)
        print(f"🧹 Apagado: {f}")

# ========================
# Geração de dados em RAM
# ========================
print("\n📦 Gerando embeddings em RAM...")
t0 = time.time()
data = np.random.randn(num_vectors, dims).astype(np.float32)
print(f"✅ Gerados {num_vectors} vetores [{time.time() - t0:.3f}s]")

# ========================
# Teste ANN com CSV
# ========================
print("\n🧪 Teste ANN: Input via CSV + criação de .bin com use_ann=True")
np.savetxt(csv_path, data, delimiter=",")
t0 = time.time()
results = search_embeddings(
    embeddings=data,
    sentences=sentences,
    query_vector=data[42],
    levels=levels,
    top_k=5,
    base_path=base_path,
    disable_ann=False,  # ANN ATIVADO
)
print(f"⏱️ Tempo total CSV (ANN): {time.time() - t0:.3f}s")
print("🔎 Resultados:")
for i, r in enumerate(results, 1):
    print(f"  {i}. idx={r['idx']} | score={r['score']:.4f} | texto={r['text']}")
