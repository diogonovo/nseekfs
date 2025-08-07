import numpy as np
import time
from nseekfs import NSeek

# 📏 Parâmetros
NUM_VECTORS = 300_000
VECTOR_DIM = 768
LEVEL = "f32"
USE_ANN = True
BASE_NAME = "benchmark_norm"

print("🚀 A gerar embeddings normalizados (300k x 768)...")
start = time.time()
embeddings = np.random.rand(NUM_VECTORS, VECTOR_DIM).astype(np.float32)

# ⚠️ Normalização manual — já vêm normalizados
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"✅ Embeddings normalizados em {time.time() - start:.2f}s")

# 🧠 Criação do motor com embeddings já normalizados
print("⚙️ A criar motor com 300k vetores normalizados...")
start = time.time()
engine = NSeek.from_embeddings(
    embeddings=embeddings,
    level=LEVEL,
    normalized=True,        # 👉 Já vêm normalizados → não vai duplicar
    ann=USE_ANN,
    base_name=BASE_NAME
)
print(f"✅ Motor criado em {time.time() - start:.2f}s")

# 🔍 Query de exemplo — já normalizada
query = embeddings[0]  # Já está normalizada, não há mais nada a fazer

print("🔍 A executar query (top_k=10)...")
start = time.time()
results = engine.query(query, top_k=10, method="simd", similarity="cosine")
elapsed = time.time() - start
print(f"✅ Query concluída em {elapsed:.6f}s")

# 📢 Mostrar resultados
print("\n🔎 Top 10 resultados:")
for i, r in enumerate(results, 1):
    print(f"{i}. idx={r['idx']}  score={r['score']:.4f}")
