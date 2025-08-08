import numpy as np
import time
from nseekfs import NSeek

# 📏 Parâmetros
NUM_VECTORS = 300_000
VECTOR_DIM = 768
LEVEL = "f32"
USE_ANN = True
BASE_NAME = "benchmark_norm"
OUTPUT_DIR = "nseek_temp"

print(f"🚀 A gerar embeddings normalizados ({NUM_VECTORS:,} x {VECTOR_DIM})...")
start = time.time()

# ⚠️ Normalização manual — os vetores já são unitários
embeddings = np.random.rand(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"✅ Embeddings normalizados em {time.time() - start:.2f}s")

# 🧠 Criação do motor com embeddings já normalizados
print(f"⚙️ A criar motor com {NUM_VECTORS:,} vetores normalizados...")
start = time.time()

engine = NSeek.from_embeddings(
    embeddings=embeddings,        # ✅ agora aceita diretamente numpy.ndarray
    level=LEVEL,
    normalized=True,              # ✅ não normalizar novamente no Rust
    ann=USE_ANN,
    base_name=BASE_NAME,
    output_dir=OUTPUT_DIR         # ✅ bin será guardado aqui
)

print(f"✅ Motor criado em {time.time() - start:.2f}s")

# 🔍 Query de exemplo — vetor real
query = embeddings[0]

print("🔍 A executar query (top_k=10)...")
start = time.time()

results = engine.query(query, top_k=10, method="simd", similarity="cosine")
elapsed = time.time() - start

print(f"✅ Query concluída em {elapsed:.6f}s")

# 📢 Mostrar resultados
print("\n🔎 Top 10 resultados:")
for i, r in enumerate(results, 1):
    print(f"{i}. idx={r['idx']:<7} score={r['score']:.4f}")
