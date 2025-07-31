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
bin_path = f"{base_path}_f32.bin"
sentences = [f"Frase {i}" for i in range(num_vectors)]

# ========================
# Limpeza do ambiente
# ========================
for f in [csv_path, npy_path, bin_path]:
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
# Teste 1: via CSV
# ========================
print("\n🧪 Teste 1: Input via CSV + criação de .bin")
np.savetxt(csv_path, data, delimiter=",")
t0 = time.time()
results = search_embeddings(
    embeddings=data,
    sentences=sentences,
    query_vector=data[42],
    levels=["f32"],
    top_k=5,
    base_path=base_path,
    disable_ann=False,
)
print(f"⏱️ Tempo total CSV: {time.time() - t0:.3f}s")
print("🔎 Resultados:")
for i, r in enumerate(results, 1):
    print(f"  {i}. idx={r['idx']} | score={r['score']:.4f} | texto={r['text']}")

# ========================
# Teste 2: via NPY (sem CSV)
# ========================
print("\n🧪 Teste 2: Input via NPY (sem .csv, mas com .bin existente)")
os.remove(csv_path)
np.save(npy_path, data)
t0 = time.time()
results = search_embeddings(
    embeddings=data,
    sentences=sentences,
    query_vector=data[42],
    levels=["f32"],
    top_k=5,
    base_path=base_path,
    disable_ann=False,
)
print(f"⏱️ Tempo total NPY: {time.time() - t0:.3f}s")
print("🔎 Resultados:")
for i, r in enumerate(results, 1):
    print(f"  {i}. idx={r['idx']} | score={r['score']:.4f} | texto={r['text']}")

# ========================
# Teste 3: Apenas em RAM
# ========================
print("\n🧪 Teste 3: Apenas RAM com PySearchEngine.from_embeddings()")
t0 = time.time()
engine = PySearchEngine.from_embeddings(data, normalize=True, use_ann=False)
results = engine.top_k_query(data[42].tolist(), 5)
print(f"⏱️ Tempo total RAM: {time.time() - t0:.3f}s")
print("🔎 Resultados:")
for i, (idx, score) in enumerate(results, 1):
    print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")

# ========================
# Teste 4: Texto como query
# ========================
print("\n🧪 Teste 4: Texto como query (se modelo disponível)")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "Frase semelhante à número 42"
    t0 = time.time()
    results = search_embeddings(
        embeddings=data,
        sentences=sentences,
        query_text=query_text,
        levels=["f32"],
        top_k=5,
        base_path=base_path,
        disable_ann=False,
    )
    print(f"⏱️ Tempo total texto: {time.time() - t0:.3f}s")
    print("🔎 Resultados:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. idx={r['idx']} | score={r['score']:.4f} | texto={r['text']}")
except ImportError:
    print("⚠️ sentence-transformers não instalado: ignorando Teste 4")
