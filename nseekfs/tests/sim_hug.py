import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from nseekfs import prepare_engine_from_embeddings, PySearchEngine

# ========================
# Simular engenheiro
# ========================
num_vectors = 500_000
dims = 384
base_path = "engenheiro_data"
level = "f32"
bin_path = f"{base_path}_{level}_ann.bin"
sentences = [f"Frase do engenheiro {i}" for i in range(num_vectors)]

print(f"📦 A gerar {num_vectors:,} embeddings externos (simulados)...")
t0 = time.time()
data = np.random.randn(num_vectors, dims).astype(np.float32)
print(f"✅ Embeddings prontos [{time.time() - t0:.2f}s]")

# ========================
# Criar bin a partir dos embeddings
# ========================
print("\n💾 A criar binário diretamente a partir de RAM...")
t0 = time.time()
path = prepare_engine_from_embeddings(
    data,
    base_path=base_path,
    precision=level,
    normalize=True,
    use_ann=False
)
print(f"✅ Binário guardado em: {path} [{time.time() - t0:.2f}s]")

# ========================
# Criar engine a partir do .bin
# ========================
print("\n⚙️ A carregar engine a partir do binário...")
t0 = time.time()
engine = PySearchEngine(path, use_ann=True)
print(f"✅ Engine carregada do binário [{time.time() - t0:.2f}s]")

# ========================
# Query 1: vetor direto (ex: embeddings recebidos por API)
# ========================
print("\n🔍 Query 1: vetor direto (gerado fora)")
query_vec = np.random.randn(dims).astype(np.float32)
t0 = time.time()
results = engine.top_k_query(query_vec.tolist(), 3)
print(f"🕒 Query direta: {time.time() - t0:.3f}s")
for i, (idx, score) in enumerate(results, 1):
    print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")

# ========================
# Query 2: por índice (ex: base pessoal do engenheiro)
# ========================
print("\n🔍 Query 2: por índice")
t0 = time.time()
results = engine.top_k_similar(123456, 3)
print(f"🕒 Query por índice: {time.time() - t0:.3f}s")
for i, (idx, score) in enumerate(results, 1):
    print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")

# ========================
# Query 3: por texto (via HuggingFace)
# ========================
print("\n🔍 Query 3: por texto (usando HuggingFace)")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "Sugestão técnica avançada para modelos de vetores"
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]
    t0 = time.time()
    results = engine.top_k_query(query_vec.tolist(), 3)
    print(f"🕒 Query por texto: {time.time() - t0:.3f}s")
    for i, (idx, score) in enumerate(results, 1):
        print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")
except ImportError:
    print("⚠️ sentence-transformers não instalado")
