import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from nseekfs import prepare_engine_from_embeddings, PySearchEngine

# Configurações
num_vectors = 100_000
dims = 384
base_path = "teste_comparativo"
npy_path = f"{base_path}.npy"
sentences = [f"Frase {i}" for i in range(num_vectors)]

# Gerar e guardar embeddings .npy
print("📦 A gerar embeddings e salvar .npy...")
t0 = time.time()
data = np.random.randn(num_vectors, dims).astype(np.float32)
np.save(npy_path, data)
print(f"✅ Embeddings guardados [{time.time() - t0:.2f}s]")

# Criação dos .bin (com e sem ANN)
bin_ann = f"{base_path}_f32_ann.bin"
bin_exact = f"{base_path}_f32.bin"

print("\n💾 A criar binário com ANN...")
t0 = time.time()
prepare_engine_from_embeddings(data, base_path, "f32", normalize=True, use_ann=True)
print(f"✅ Bin ANN criado [{time.time() - t0:.2f}s]")

print("\n💾 A criar binário EXATO...")
t0 = time.time()
prepare_engine_from_embeddings(data, base_path, "f32", normalize=True, use_ann=False)
print(f"✅ Bin EXATO criado [{time.time() - t0:.2f}s]")

# Carregar engines
print("\n⚙️ A carregar engines...")
t0 = time.time()
engine_ann = PySearchEngine(bin_ann, use_ann=True)
engine_exact = PySearchEngine(bin_exact, use_ann=False)
print(f"✅ Engines carregadas [{time.time() - t0:.2f}s]")

# Query (vetor fora)
query_vector = data[42].tolist()

# Pesquisa com ANN
print("\n🔍 Pesquisa ANN:")
t0 = time.time()
results_ann = engine_ann.top_k_query(query_vector, 5)
elapsed_ann = time.time() - t0
print(f"⏱️ Tempo: {elapsed_ann:.4f}s")
for i, (idx, score) in enumerate(results_ann, 1):
    print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")

# Pesquisa exata
print("\n🔍 Pesquisa EXATA:")
t0 = time.time()
results_exact = engine_exact.top_k_query(query_vector, 5)
elapsed_exact = time.time() - t0
print(f"⏱️ Tempo: {elapsed_exact:.4f}s")
for i, (idx, score) in enumerate(results_exact, 1):
    print(f"  {i}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")
