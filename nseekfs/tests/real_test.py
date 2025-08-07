import os
import time
import random
import numpy as np
from itertools import product
from sentence_transformers import SentenceTransformer
from nseekfs import prepare_engine_from_embeddings, PySearchEngine

# ========================
# Parâmetros
# ========================
num_vectors = 100_000
dims = 384
base_path = "realistic_data"
level = "f32"
bin_path = f"{base_path}_{level}_ann.bin"

# ========================
# Gerar frases técnicas realistas únicas
# ========================
topics = ["Rust", "Python", "Machine Learning", "Data Science", "NLP", "Pandas", "NumPy", "Optimization", "GPU", "Vectors"]
actions = [
    "How to use", "Fixing", "Understanding", "Speeding up", "Best way to implement", "Common issues with",
    "Fastest method for", "Avoiding", "Troubleshooting", "Comparing"
]
subjects = [
    "vector operations", "cosine similarity", "binary search", "memory management", "data loading",
    "parallel processing", "ANN search", "embedding generation", "transformers", "csv parsing"
]

# Gerar combinações únicas até atingir num_vectors
all_combinations = list(product(actions, subjects, topics))
random.shuffle(all_combinations)
sentences = [
    f"{a} {s} in {t}"
    for a, s, t in (all_combinations * ((num_vectors // len(all_combinations)) + 1))
][:num_vectors]

# ========================
# Gerar embeddings reais
# ========================
print(f"📦 A gerar {num_vectors} embeddings técnicos...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, batch_size=64, normalize_embeddings=True)
print(f"✅ Embeddings gerados [{time.time() - t0:.2f}s]")

# ========================
# Criar binário com ANN
# ========================
print("\n💾 A criar binário a partir de RAM...")
t0 = time.time()
bin_created = prepare_engine_from_embeddings(
    embeddings.tolist(),
    base_path=base_path,
    precision=level,
    normalize=True,
    ann=True
)
print(f"✅ Binário criado: {bin_created} [{time.time() - t0:.2f}s]")

# ========================
# Carregar engine
# ========================
print("\n⚙️ A carregar engine...")
t0 = time.time()
engine = PySearchEngine(bin_created, ann=True)
print(f"✅ Engine carregada [{time.time() - t0:.2f}s]")

# ========================
# Queries reais
# ========================
queries = [
    "How to optimize cosine similarity search?",
    "Speed up vector operations in Rust",
    "Loading large CSVs with pandas efficiently"
]

for i, q in enumerate(queries, 1):
    print(f"\n🔍 Query {i}: “{q}”")
    q_vec = model.encode([q], normalize_embeddings=True)[0]
    t0 = time.time()
    results = engine.top_k_query(q_vec.tolist(), 5)
    print(f"🕒 Tempo: {time.time() - t0:.4f}s")
    for j, (idx, score) in enumerate(results, 1):
        print(f"  {j}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")
