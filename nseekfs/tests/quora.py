import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nseekfs import prepare_engine_from_embeddings, PySearchEngine

# ========================
# Parâmetros
# ========================
base_path = "quora_data"
level = "f32"
bin_path = f"{base_path}_{level}_ann.bin"
csv_path = "questions.csv"
max_rows = 10_000

# ========================
# Carregar perguntas reais da Quora
# ========================
print(f"📂 A carregar perguntas do ficheiro '{csv_path}'...")
df = pd.read_csv(csv_path)
sentences = pd.concat([df['question1'], df['question2']]).dropna().unique()[:max_rows]
print(f"✅ Perguntas carregadas: {len(sentences)}")

# ========================
# Gerar embeddings
# ========================
print(f"\n📦 A gerar {len(sentences)} embeddings...")
t0 = time.time()
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")  # 768 dimensões

embeddings = model.encode(sentences.tolist(), batch_size=64, normalize_embeddings=True)
print(f"✅ Embeddings prontos [{time.time() - t0:.2f}s]")

# ========================
# Criar binário ANN
# ========================
print("\n💾 A criar binário ANN...")
t0 = time.time()
bin_created = prepare_engine_from_embeddings(
    embeddings.tolist(),
    base_path=base_path,
    precision=level,
    normalize=True,
    ann=False
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
    "How can I improve my writing skills?",
    "What is the best way to learn Python?",
    "How do I lose weight without dieting?"
]

for i, q in enumerate(queries, 1):
    print(f"\n🔍 Query {i}: “{q}”")
    q_vec = model.encode([q], normalize_embeddings=True)[0]
    t0 = time.time()
    results = engine.top_k_query(q_vec.tolist(), 5)
    print(f"🕒 Tempo: {time.time() - t0:.4f}s")
    for j, (idx, score) in enumerate(results, 1):
        print(f"  {j}. idx={idx} | score={score:.4f} | texto={sentences[idx]}")
