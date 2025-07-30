from sentence_transformers import SentenceTransformer
import numpy as np
import random
import string
import time
from nseekfs import prepare_bin_from_numpy
from nseek_hierarchical_engine import HierarchicalEngine

# Configuração
N = 100_000
D = 384
BASE = "hug"
NIVEIS = ["f16", "f32"]

# Modelo HuggingFace
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Gerar frases aleatórias
print("📝 A gerar frases...")
def frase_aleatoria():
    return " ".join(
        ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(random.randint(5, 12))]
    )

sent_start = time.time()
sentencas = [frase_aleatoria() for _ in range(N)]
print(f"✅ {N} frases geradas em {time.time() - sent_start:.2f}s")

# 2. Gerar embeddings
print("⚙️ A gerar embeddings com HuggingFace...")
emb_start = time.time()
embeddings = model.encode(sentencas, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
print(f"✅ Embeddings gerados em {time.time() - emb_start:.2f}s")

# 3. Gerar binários para cada nível
for nivel in NIVEIS:
    print(f"💾 A preparar binário para {nivel}...")
    start = time.time()
    path = prepare_bin_from_numpy(embeddings, nivel, BASE)
    print(f"✅ {nivel} pronto: {path} ({time.time() - start:.2f}s)")

# 4. Criar engine hierárquico
print("🧠 A criar engine hierárquica...")
paths = {nivel: f"{BASE}_{nivel}.bin" for nivel in NIVEIS}
engine = HierarchicalEngine(paths)

# 5. Escolher uma query real e pesquisar
query_text = sentencas[123456]  # qualquer frase do dataset
print(f"🔎 A pesquisar por: \"{query_text}\"")

query_vec = model.encode([query_text], normalize_embeddings=True)[0]
start = time.time()
results = engine.search(query_vec, path=NIVEIS, top_k=5)
print(f"✅ Pesquisa concluída em {time.time() - start:.4f}s")

# 6. Mostrar resultados
for i, (idx, score) in enumerate(results, 1):
    print(f"#{i}: idx={idx}, score={score:.6f}, frase='" + sentencas[idx] + "'")
