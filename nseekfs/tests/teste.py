import numpy as np
from sentence_transformers import SentenceTransformer
from nseekfs import prepare_bin_from_numpy
from nseek_hierarchical_engine import HierarchicalEngine

# === 1) Gerar embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["isto é um teste", "o gato dorme", "o cão ladra", "a IA responde", "o código compila"]
embeddings = model.encode(texts, normalize_embeddings=True)  # np.ndarray (float32)

# === 2) Gerar apenas os binários necessários ===
prepare_bin_from_numpy(embeddings, "f16", "data")
prepare_bin_from_numpy(embeddings, "f32", "data")

# === 3) Criar o motor hierárquico ===
engine = HierarchicalEngine({
    "f16": "data_f16.bin",
    "f32": "data_f32.bin"
})

# === 4) Pesquisa encadeada ===
query = model.encode(["o animal emite som"], normalize_embeddings=True)[0]
results = engine.search(query, path=["f16", "f32"], top_k=3)

# === 5) Mostrar resultados ===
print("Top resultados:")
for rank, (idx, score) in enumerate(results, start=1):
    print(f"#{rank}: idx {idx}, score={score:.4f}, texto='{texts[idx]}'")
