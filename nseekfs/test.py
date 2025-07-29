import time
from nseekfs import load_vectors, get_cached_vectors, top_k_similar

path = "data2.csv"
input_index = 0
k = 10

print("⏳ a carregar os vetores do ficheiro (usando cache global)...")
start_load = time.time()
load_vectors(path)
vectors = get_cached_vectors()
print(f"✅ carregados {len(vectors)} vetores em {time.time() - start_load:.2f}s\n")

print(f"🎯 vetor de input: índice {input_index}")
print(f"→ primeiros 5 valores: {vectors[input_index][:5]}\n")

print("🔎 a calcular top-10 vetores mais semelhantes...")
start_topk = time.time()
results = top_k_similar(input_index, k)
print(f"✅ top-{k} encontrados em {time.time() - start_topk:.4f}s\n")

print("📊 resultados (índice, similaridade):")
for i, (idx, score) in enumerate(results):
    star = "⭐" if idx == input_index else "  "
    print(f"{star} #{i+1}: índice {idx}, score = {score:.6f}")

print(f"\n⏱ total: {time.time() - start_load:.2f}s")
