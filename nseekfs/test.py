import nseekfs
import time

# carregar o ficheiro CSV
print("⏳ a carregar os vetores do ficheiro...")
start_load = time.perf_counter()
data = nseekfs.py_load_csv_f32("data2.csv")
end_load = time.perf_counter()
print(f"✅ carregados {len(data)} vetores em {end_load - start_load:.2f}s")

# escolher o vetor de input (aqui usámos o primeiro)
input_index = 0
input_vector = data[input_index]
print(f"\n🎯 vetor de input: índice {input_index}")
print(f"→ primeiros 5 valores: {input_vector[:5]}")

# calcular os top-k vetores mais semelhantes
k = 10
print(f"\n🔎 a calcular top-{k} vetores mais semelhantes...")
start_topk = time.perf_counter()
top = nseekfs.py_top_k_similar(input_vector, data, k)
end_topk = time.perf_counter()
print(f"✅ top-{k} encontrados em {end_topk - start_topk:.4f}s")

# mostrar os resultados
print("\n📊 resultados (índice, similaridade):")
for rank, (i, score) in enumerate(top):
    prefix = "⭐ input original →" if i == input_index else " "
    print(f"{prefix} #{rank + 1}: índice {i}, score = {score:.6f}")

# total
print(f"\n⏱ total: {end_topk - start_load:.2f}s")
