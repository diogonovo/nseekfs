import time
from nseekfs import PrepareEngine, PySearchEngine

dataset_csv = "data2.csv"

print("=== 1) PREPARAR DATASET (CSV -> BIN) ===")
start = time.time()
bin_path = PrepareEngine(dataset_csv, normalize=True, force=False)
print(f"✅ Binário pronto: {bin_path} ({time.time()-start:.2f}s)\n")

print("=== 2) CARREGAR ENGINE (.bin preferido) ===")
start = time.time()
engine = PySearchEngine(dataset_csv)  # pode ser .csv ou .bin
print(f"✅ Carregado {engine.rows()} vetores em {time.time()-start:.2f}s")
print(f"→ Dimensões: {engine.dims()}\n")

print("=== 3) PESQUISA POR ÍNDICE ===")
input_index = 0
k = 10
start = time.time()
results_idx = engine.top_k_similar(input_index, k)
print(f"✅ top-{k} (index {input_index}) encontrados em {time.time()-start:.4f}s")
for i, (idx, score) in enumerate(results_idx, start=1):
    star = "⭐" if idx == input_index else " "
    print(f"{star} #{i}: índice {idx}, score = {score:.6f}")
print()

print("=== 4) PESQUISA POR VETOR EXTERNO ===")
query_vector = [0.1] * engine.dims()  # vetor fictício
start = time.time()
try:
    results_vec = engine.top_k_query(query_vector, k, normalize=True)
    print(f"✅ top-{k} (vetor externo) encontrados em {time.time()-start:.4f}s")
    for i, (idx, score) in enumerate(results_vec, start=1):
        print(f" #{i}: índice {idx}, score = {score:.6f}")
except Exception as e:
    print("❌ Erro:", e)

print("\n=== TESTE COMPLETO ===")
