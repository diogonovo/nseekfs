import numpy as np
import time
import os
from nseekfs import PySearchEngine, prepare_engine_py

# Parâmetros
num_vectors = 100_000
dims = 384
csv_path = "testdata.csv"
npy_path = "testdata.npy"
bin_path = "testdata.bin"

# Geração fictícia de embeddings
def gerar_dados():
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        data = np.random.randn(num_vectors, dims).astype(np.float32)
        np.savetxt(csv_path, data, delimiter=",")
        np.save(npy_path, data)
        print(f"✅ Gerados {num_vectors} vetores em {dims} dimensões e salvos como .npy e .csv")
    else:
        print(f"✅ Ficheiros .csv e .npy já existem")

# Testa com CSV (via prepare_engine)
def benchmark_csv(use_ann: bool):
    modo = "Aproximado (ANN)" if use_ann else "Exato"
    print(f"\n🔍 Teste de pesquisa com modo: {modo}")
    
    t0 = time.time()
    path = prepare_engine_py(csv_path, normalize=False, force=True, use_ann=use_ann)
    print(f"✅ Engine preparado em: {path} [{time.time() - t0:.2f}s]")
    
    engine = PySearchEngine(path, normalize=False, use_ann=use_ann)
    results = engine.top_k_similar(42, 5)
    
    print(f"🔎 Resultados para índice 42 (top 5):")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. idx={idx} | score={score:.4f}")
    
    print(f"⏱️ Tempo de busca: {time.time() - t0:.4f}s")

# Testa com embeddings diretamente (como HuggingFace)
def benchmark_embeddings():
    print(f"\n🔍 Teste direto com vetor numpy (simulando HuggingFace/OpenAI)")
    
    data = np.load(npy_path)
    t0 = time.time()
    engine = PySearchEngine.from_embeddings(data, normalize=True, use_ann=True)
    print(f"✅ Engine direto carregado em {time.time() - t0:.2f}s")
    
    results = engine.top_k_similar(42, 5)
    
    print(f"🔎 Resultados para índice 42 (top 5):")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. idx={idx} | score={score:.4f}")
    
    print(f"⏱️ Tempo de busca: {time.time() - t0:.4f}s")


if __name__ == "__main__":
    gerar_dados()
    benchmark_csv(use_ann=False)
    benchmark_csv(use_ann=True)
    benchmark_embeddings()
