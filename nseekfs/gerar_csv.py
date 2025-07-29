import numpy as np

N = 100_000  # número de vetores
D = 768        # dimensão dos vetores
FICHEIRO = "data2.csv"

print("a gerar os dados...")
data = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)

print("a guardar para CSV...")
np.savetxt(FICHEIRO, data, delimiter=",", fmt="%.6f")

print("ficheiro gerado:", FICHEIRO)
