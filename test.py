import numpy as np
from nseekfs import NSeek

# Gerar 100.000 vetores de 128 dimensões
print("🔧 Gerar embeddings...")
embeddings = np.random.rand(100_000, 128).astype(np.float32)

# Criar o motor com ANN ativado
print("⚙️ Criar motor...")
engine = NSeek.from_embeddings(
    embeddings=embeddings,
    level="f32",
    use_ann=True,
    base_dir="nseek_test",
    base_name="bench"
)


# Escolher uma query
query = embeddings[0]

# Fazer pesquisa
print("🔍 Pesquisar top 5...")
results = engine.query(query, top_k=5)

# Mostrar resultados
for r in results:
    print(f"idx={r['idx']} score={r['score']:.4f}")
