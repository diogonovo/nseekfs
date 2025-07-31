import numpy as np
from nseekfs import prepare_bin_from_numpy

# Carregar embeddings simulados
embeddings = np.load("embeddings.npy")

# Base e níveis em falta
BASE = "simulado"
MISSING_LEVELS = ["f8", "f64"]

for nivel in MISSING_LEVELS:
    print(f"💾 A gerar binário para {nivel}...")
    path = prepare_bin_from_numpy(embeddings, nivel, BASE)
    print(f"✅ {nivel} criado: {path}")
