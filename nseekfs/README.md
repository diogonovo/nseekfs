python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop

git add .
git commit -m "melhoria: paralelização do top_k_similar"
git push

git status       # mostra ficheiros alterados
git diff         # mostra diferenças linha a linha

python -m venv .venv
.venv\Scripts\activate
pip install maturin
maturin develop

pip install numpy
