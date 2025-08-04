# 🔎 nseekfs – Fast Hierarchical Vector Search Engine with ANN

`nseekfs` is a high-performance vector search engine with optional Approximate Nearest Neighbor (ANN) indexing. Designed for semantic retrieval, hybrid AI pipelines, and large-scale search, it combines Rust speed with a clean Python interface.

> ⚠️ This public release includes selected components only. Internal modules for advanced reasoning, graph traversal, or compression are withheld.

---

## 🚀 Features

- ✅ Hierarchical search across quantization levels (`f8`, `f16`, `f32`, `f64`)
- ✅ Optional Approximate Nearest Neighbor (LSH)
- ✅ PyO3 bindings – native Python interface
- ✅ Memory-mapped `.bin` index format
- ✅ Parallelism with Rayon + fast float parsing
- 🧩 Designed for plug-in use in larger AI systems

---

## 📦 Installation

> ℹ️ Once published, install via:

```bash
pip install nseekfs
```

For local builds:

```bash
pip install maturin
maturin develop
```

---

## 🔍 Quick Start

```python
from nseek_highlevel import search_embeddings

# Search using vector (example with manual vector)
query_vector = [0.12, -0.07, 0.33, ...]  # vector of the same dimension as the embeddings

results = search_embeddings(
    embeddings=my_embeddings,           # np.ndarray or list of vectors
    sentences=my_sentences,             # list of corresponding texts
    query_vector=query_vector,          # vector to query 
    levels=["f16", "f32"],              # levels to search through
    top_k=5
)

for r in results:
    print(f"{r['score']:.4f} → {r['text']}")
```

---

## 🧠 Use Binary Index Files (Hierarchical)

```python
from nseek_hierarchical_engine import HierarchicalEngine

engine = HierarchicalEngine({
    "f16": "vectors_f16.bin",
    "f32": "vectors_f32.bin"
})

results = engine.search(query_vector, top_k=5)
```

---

## 🐍 Python API (PyO3)

```python
from nseekfs import PySearchEngine, prepare_engine_from_embeddings

engine = PySearchEngine.from_embeddings(embeddings, normalize=True, use_ann=True)
results = engine.top_k_query(query_vector.tolist(), k=5)

# or load from binary
engine = PySearchEngine("vectors_f16.bin", normalize=False, use_ann=True)
```

---

## ⚡ ANN (LSH with Random Hyperplanes)

- Enabled by default via `use_ann=True`
- Internally uses random hyperplanes to hash vectors
- Supports `bits`, `seed` customization via `from_embeddings_custom`

---

## 💾 Binary Index Creation

In Rust:

```rust
use nseekfs::io::write_bin_file;

let bin_path = write_bin_file("my_vectors.csv", true, false, true)?;
```

Or in Python:

```python
prepare_engine_from_embeddings(embeddings, "my_vectors", "f16", normalize=True, use_ann=True)
```

---

## 🧪 Benchmarks

| Dataset Size | Search Path            | Avg Time |
|--------------|------------------------|----------|
| 10,000       | `["f32"]`              | ~0.04s   |
| 100,000      | `["f16", "f32"]`       | ~0.12s   |
| 500,000      | `["f8", "f16", "f32"]` | ~0.21s   |

---

## 📂 Project Structure

```
nseekfs/
├── engine.rs                 # Core vector search engine
├── ann.rs                    # ANN via LSH
├── io.rs                     # Binary I/O helpers
├── utils.rs                  # Cosine, CSV, normalize
├── lib.rs                    # Python bindings (PyO3)
├── nseek_highlevel.py        # One-line entrypoint
├── nseek_hierarchical_engine.py # Level chaining
```

---

## 🔒 Privacy & Packaging

This package exposes only the compiled Rust library via PyO3. Source code is excluded from the distributed wheel.

> You can use `nseekfs` via `pip` but cannot view or modify the internal Rust logic.

---

## 📜 License

MIT License – see [LICENSE](LICENSE)

This is a partial release. For advanced modules or enterprise licensing, please contact the author.

---

## 🙋‍♂️ Contact

For questions, support or commercial use, contact [Diogo Novo](mailto:diogonovo@outlook.pt)