# 🔎 nseekfs – High-Performance Vector Search Engine

**nseekfs** is a fast, lightweight vector search engine built for semantic retrieval, ranking pipelines, and large-scale AI workloads. It offers a clean and intuitive Python interface backed by a highly optimized native core written in Rust.

> ⚠️ This public release focuses on the vector similarity engine. Advanced modules (hierarchical search, reasoning, graph inference) are under active development and will be released progressively.

---

## 🚀 Features

- ✅ One-line search engine initialization from embeddings
- ✅ Optional Approximate Nearest Neighbor (ANN) indexing
- ✅ Supports quantization levels: `"f8"`, `"f16"`, `"f32"`, `"f64"`
- ✅ Efficient `.bin` index generation and reuse
- ✅ Built with Rust + PyO3 for performance
- 🧩 Ready for integration into custom AI pipelines

---

## 📦 Installation

```bash
pip install nseekfs
```

---

## 🧠 Basic Usage

```python
from nseekfs import NSeek

engine = NSeek.from_embeddings(
    embeddings=my_vectors,     # np.ndarray, List[List[float]], or path to .npy / .csv
    level="f16",               # "f8", "f16", "f32", or "f64"
    use_ann=True,              # enables ANN indexing
    base_dir="nseek_indexes",  # where to store .bin files
    base_name="my_index"
)

results = engine.query(query_vector, top_k=5)

for r in results:
    print(f"{r['score']:.4f} → idx {r['idx']}")
```

---

## 📥 Embeddings Input Options

You can provide your embeddings as:

- ✅ `np.ndarray` (2D)
- ✅ `List[List[float]]`
- ✅ `.npy` or `.csv` file paths

---

## 📤 Query Vector

- Must be a 1D `List[float]` or `np.ndarray`
- Will be automatically normalized
- Must match the same dimension as your embeddings

---

## ⚡ ANN (Approximate Nearest Neighbors)

- Enabled via `use_ann=True` during engine creation
- Uses random hyperplane hashing for fast lookup
- Fully integrated into the `.bin` index

---

## 🧪 Example Index Structure

```
nseek_indexes/
└── my_index/
    └── f16.bin
```

Each file stores the quantized and indexed representation of your vectors.

---

## ⚙️ Parameters

| Parameter     | Description                                  |
|---------------|----------------------------------------------|
| `embeddings`  | Embeddings as array, list or file            |
| `level`       | Quantization: `"f8"`, `"f16"`, `"f32"`, `"f64"` |
| `use_ann`     | Enable/disable ANN indexing                  |
| `base_dir`    | Directory where `.bin` files are stored      |
| `base_name`   | Subdirectory/index name                      |

---

## 📌 Roadmap (Under Development)

- 🌐 Hierarchical search across multiple levels
- 🧠 Graph-based semantic traversal and reasoning
- 🔄 Index updates and append-only formats
- 🧩 Integration with Hugging Face models

> Future versions will expose higher-level reasoning, chaining and graph traversal engines.

---

## 🔒 Privacy & Packaging

- Only the Python interface is exposed
- Core engine is compiled and optimized in Rust (not exposed in wheel)
- No external API dependencies or network access required

---

## 📜 License

MIT License – see [LICENSE](LICENSE)

---

## 🙋 Contact

For professional use, enterprise integration, or questions:

📧 [diogonovo@outlook.pt](mailto:diogonovo@outlook.pt)  
🔗 [github.com/diogonovo/nseekfs](https://github.com/diogonovo/nseekfs)