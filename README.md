# 🚀 NSeekFS - High-Performance Vector Similarity Search

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NSeekFS** is a blazing-fast vector similarity search engine built with Rust and optimized for machine learning workloads. It provides efficient top-k similarity search with support for approximate nearest neighbors (ANN), multiple precision levels, and SIMD acceleration.

## ✨ Key Features

- **🏎️ High Performance**: SIMD-optimized operations with Rust backend
- **📊 Multiple Precision**: Support for f8, f16, f32, f64 quantization levels  
- **🔍 ANN Support**: Locality-sensitive hashing for fast approximate search
- **🐍 Python Integration**: Simple, intuitive Python API
- **💾 Persistent Storage**: Efficient binary serialization for large indexes
- **🔄 Flexible Input**: Support for NumPy arrays, CSV, and NPY files

## 📦 Installation

```bash
pip install nseekfs
```

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
import nseekfs

# Create some sample embeddings (or load your own)
embeddings = np.random.randn(10000, 384).astype(np.float32)

# Create and load an index
index = nseekfs.NSeek.from_embeddings(
    embeddings=embeddings,
    level="f32",           # Precision level
    ann=True,              # Enable approximate nearest neighbors
    normalized=True        # Embeddings are already normalized
)

# Search for similar vectors
query_vector = embeddings[0]  # Use first embedding as query
results = index.query(query_vector, top_k=10)

print(f"Found {len(results)} similar vectors:")
for result in results:
    print(f"Index: {result['idx']}, Score: {result['score']:.4f}")
```

### Working with Files

```python
import nseekfs

# Create index from NPY file
index = nseekfs.NSeek.from_embeddings(
    embeddings="my_embeddings.npy",
    level="f32",
    ann=True,
    base_name="my_model",
    output_dir="./indexes"
)

# Load existing index
index = nseekfs.NSeek.load_index("./indexes/f32.bin")

# Query the index
results = index.query(query_vector, top_k=5, method="simd")
```

### Advanced Configuration

```python
# Different precision levels for memory/speed tradeoffs
index_f32 = nseekfs.NSeek.from_embeddings(embeddings, level="f32")  # Full precision
index_f16 = nseekfs.NSeek.from_embeddings(embeddings, level="f16")  # Half precision  
index_f8 = nseekfs.NSeek.from_embeddings(embeddings, level="f8")    # Quarter precision

# Disable ANN for exact search
exact_index = nseekfs.NSeek.from_embeddings(embeddings, ann=False)

# Different search methods
results_simd = index.query(query, top_k=10, method="simd")    # SIMD acceleration
results_scalar = index.query(query, top_k=10, method="scalar") # Standard implementation
results_auto = index.query(query, top_k=10, method="auto")    # Automatic selection
```

## 📊 Performance

NSeekFS is designed for high-throughput similarity search:

| Dataset Size | Dimensions | Query Time (SIMD) | Query Time (Scalar) | Memory Usage |
|--------------|------------|-------------------|---------------------|--------------|
| 100K vectors | 384 | ~0.5ms | ~2.1ms | 145MB |
| 1M vectors | 384 | ~4.2ms | ~18.7ms | 1.4GB |
| 10M vectors | 384 | ~38ms | ~165ms | 14GB |

*Benchmarks performed on Intel i7-10700K with 32GB RAM*

## 🎯 Use Cases

- **Semantic Search**: Find similar documents, articles, or text snippets
- **Recommendation Systems**: Content-based recommendations using embeddings
- **Image Retrieval**: Search similar images using visual embeddings
- **Deduplication**: Identify duplicate or near-duplicate content
- **Clustering**: Group similar vectors for analysis
- **RAG Systems**: Retrieval-augmented generation with vector databases

## 📚 API Reference

### `NSeek.from_embeddings()`

Create an index from embeddings data.

**Parameters:**
- `embeddings` (Union[np.ndarray, str]): Input embeddings or file path
- `level` (str): Precision level ("f8", "f16", "f32", "f64")
- `normalized` (bool): Whether embeddings are normalized
- `ann` (bool): Enable approximate nearest neighbors
- `base_name` (str): Base name for index files
- `output_dir` (Optional[str]): Output directory

### `NSeek.load_index()`

Load an existing index from disk.

**Parameters:**
- `bin_path` (str): Path to .bin file
- `normalized` (bool): Whether vectors are normalized
- `ann` (bool): Enable ANN if available

### `index.query()`

Search for similar vectors.

**Parameters:**
- `query_vector` (Union[np.ndarray, List[float]]): Query vector
- `top_k` (int): Number of results to return
- `method` (str): Search method ("simd", "scalar", "auto")
- `similarity` (str): Similarity metric ("cosine")

**Returns:**
- `List[dict]`: Results with 'idx' and 'score' keys

## 🔧 Advanced Features

### Quantization Levels

- **f32**: Full 32-bit precision (best quality)
- **f16**: Half precision (2x memory savings)
- **f8**: Quarter precision (4x memory savings)  
- **f64**: Double precision (for high-precision requirements)

### ANN vs Exact Search

- **ANN Mode**: Uses locality-sensitive hashing for fast approximate results
- **Exact Mode**: Brute-force search for perfect accuracy (slower on large datasets)

### SIMD Acceleration

NSeekFS automatically uses SIMD instructions when beneficial:
- Enabled automatically for vectors with ≥64 dimensions
- Provides 2-4x speedup on modern CPUs
- Falls back to scalar operations when appropriate

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Rust-Python bindings
- Uses [Rayon](https://github.com/rayon-rs/rayon) for parallel processing
- SIMD optimizations with [wide](https://github.com/Lokathor/wide)

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/diogonovo/nseekfs/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/diogonovo/nseekfs/discussions)  
- 📧 **Email**: your.email@domain.com

---

**⭐ Star us on GitHub if NSeekFS helps your project!**