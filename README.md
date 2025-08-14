# 🚀 NSeekFS

**High-performance vector similarity search with Rust backend**

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python Support](https://img.shields.io/pypi/pyversions/nseekfs.svg)](https://pypi.org/project/nseekfs/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)](https://pypi.org/project/nseekfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NSeekFS is **NSeek**'s first tool in the vector similarity search ecosystem. Built with Rust for maximum performance and Python for ease of use.

## ✨ **Key Features**

- 🎯 **Production-ready exact search** with 100% recall guarantee
- ⚡ **SIMD-accelerated** operations (2-4x speedup on 64+ dimensions)
- 💾 **Memory-efficient** with multiple precision levels (f32, f16, f8)
- 🔒 **Thread-safe** and **zero-copy** operations
- 📦 **Professional persistence** with binary index format
- 🧪 **Experimental ANN** for research and prototyping

## 🚀 **Quick Start**

### Installation

```bash
pip install nseekfs
```

### Simple Usage

```python
import nseekfs
import numpy as np

# Create sample vectors (e.g., sentence embeddings)
embeddings = np.random.randn(1000, 384).astype(np.float32)

# ✅ Create index (simple API)
index = nseekfs.from_embeddings(embeddings)

# 🔍 Query for similar vectors
query_vector = embeddings[42]
results = index.query(query_vector, top_k=10)

print(f"Found {len(results)} similar vectors")
for i, result in enumerate(results):
    print(f"{i+1}. Vector #{result['idx']} (similarity: {result['score']:.4f})")
```

That's it! Three lines to get production-ready vector search.

## 🔧 **API Reference**

### Primary API (Recommended)

Most users should use these simple functions:

```python
import nseekfs

# Create index from embeddings
index = nseekfs.from_embeddings(
    embeddings,                 # numpy array of vectors
    level="f32",               # precision: "f32", "f16", "f8"
    ann=False,                 # exact search (recommended for production)
    output_dir="./indexes",    # where to save index
    base_name="my_index"       # filename prefix
)

# Load existing index
index = nseekfs.load_index("./indexes/my_index_f32.bin")

# Query the index
results = index.query(
    query_vector,              # your query vector
    top_k=10,                  # number of results
    return_scores=True         # include similarity scores
)
```

### Advanced API (Power Users)

For advanced users who need full control:

```python
# Direct class access with all options
index = nseekfs.VectorSearch.from_embeddings(
    embeddings,
    level="f16",               # half precision for memory savings
    normalized=True,           # normalize vectors
    ann=False,                 # exact search
    output_dir="./prod",
    base_name="production_index",
    force_rebuild=False        # reuse existing if available
)

# Access low-level engine (experts only)
engine = nseekfs.PySearchEngine("index.bin", ann=False)
```

## 📊 **Performance & Memory**

### Exact Search Performance

| Dataset Size | Dimensions | Build Time | Query Time | Memory |
|-------------|------------|------------|------------|---------|
| 10K vectors | 384D | 0.2s | 0.5ms | 15MB |
| 50K vectors | 384D | 0.8s | 0.7ms | 73MB |
| 200K vectors | 384D | 2.1s | 0.9ms | 292MB |
| 500K vectors | 384D | 5.2s | 1.1ms | 730MB |

### Memory Optimization

Choose precision level based on your needs:

```python
# Full precision (baseline)
index_f32 = nseekfs.from_embeddings(embeddings, level="f32")  # 100% memory

# Half precision (recommended for most cases)
index_f16 = nseekfs.from_embeddings(embeddings, level="f16")  # 50% memory, ~0.1% accuracy loss

# Quarter precision (extreme memory savings)
index_f8 = nseekfs.from_embeddings(embeddings, level="f8")    # 25% memory, ~2-5% accuracy loss
```

### SIMD Acceleration

Automatic SIMD acceleration provides 2-4x speedup:
- ✅ **Enabled**: 64+ dimensional vectors
- ⚪ **Disabled**: <64 dimensional vectors (uses scalar operations)

## 🎯 **Use Cases**

### ✅ **Perfect For**

- **Semantic search** in documents, products, or content
- **Recommendation systems** with embedding-based similarity
- **RAG (Retrieval-Augmented Generation)** applications
- **Data deduplication** and clustering
- **Production deployments** requiring 100% recall
- **Memory-constrained environments** with quantization

### 🤔 **Consider Alternatives For**

- **Production ANN at massive scale** → Use [Faiss](https://github.com/facebookresearch/faiss)
- **Graph-based ANN** → Use [HNSW](https://github.com/nmslib/hnswlib)
- **Distributed search** → Use specialized distributed solutions

## 🔧 **Advanced Examples**

### Professional Persistence

```python
import nseekfs
import numpy as np

# Create and save index
embeddings = np.random.randn(10000, 384).astype(np.float32)
index = nseekfs.from_embeddings(
    embeddings,
    level="f32",
    output_dir="./production",
    base_name="product_embeddings"
)

# Later, load the saved index
index = nseekfs.load_index("./production/product_embeddings_f32.bin")
results = index.query(query_vector, top_k=5)
```

### Memory-Optimized Deployment

```python
# Reduce memory usage with minimal accuracy loss
large_embeddings = np.random.randn(100000, 512).astype(np.float32)

# Half precision: 50% memory reduction, ~0.1% accuracy loss
index_f16 = nseekfs.from_embeddings(large_embeddings, level="f16")

# Quarter precision: 75% memory reduction, ~2-5% accuracy loss  
index_f8 = nseekfs.from_embeddings(large_embeddings, level="f8")

print(f"f16 index: {index_f16.dims}D x {index_f16.rows} vectors")
print(f"f8 index: {index_f8.dims}D x {index_f8.rows} vectors")
```

### Concurrent Usage

```python
import threading

# Thread-safe operations
index = nseekfs.from_embeddings(embeddings)

def worker(query_id):
    query = embeddings[query_id]
    results = index.query(query, top_k=5)
    print(f"Query {query_id}: {len(results)} results")

# Multiple threads can safely query simultaneously
threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Context Manager Support

```python
# Automatic resource management
with nseekfs.from_embeddings(embeddings) as index:
    results = index.query(query_vector, top_k=10)
    print(f"Stats: {index.stats}")
```

## 🛠️ **Utilities**

```python
# System health check
health = nseekfs.health_check()
print(f"Status: {health['status']}")
print(f"SIMD support: {health['rust_engine_working']}")

# System information
info = nseekfs.get_system_info()
print(f"Platform: {info['platform']}")
print(f"Python: {info['python_version']}")

# Configuration validation
config = nseekfs.validate_config()
print(f"Max concurrent queries: {config['max_concurrent_queries']}")
```

## 🏢 **NSeek Ecosystem**

NSeek is building a comprehensive toolkit for AI applications:

- **nseekfs**: Vector similarity search (this package) ✅
- **nseekplus**: Advanced analytics (coming soon)
- **nseekgraph**: Graph analysis (coming soon)  
- **nseektext**: Text processing (coming soon)

Each tool follows the same design principles: **simple API**, **high performance**, and **production-ready**.

## 🚨 **Known Limitations**

### ANN Quality (Experimental)
- **Current ANN recall**: ~10-15% recall@10 (experimental quality)
- **Recommendation**: Use `ann=False` for production workloads
- **Alternative**: Use [Faiss](https://github.com/facebookresearch/faiss) for production ANN

### Memory Requirements
- **RAM constraint**: All data must fit in memory
- **Large datasets**: >1M vectors require substantial RAM
- **Future**: Disk-based search planned for v0.3.0

### Platform Support
- **Supported**: Linux x86_64, macOS (Intel/Apple Silicon), Windows 10+ x86_64
- **Not supported**: ARM Linux, 32-bit systems
- **Future**: ARM support planned for v0.3.0

## 🛣️ **Roadmap**

### v0.2.0 - Enhanced ANN (Q2 2025)
- Improve ANN recall to 60-70% recall@10
- GPU acceleration for large datasets
- Hierarchical search with quality/speed tradeoffs

### v0.3.0 - Scalability (Q3 2025)  
- Disk-based search for datasets larger than RAM
- Incremental updates (add/remove vectors)
- ARM Linux platform support

### v1.0.0 - Production ANN (Q4 2025)
- Production-grade ANN competitive with Faiss
- Real-time streaming vector ingestion
- Enterprise monitoring and management

## 📋 **Requirements**

- **Python**: 3.8+ (3.11+ recommended)
- **NumPy**: 1.21.0+
- **Platform**: x86_64 (Intel/AMD 64-bit)
- **Memory**: Varies by dataset size

## 🤝 **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/diogonovo/nseekfs.git
cd nseekfs

# Install development dependencies
pip install maturin pytest numpy

# Build and install in development mode
maturin develop --release

# Run tests
pytest

# Run examples
python examples/basic_usage.py
```

## 📜 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 **Links**

- **PyPI**: https://pypi.org/project/nseekfs/
- **GitHub**: https://github.com/diogonovo/nseekfs
- **Documentation**: https://github.com/diogonovo/nseekfs#readme
- **Issues**: https://github.com/diogonovo/nseekfs/issues
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🙏 **Acknowledgments**

- **Rust ecosystem**: rayon, wide, memmap2, and other excellent crates
- **Python ecosystem**: NumPy, PyO3 for seamless integration
- **Community**: Contributors and users who make this project better

---

**Made with ❤️ by the NSeek team**

*Building the future of AI infrastructure, one tool at a time.*