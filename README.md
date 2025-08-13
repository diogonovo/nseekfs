# 🚀 NSeekFS - High-Performance Vector Similarity Search

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/diogonovo/nseekfs/workflows/Tests/badge.svg)](https://github.com/diogonovo/nseekfs/actions)

**NSeekFS** is a high-performance vector similarity search engine with a **professional-grade Rust backend** and intuitive Python API. Built for production workloads requiring fast exact search with optional experimental ANN capabilities.

## ✨ **Why NSeekFS?**

- **🏆 Production-Ready Exact Search**: Enterprise-grade SIMD-optimized exact similarity search
- **⚡ True SIMD Performance**: Hardware-accelerated operations with `wide` crate (2-4x speedup)
- **🛡️ Memory Safety**: Zero-copy operations, memory-mapped files, thread-safe design
- **🔬 Experimental ANN**: Multi-layer LSH with 64-bit precision and smart fallbacks
- **📊 Multiple Precision**: f8/f16/f32/f64 quantization with minimal accuracy loss
- **🐍 Pythonic API**: Simple one-liner interface with full type safety

## 🎯 **Use Cases & Positioning**

### ✅ **Excellent For:**
- **Exact similarity search** on medium datasets (<1M vectors)
- **SIMD-optimized performance** critical applications  
- **Prototyping and development** with real ML embeddings
- **Memory-efficient** deployments with quantization
- **Production systems** requiring 100% recall

### 🧪 **Experimental:**
- **Large-scale ANN** (>1M vectors) - use Faiss for production ANN workloads
- **Disk-based search** - currently RAM-only

## 📦 Installation

```bash
pip install nseekfs
```

**Requirements:** Python 3.8+, NumPy
**Platforms:** Linux x86_64, macOS (Intel/Apple Silicon), Windows 10+ x86_64

## 🚀 Quick Start

```python
import numpy as np
import nseekfs

# Your embeddings (e.g., from sentence-transformers)
embeddings = np.random.randn(50000, 384).astype(np.float32)

# Fast exact search (recommended for production)
index = nseekfs.from_embeddings(embeddings, ann=False)

# Query similar vectors
query_vector = embeddings[0]
results = index.query(query_vector, top_k=10)

print(f"Found {len(results)} similar vectors:")
for result in results:
    print(f"Index: {result['idx']}, Score: {result['score']:.4f}")
```

## ⚡ **Performance Benchmarks**

### **Exact Search Performance** (Production-Ready)
| Dataset Size | Dimensions | Build Time | Query Time | Memory Usage |
|--------------|------------|------------|------------|--------------|
| 10K vectors  | 384       | 0.15s      | **0.4ms**  | 15MB        |
| 50K vectors  | 384       | 0.8s       | **0.7ms**  | 73MB        |
| 200K vectors | 384       | 2.1s       | **0.9ms**  | 292MB       |
| 500K vectors | 384       | 5.2s       | **1.1ms**  | 730MB       |

### **ANN Performance** (Experimental)
| Dataset Size | Build Time | Query Time | Self-Hit Rate | Recall@10 |
|--------------|------------|------------|---------------|-----------|
| 10K vectors  | 0.3s       | 0.6ms      | 100%          | ~15%      |
| 50K vectors  | 1.4s       | 0.8ms      | 95%           | ~12%      |
| 200K vectors | 4.1s       | 0.6ms      | 90%           | ~10%      |

*Benchmarks on Intel i7-10700K, 32GB RAM*

## 🏗️ **Technical Architecture**

### **Rust Backend Excellence**
```rust
// Real SIMD implementation
pub fn compute_score_simd(query: &[f32], vec: &[f32]) -> f32 {
    let chunks = query.len() / LANES;
    let mut simd_sum = f32x8::splat(0.0);
    
    for i in 0..chunks {
        let q = f32x8::new(query[i * LANES..i * LANES + 8]);
        let v = f32x8::new(vec[i * LANES..i * LANES + 8]);
        simd_sum += q * v;
    }
    // Remainder handling + final reduction
}
```

### **Multi-Layer ANN Architecture**
- **16-bit Legacy Table**: Backward compatibility
- **64-bit Main Table**: High-resolution primary hash table  
- **32-bit Multi-Tables**: Diversity through multiple projections
- **Smart Multi-Probe**: Hamming distance exploration with limits
- **SafeBucket System**: Memory explosion prevention

### **Adaptive Configuration**
```python
# Automatic scaling based on dataset size
<1K vectors:    8 bits,  2 tables
1K-10K:        12 bits,  3 tables  
10K-100K:      16 bits,  4 tables
100K-1M:       20 bits,  5 tables
>1M:           24 bits,  6 tables
```

## 🛠️ **Advanced Usage**

### **Production Exact Search**
```python
# Optimized for production workloads
index = nseekfs.from_embeddings(
    embeddings,
    ann=False,              # Exact search
    level="f32",            # Full precision
    normalized=True,        # Pre-normalized embeddings
    method="auto"           # Auto SIMD detection
)

# Batch queries with SIMD acceleration
for query in queries:
    results = index.query(query, top_k=10)
```

### **Memory-Optimized Deployment**
```python
# Reduce memory usage with quantization
index_f16 = nseekfs.from_embeddings(
    embeddings, 
    level="f16"             # 50% memory reduction, ~0.1% accuracy loss
)

index_f8 = nseekfs.from_embeddings(
    embeddings, 
    level="f8"              # 75% memory reduction, ~2-5% accuracy loss
)
```

### **Experimental ANN (for Testing)**
```python
# For research and prototyping only
index_ann = nseekfs.from_embeddings(
    embeddings,
    ann=True,               # Enable experimental ANN
    level="f32"
)

# Note: Use Faiss for production ANN workloads
```

### **Persistent Storage**
```python
# Professional binary format with versioning
index = nseekfs.from_embeddings(
    embeddings,
    output_dir="./indexes",
    base_name="production_v1"
)

# Load with memory mapping
index = nseekfs.load_index("./indexes/production_v1_f32.bin")
```

## 🔧 **API Reference**

### **Core Functions**
```python
# Quick index creation
nseekfs.from_embeddings(embeddings, ann=False, level="f32", **kwargs)

# Load existing index  
nseekfs.load_index(path, ann=False)

# Create index file without loading
nseekfs.create_index(embeddings, output_dir, **kwargs)
```

### **Query Parameters**
```python
results = index.query(
    query_vector,           # np.ndarray
    top_k=10,              # Number of results
    method="auto",         # "auto" | "simd" | "scalar"
    similarity="cosine"    # "cosine" | "euclidean" | "dot_product"
)
```

### **Index Properties**
```python
print(f"Dimensions: {index.dims}")
print(f"Vectors: {index.rows}")
print(f"Precision: {index.level}")
print(f"Normalized: {index.normalized}")
```

## 🚨 **Current Limitations & Honest Assessment**

### **ANN Quality (Experimental Status)**
- **Current Recall**: ~10-15% recall@10 (honest assessment)
- **Best For**: Research, prototyping, similarity exploration
- **Production Recommendation**: Use exact search or Faiss for ANN
- **Roadmap**: Significant ANN improvements planned for v0.2.0

### **Memory Requirements**
- **Current**: All data must fit in RAM
- **f32 Usage**: ~4 bytes per dimension per vector  
- **1M x 384D**: ~1.4GB RAM required
- **Future**: Disk-based search planned for v0.3.0

### **Platform Support**
- ✅ **Fully Supported**: Linux x86_64, macOS (Intel/Apple Silicon), Windows 10+ x86_64
- ❌ **Not Yet**: ARM Linux, 32-bit systems
- 🔮 **Planned**: ARM support in future releases

## 🎯 **When to Use NSeekFS**

### ✅ **Choose NSeekFS When:**
- You need **fast exact search** with guaranteed 100% recall
- **SIMD performance** is critical for your application
- Working with **medium datasets** (<1M vectors)
- You want **memory-efficient** quantization options
- **Professional binary format** and persistence is important
- You need a **production-ready** Rust backend

### 🤔 **Consider Alternatives When:**
- You need **production ANN** at scale → Use [Faiss](https://github.com/facebookresearch/faiss)
- You have **>10M vectors** → Use distributed solutions
- You need **graph-based ANN** → Use [HNSW](https://github.com/nmslib/hnswlib)
- You're on **ARM Linux** → Currently unsupported

## 🛣️ **Roadmap**

### **v0.2.0 - Enhanced ANN** (Q2 2025)
- 🎯 **Target**: 60-70% recall@10 with hierarchical search
- ⚡ **GPU acceleration** for large datasets
- 📊 **Batch query operations**
- 🔧 **Advanced quantization** methods

### **v0.3.0 - Scalability** (Q3 2025)  
- 💾 **Disk-based search** for massive datasets
- 🔄 **Incremental index updates**
- 🌐 **Distributed search** capabilities
- 🏗️ **ARM Linux support**

### **Future Releases**
- 🔄 **Real-time streaming** updates
- 🕸️ **Graph-based ANN** algorithms
- 🤖 **ML framework** integrations (PyTorch, TensorFlow)
- 🌐 **WebAssembly** support

## 🤝 **Contributing**

We welcome contributions! The codebase is well-structured with:
- **Comprehensive Rust backend** with excellent test coverage
- **Professional error handling** and memory safety
- **Clear separation** between exact and ANN algorithms
- **Extensive benchmarking** and validation

See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **[PyO3](https://pyo3.rs/)** for seamless Rust-Python integration
- **[Rayon](https://github.com/rayon-rs/rayon)** for parallel processing excellence
- **[wide](https://github.com/Lokathor/wide)** for SIMD optimization
- Inspired by **[Faiss](https://github.com/facebookresearch/faiss)** architecture patterns

---

**💡 TL;DR**: Professional exact search engine with experimental ANN. Use for production exact search, prototype with ANN, graduate to Faiss for production ANN at scale.

**⭐ Star us on GitHub if NSeekFS helps your project!**