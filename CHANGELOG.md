# Changelog

All notable changes to NSeekFS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-13

### 🎉 **Initial Release - Production-Ready Exact Search**

The first public release of NSeekFS featuring **enterprise-grade exact search** with experimental ANN capabilities.

---

## ✨ **Added Features**

### 🏆 **Production-Ready Core**

#### **High-Performance Exact Search**
- **SIMD-Optimized Operations**: True vectorized computations using `wide` crate
  - Automatic f32x8 SIMD for 64+ dimensional vectors
  - 2-4x performance improvement on modern CPUs
  - Graceful fallback to scalar operations
- **Memory-Mapped Storage**: Zero-copy operations with professional binary format
- **Thread-Safe Design**: Parallel processing with Rayon, safe for concurrent access
- **Comprehensive Validation**: Input sanitization, bounds checking, NaN detection

#### **Advanced Quantization System**
- **f64**: Double precision for research and high-accuracy requirements
- **f32**: Full precision baseline (recommended for production)
- **f16**: Half precision with 2x memory savings and ~0.1% accuracy loss
- **f8**: Quarter precision with 4x memory savings and ~2-5% accuracy loss
- **Smart Precision Selection**: Automatic precision recommendations based on use case

### 🧪 **Experimental ANN Engine**

#### **Multi-Layer LSH Architecture**
```
Layer 1: 16-bit compatibility table (backward compatibility)
Layer 2: 64-bit main table (high-resolution primary hashing)  
Layer 3: 32-bit multi-tables (diversity through multiple projections)
```

#### **Adaptive Configuration System**
- **Smart Scaling**: Hash bits and table count adapt to dataset size
  - <1K vectors: 8 bits, 2 tables
  - 1K-10K: 12 bits, 3 tables  
  - 10K-100K: 16 bits, 4 tables
  - 100K-1M: 20 bits, 5 tables
  - >1M: 24 bits, 6 tables
- **SafeBucket System**: Memory explosion prevention with configurable limits
- **Multi-Probe Search**: Hamming distance exploration with intelligent bounds

#### **Current ANN Performance (Honest Assessment)**
- **Self-Hit Rate**: 90-100% (finds query vector itself)
- **Recall@10**: ~10-15% (experimental quality)
- **Speed**: 0.6-0.8ms average query time
- **Recommendation**: Use exact search for production, ANN for research/prototyping

### 🐍 **Python API Excellence**

#### **Simple & Pythonic Interface**
```python
# One-liner for most use cases
index = nseekfs.from_embeddings(embeddings)

# Full control when needed
index = nseekfs.NSeek.from_embeddings(
    embeddings,
    ann=False,              # Production exact search
    level="f32",            # Full precision
    normalized=True,        # Pre-normalized data
    output_dir="./indexes"  # Persistent storage
)
```

#### **Comprehensive API Design**
- **Multiple Import Styles**: Flexible usage patterns
- **Full Type Hints**: IDE support and runtime validation  
- **Rich Error Messages**: Detailed exceptions with context
- **NumPy Integration**: Zero-copy operations where possible

### 🔧 **Technical Implementation**

#### **Rust Backend Architecture**
- **Memory Safety**: Zero buffer overflows, automatic resource cleanup
- **Cross-Platform**: Native compilation for Linux, macOS, Windows
- **Professional Logging**: Structured logging with configurable levels
- **Extensive Testing**: Comprehensive test suite with edge case coverage

#### **File Format & Persistence**
- **Binary Format v4**: Professional serialization with magic numbers
- **Version Detection**: Automatic migration and compatibility checking
- **Memory Mapping**: Efficient loading of large indexes
- **Backward Compatibility**: Support for previous format versions

#### **Performance Optimizations**
- **Parallel Construction**: Multi-threaded index building with Rayon
- **SIMD Query Processing**: Vectorized similarity computations
- **Memory Pool Management**: Efficient allocation patterns
- **Cache-Friendly Design**: Optimized data layouts for modern CPUs

---

## 📊 **Performance Benchmarks**

### **Exact Search Performance** (Production Ready)
- **10K x 384D**: 0.15s build, 0.4ms query, 15MB memory
- **50K x 384D**: 0.8s build, 0.7ms query, 73MB memory  
- **200K x 384D**: 2.1s build, 0.9ms query, 292MB memory
- **500K x 384D**: 5.2s build, 1.1ms query, 730MB memory

### **ANN Performance** (Experimental)
- **Build Speed**: 1.5-2x slower than exact (hash table construction)
- **Query Speed**: 0.6-0.8ms (similar to exact due to candidate filtering)
- **Memory Usage**: 1.2-1.5x exact search (hash table overhead)
- **Recall Quality**: 10-15% recall@10 (current limitation)

### **SIMD Performance Gains**
- **64+ dimensions**: 2-4x speedup vs scalar operations
- **<64 dimensions**: Automatic scalar fallback
- **Platform Support**: SSE2, AVX2 detection and usage

---

## 🎯 **API Examples**

### **Production Exact Search**
```python
import nseekfs
import numpy as np

# Production-ready exact search
embeddings = load_embeddings()  # Your data
index = nseekfs.from_embeddings(
    embeddings, 
    ann=False,              # Exact search
    level="f32",            # Full precision
    output_dir="./prod"     # Persistent storage
)

# Fast similarity search with 100% recall
results = index.query(query_vector, top_k=10)
```

### **Memory-Optimized Deployment**
```python
# Reduce memory usage with minimal accuracy loss
index_f16 = nseekfs.from_embeddings(
    embeddings,
    level="f16",            # 50% memory reduction
    ann=False               # Still exact search
)

# Extreme memory savings for large-scale deployment
index_f8 = nseekfs.from_embeddings(
    embeddings,
    level="f8"              # 75% memory reduction
)
```

### **Experimental ANN Testing**
```python
# For research and development only
index_ann = nseekfs.from_embeddings(
    embeddings,
    ann=True,               # Experimental ANN
    level="f32"
)

# Note: Use Faiss for production ANN workloads
results = index_ann.query(query, top_k=50)  # May need higher k
```

---

## 🚨 **Known Limitations (Honest Assessment)**

### **ANN Quality (Experimental Status)**
- **Current State**: Research/prototyping quality
- **Recall Performance**: 10-15% recall@10 (needs improvement)
- **Production Readiness**: Not recommended for production ANN
- **Alternative**: Use Faiss for production ANN requirements
- **Improvement Plan**: Significant enhancements planned for v0.2.0

### **Memory & Scalability**
- **RAM Requirement**: All data must fit in memory
- **Large Datasets**: >1M vectors require substantial RAM
- **No Disk Overflow**: Disk-based search planned for v0.3.0
- **Scaling Guidance**: Consider distributed solutions for >10M vectors

### **Platform Support**
- **Fully Supported**: Linux x86_64, macOS (Intel/Apple Silicon), Windows 10+ x86_64
- **Not Supported**: ARM Linux, 32-bit systems, older Windows versions
- **Future Plans**: ARM support planned for v0.3.0

### **Feature Gaps**
- **No Incremental Updates**: Requires full rebuild for new data
- **No GPU Acceleration**: CPU-only implementation
- **Limited ANN Algorithms**: Only LSH-based approach

---

## 🛣️ **Roadmap & Future Development**

### **v0.2.0 - Enhanced ANN** (Q2 2025)
- **🎯 Primary Goal**: Improve ANN recall to 60-70% recall@10
- **📊 Hierarchical Search**: Multi-resolution ANN with quality/speed tradeoffs
- **⚡ GPU Acceleration**: CUDA support for large datasets
- **📈 Batch Operations**: Multi-query processing optimizations
- **🔧 Advanced Quantization**: Product quantization and learned indices

### **v0.3.0 - Scalability** (Q3 2025)
- **💾 Disk-Based Search**: Support for datasets larger than RAM
- **🔄 Incremental Updates**: Add/remove vectors without full rebuild
- **🌐 Distributed Search**: Multi-node clustering capabilities
- **🏗️ Platform Expansion**: ARM Linux support
- **📊 Advanced Analytics**: Query performance monitoring and optimization

### **v1.0.0 - Production ANN** (Q4 2025)
- **🏆 Production-Grade ANN**: Competitive with Faiss for quality
- **🔄 Real-Time Updates**: Streaming vector ingestion
- **🕸️ Graph-Based Methods**: HNSW and NSW algorithm implementations
- **🤖 ML Framework Integration**: Native PyTorch, TensorFlow, Hugging Face support
- **📊 Enterprise Features**: Monitoring, observability, management APIs

---

## 🎯 **Usage Recommendations**

### **✅ Use NSeekFS v0.1.0 For:**
- **Production exact search** on medium datasets (<1M vectors)
- **High-performance SIMD** requirements
- **Memory-efficient** deployments with quantization
- **Research and prototyping** with real embeddings
- **Professional binary storage** and persistence needs

### **🤔 Consider Alternatives For:**
- **Production ANN at scale** → Use [Faiss](https://github.com/facebookresearch/faiss)
- **Very large datasets** (>10M vectors) → Use distributed solutions  
- **Graph-based ANN** → Use [HNSW](https://github.com/nmslib/hnswlib)
- **ARM Linux deployment** → Wait for v0.3.0 or use alternatives

---

## 📈 **Metrics & Quality Assurance**

### **Test Coverage**
- **Core Engine**: 95%+ test coverage
- **ANN Implementation**: Comprehensive recall/precision testing
- **API Layer**: Full integration test suite
- **Memory Safety**: Extensive fuzzing and stress testing

### **Performance Validation**
- **Benchmarked** on Intel i7-10700K, 32GB RAM
- **Cross-platform** testing on Linux, macOS, Windows
- **Memory profiling** for leak detection
- **SIMD validation** across different CPU architectures

### **Quality Metrics**
- **Zero memory leaks** in production testing
- **Thread safety** validated with concurrent access
- **Numerical stability** across all precision levels
- **Error handling** covers all edge cases

---

**💡 Summary**: NSeekFS v0.1.0 delivers production-ready exact search with experimental ANN. The Rust backend provides enterprise-grade performance and safety, while the Python API offers simplicity and power. Use for exact search today, experiment with ANN, and watch for v0.2.0's enhanced ANN capabilities.

**🎯 Next Steps**: [Try the examples](examples/), [read the docs](README.md), and [contribute](CONTRIBUTING.md) to help improve ANN quality in v0.2.0!