# nseekfs/__init__.py
"""
NSeekFS - High-performance vector similarity search with Timer Fix

NSeekFS provides a production-ready vector similarity search engine with:
- Fast exact search with SIMD optimization
- Experimental ANN for research use
- Professional Rust backend with Python bindings
- Multiple precision levels (f8/f16/f32/f64)
- Comprehensive timing and performance metrics

Simple Usage:
    import nseekfs
    index = nseekfs.from_embeddings(vectors)
    results = index.query(query_vector, top_k=10)

Advanced Usage with Timing:
    results, timing = index.query(query_vector, top_k=10, return_timing=True)
    print(f"Query took {timing['query_time_ms']:.2f}ms using {timing['method_used']}")
"""

import sys
import warnings
from importlib import import_module
from typing import Any, Dict, List, Optional, Union

# =============== METADATA ===============
__version__ = "0.1.0"
__author__ = "Diogo Novo"
__email__ = "diogonovo@outlook.pt"
__license__ = "MIT"
__description__ = "High-performance vector similarity search with Rust backend and timer fixes"
__url__ = "https://github.com/diogonovo/nseekfs"

# Version info tuple
__version_info__ = tuple(map(int, __version__.split('.')))

# =============== LAZY LOADING WITH ERROR HANDLING ===============

def _hl():
    """Lazy import of highlevel module with error handling"""
    try:
        return import_module("nseekfs.highlevel")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.highlevel: {e}. Please ensure NSeekFS is properly installed.")

def _ll():
    """Lazy import of low-level module with error handling"""
    try:
        return import_module("nseekfs.nseekfs")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.nseekfs: {e}. Please ensure the Rust extension is compiled.")

# =============== SYSTEM COMPATIBILITY CHECKS ===============

def _check_system_compatibility():
    """Check if system is compatible with NSeekFS"""
    issues = []
    
    # Python version check
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, got {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check for numpy
    try:
        import numpy as np
        if hasattr(np, '__version__'):
            np_version = tuple(map(int, np.__version__.split('.')[:2]))
            if np_version < (1, 19):
                issues.append(f"NumPy 1.19+ recommended, got {np.__version__}")
    except ImportError:
        issues.append("NumPy is required but not installed")
    
    # Check for basic Rust module
    try:
        _ll()
    except ImportError as e:
        issues.append(f"Rust extension not available: {e}")
    
    if issues:
        warning_msg = "NSeekFS compatibility issues detected:\n" + "\n".join(f"  - {issue}" for issue in issues)
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
    
    return len(issues) == 0

# Run compatibility check on import
_SYSTEM_COMPATIBLE = _check_system_compatibility()

# =============== PRIMARY API (RECOMMENDED) ===============
# This is what 99% of users should use

def from_embeddings(*args, **kwargs):
    """
    Create vector search index from embeddings with timer support
    
    Args:
        embeddings: numpy array of shape (n_vectors, dimensions)
        level: precision level ("f32", "f16", "f8") - default "f32"
        ann: enable approximate nearest neighbor (bool) - default True
        base_name: file name for saved index - default "default"
        output_dir: directory to save index files - default "./nseekfs_indexes"
        normalized: whether to normalize vectors - default True
        force_rebuild: force rebuild even if exists - default False
        
    Returns:
        VectorSearch: Ready-to-query vector search index with timing support
        
    Example:
        import nseekfs
        import numpy as np
        
        vectors = np.random.randn(1000, 384).astype(np.float32)
        index = nseekfs.from_embeddings(vectors)
        
        # Basic query
        results = index.query(query_vector, top_k=10)
        
        # Query with timing
        results, timing = index.query(query_vector, top_k=10, return_timing=True)
        print(f"Query took {timing['query_time_ms']:.2f}ms")
    """
    if not _SYSTEM_COMPATIBLE:
        warnings.warn("System compatibility issues detected. NSeekFS may not work correctly.", UserWarning)
    
    return _hl().VectorSearch.from_embeddings(*args, **kwargs)

def load_index(*args, **kwargs):
    """
    Load existing vector search index from file with timer support
    
    Args:
        bin_path: path to .bin index file
        ann: enable approximate nearest neighbor (bool) - default True
        normalized: whether vectors are normalized - default True
        level: precision level (auto-detected if None)
        
    Returns:
        VectorSearch: Loaded vector search index with timing support
        
    Example:
        import nseekfs
        
        index = nseekfs.load_index("my_index_f32.bin")
        results = index.query(query_vector, top_k=10)
    """
    if not _SYSTEM_COMPATIBLE:
        warnings.warn("System compatibility issues detected. NSeekFS may not work correctly.", UserWarning)
    
    return _hl().VectorSearch.load_index(*args, **kwargs)

def create_index(*args, **kwargs):
    """
    Create index with advanced options (low-level API)
    
    This creates only the binary file without loading it.
    Most users should use from_embeddings() instead.
    """
    return _hl().VectorSearch.create_index(*args, **kwargs)

# =============== UTILITIES ===============

def health_check() -> Dict[str, Any]:
    """
    Check system capabilities and NSeekFS performance
    
    Returns:
        Dictionary with health status, timing info, and system capabilities
        
    Example:
        health = nseekfs.health_check()
        print(f"Status: {health['status']}")
        print(f"Basic test time: {health['basic_test_time_ms']:.2f}ms")
    """
    import time
    import numpy as np
    
    health_info = {
        "status": "unknown",
        "system_compatible": _SYSTEM_COMPATIBLE,
        "rust_engine_working": False,
        "basic_test_time_ms": 0.0,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "nseekfs_version": __version__,
        "errors": []
    }
    
    try:
        # Test basic engine loading
        engine_test_start = time.time()
        _ = _ll()
        health_info["rust_engine_working"] = True
        
        # Basic functionality test
        test_vectors = np.random.randn(50, 32).astype(np.float32)
        
        test_start = time.time()
        index = from_embeddings(test_vectors, base_name="health_check", ann=False)
        query_vector = test_vectors[0]
        results = index.query(query_vector, top_k=5)
        test_time = (time.time() - test_start) * 1000
        
        health_info["basic_test_time_ms"] = test_time
        
        if len(results) == 5 and results[0]["idx"] == 0:
            health_info["status"] = "healthy"
        else:
            health_info["status"] = "degraded"
            health_info["errors"].append("Basic test returned unexpected results")
            
    except Exception as e:
        health_info["status"] = "error"
        health_info["errors"].append(str(e))
    
    return health_info

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information and capabilities
    
    Returns:
        Dictionary with platform, Python version, dependencies, etc.
    """
    import platform
    
    info = {
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "python_version": sys.version,
        "nseekfs_version": __version__,
        "system_compatible": _SYSTEM_COMPATIBLE,
    }
    
    # Check dependencies
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "not installed"
    
    # Check Rust engine
    try:
        ll_module = _ll()
        info["rust_engine"] = "available"
        if hasattr(ll_module, '__version__'):
            info["rust_engine_version"] = ll_module.__version__
    except ImportError:
        info["rust_engine"] = "not available"
    
    return info

def validate_config() -> Dict[str, Any]:
    """
    Validate current NSeekFS configuration
    
    Returns:
        Dictionary with configuration validation results
    """
    config = {
        "config_valid": True,
        "max_concurrent_queries": 1000,  # Conservative default
        "memory_warning_threshold_gb": 2.0,
        "supported_levels": ["f8", "f16", "f32", "f64"],
        "issues": []
    }
    
    # Check system compatibility
    if not _SYSTEM_COMPATIBLE:
        config["config_valid"] = False
        config["issues"].append("System compatibility issues detected")
    
    # Check available memory (approximate)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 1.0:
            config["issues"].append(f"Low available memory: {available_gb:.1f}GB")
    except ImportError:
        config["issues"].append("psutil not available - cannot check memory")
    
    return config

# =============== ADVANCED ACCESS (POWER USERS) ===============

def _get_vector_search_class():
    """Get VectorSearch class (lazy)"""
    return _hl().VectorSearch

def _get_pysearchengine_class():
    """Get PySearchEngine class (lazy)"""
    return _ll().PySearchEngine

# Property-like access for backward compatibility
class _LazyClass:
    def __init__(self, loader_func):
        self._loader = loader_func
        self._loaded = None
    
    def __call__(self, *args, **kwargs):
        if self._loaded is None:
            self._loaded = self._loader()
        return self._loaded(*args, **kwargs)
    
    def __getattr__(self, name):
        if self._loaded is None:
            self._loaded = self._loader()
        return getattr(self._loaded, name)

# Advanced class access
VectorSearch = _LazyClass(_get_vector_search_class)
PySearchEngine = _LazyClass(_get_pysearchengine_class)

# Legacy aliases for backward compatibility
VectorIndex = VectorSearch  # Legacy name
NSeek = VectorSearch        # Alternative name

# =============== EXCEPTIONS (LAZY LOADED) ===============

def _get_exceptions():
    """Get exception classes from highlevel module"""
    hl = _hl()
    return {
        "ValidationError": getattr(hl, "ValidationError", ValueError),
        "IndexError": getattr(hl, "IndexError", Exception),
    }

class _LazyExceptions:
    def __init__(self):
        self._exceptions = None
    
    def _load(self):
        if self._exceptions is None:
            self._exceptions = _get_exceptions()
        return self._exceptions
    
    @property
    def ValidationError(self):
        return self._load()["ValidationError"]
    
    @property
    def IndexError(self):
        return self._load()["IndexError"]

_exc = _LazyExceptions()
ValidationError = _exc.ValidationError
IndexError = _exc.IndexError

# =============== DEBUGGING AND DEVELOPMENT ===============

def _enable_debug_logging():
    """Enable debug logging for development"""
    import logging
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable debug for nseekfs modules
    for module_name in ['nseekfs.highlevel', 'nseekfs.nseekfs']:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)

# Environment variable to enable debug mode
import os
if os.getenv('NSEEKFS_DEBUG', '0').lower() in ('1', 'true', 'yes'):
    _enable_debug_logging()

# =============== EXPORTS ===============

__all__ = [
    # Primary API (recommended for 99% of users)
    "from_embeddings",      # Main entry point
    "load_index",           # Load existing index
    "create_index",         # Low-level creation
    
    # Utilities
    "health_check",         # System health and performance
    "get_system_info",      # Platform information
    "validate_config",      # Configuration validation
    
    # Advanced access (power users)
    "VectorSearch",         # Main class
    "PySearchEngine",       # Low-level engine
    
    # Legacy aliases
    "VectorIndex",          # Old name for VectorSearch
    "NSeek",               # Alternative name
    
    # Exceptions
    "ValidationError",      # Input validation errors
    "IndexError",          # Index operation errors
    
    # Metadata
    "__version__",         # Version string
    "__version_info__",    # Version tuple
    "__author__",          # Author name
    "__license__",         # License
    "__description__",     # Package description
    "__url__",            # Homepage URL
]

# =============== PACKAGE-LEVEL CONFIGURATION ===============

# Default warning filters for common issues
warnings.filterwarnings('ignore', category=UserWarning, module='nseekfs', message='.*compatibility.*')

# Print helpful info on first import (only in interactive sessions)
if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
    # Interactive session detected
    if not _SYSTEM_COMPATIBLE:
        print(f"⚠️  NSeekFS {__version__} loaded with compatibility issues. Run nseekfs.health_check() for details.")
    else:
        print(f"✅ NSeekFS {__version__} loaded successfully. Use nseekfs.from_embeddings() to get started.")