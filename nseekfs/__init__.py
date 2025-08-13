"""
NSeekFS - High-Performance Vector Similarity Search

A blazing-fast vector similarity search engine built with Rust and optimized 
for machine learning workloads.

Basic usage:
    import nseekfs
    import numpy as np
    
    # Create embeddings
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    
    # Create index (simple way)
    index = nseekfs.from_embeddings(embeddings)
    
    # Or using the class directly
    index = nseekfs.NSeek.from_embeddings(embeddings)
    
    # Search
    results = index.query(embeddings[0], top_k=10)

Safety Features:
    - Comprehensive input validation
    - Memory leak prevention
    - Thread-safe operations
    - Resource limit enforcement
    - Graceful error handling
"""

import logging
import warnings
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import order is important for proper initialization
try:
    from .highlevel import NSeek, NSeekError, NSeekValidationError, NSeekMemoryError, NSeekIndexError
    from .nseekfs import PySearchEngine, py_prepare_bin_from_embeddings
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    raise ImportError(f"NSeekFS core modules not available: {e}")

# Package metadata
__version__ = "0.1.0"
__author__ = "Diogo Novo"
__email__ = "diogonovo@outlook.pt"
__license__ = "MIT"
__description__ = "High-performance vector similarity search with Rust backend"

# Convenience functions with enhanced error handling
def from_embeddings(*args, **kwargs):
    """
    Create an index from embeddings (convenience function).
    
    This is equivalent to NSeek.from_embeddings() but can be called directly
    as nseekfs.from_embeddings().
    
    Args:
        embeddings: Input embeddings (array, list, or file path)
        level: Precision level ("f8", "f16", "f32", "f64")
        normalized: Whether embeddings are normalized
        ann: Enable approximate nearest neighbors
        base_name: Base name for index files
        output_dir: Output directory path
        force_rebuild: Force rebuild even if index exists
        
    Returns:
        NSeek: Ready-to-use index instance
        
    Raises:
        NSeekValidationError: Invalid input parameters
        NSeekMemoryError: Memory constraints exceeded
        NSeekIndexError: Index creation failed
        
    Example:
        >>> import nseekfs
        >>> import numpy as np
        >>> embeddings = np.random.randn(1000, 384).astype(np.float32)
        >>> index = nseekfs.from_embeddings(embeddings, ann=True)
        >>> results = index.query(embeddings[0], top_k=5)
    """
    try:
        return NSeek.from_embeddings(*args, **kwargs)
    except Exception as e:
        logger.error(f"from_embeddings failed: {e}")
        raise

def load_index(*args, **kwargs):
    """
    Load an existing index from disk (convenience function).
    
    This is equivalent to NSeek.load_index() but can be called directly
    as nseekfs.load_index().
    
    Args:
        bin_path: Path to the .bin file
        normalized: Whether vectors are normalized
        ann: Enable ANN if available
        level: Precision level (inferred from filename if not specified)
        
    Returns:
        NSeek: Loaded index instance
        
    Raises:
        FileNotFoundError: Binary file not found
        NSeekIndexError: Failed to load index
        NSeekValidationError: Invalid parameters
        
    Example:
        >>> import nseekfs
        >>> index = nseekfs.load_index("my_index.bin", ann=True)
        >>> results = index.query(query_vector, top_k=10)
    """
    try:
        return NSeek.load_index(*args, **kwargs)
    except Exception as e:
        logger.error(f"load_index failed: {e}")
        raise

def create_index(*args, **kwargs):
    """
    Create a binary index from embeddings (convenience function).
    
    This is equivalent to NSeek.create_index() but can be called directly
    as nseekfs.create_index().
    
    Args:
        embeddings: Input embeddings
        level: Precision level
        normalized: Normalization setting
        ann: Enable ANN
        base_name: Base name for files
        output_dir: Output directory
        seed: Random seed for ANN index
        
    Returns:
        str: Path to the created .bin file
        
    Raises:
        NSeekValidationError: Invalid input parameters
        NSeekMemoryError: Memory constraints exceeded
        NSeekIndexError: Index creation failed
        
    Example:
        >>> import nseekfs
        >>> import numpy as np
        >>> embeddings = np.random.randn(500, 128).astype(np.float32)
        >>> bin_path = nseekfs.create_index(embeddings, level="f32", ann=True)
        >>> print(f"Index created at: {bin_path}")
    """
    try:
        return NSeek.create_index(*args, **kwargs)
    except Exception as e:
        logger.error(f"create_index failed: {e}")
        raise

# Health check function
def health_check() -> Dict[str, Any]:
    """
    Perform a system health check for NSeekFS.
    
    Returns:
        Dict containing health status and system information
        
    Example:
        >>> import nseekfs
        >>> health = nseekfs.health_check()
        >>> print(f"Status: {health['status']}")
    """
    try:
        import numpy as np
        import tempfile
        
        health = {
            "status": "healthy",
            "version": __version__,
            "numpy_available": True,
            "rust_backend": True,
            "temp_dir_writable": False,
            "warnings": []
        }
        
        # Test numpy
        try:
            test_array = np.random.randn(10, 8).astype(np.float32)
            health["numpy_version"] = np.__version__
        except Exception as e:
            health["status"] = "degraded"
            health["warnings"].append(f"NumPy test failed: {e}")
        
        # Test temp directory
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                test_index = from_embeddings(
                    test_array, output_dir=temp_dir, base_name="health_check"
                )
                results = test_index.query(test_array[0], top_k=3)
                if len(results) >= 1 and results[0]["idx"] == 0:
                    health["temp_dir_writable"] = True
                else:
                    health["warnings"].append("Basic functionality test failed")
        except Exception as e:
            health["status"] = "degraded"
            health["warnings"].append(f"Temp directory test failed: {e}")
        
        # Test Rust backend
        try:
            engine = PySearchEngine.__new__(PySearchEngine)
            health["rust_backend"] = True
        except Exception as e:
            health["status"] = "unhealthy"
            health["warnings"].append(f"Rust backend unavailable: {e}")
        
        return health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "version": __version__,
            "error": str(e)
        }

# System information
def get_system_info() -> Dict[str, Any]:
    """
    Get system information relevant to NSeekFS.
    
    Returns:
        Dict containing system information
    """
    import sys
    import platform
    
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "not available"
    
    return {
        "nseekfs_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "numpy_version": numpy_version,
        "rust_backend": "available" if hasattr(PySearchEngine, '__new__') else "unavailable"
    }

# Configuration and limits
class Config:
    """NSeekFS configuration constants"""
    
    # Dimension limits
    MIN_DIMENSIONS = 8
    MAX_DIMENSIONS = 10000
    
    # Dataset limits  
    MIN_VECTORS = 1
    MAX_VECTORS = 100_000_000
    
    # Query limits
    MAX_TOP_K = 100000
    
    # Memory limits (bytes)
    MEMORY_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB
    
    # Supported precision levels
    PRECISION_LEVELS = {"f8", "f16", "f32", "f64"}
    
    # Supported similarity metrics
    SIMILARITY_METRICS = {"cosine", "euclidean", "dot_product", "dot"}
    
    # Supported search methods
    SEARCH_METHODS = {"simd", "scalar", "auto"}

# Utility functions
def validate_config() -> bool:
    """
    Validate NSeekFS configuration and dependencies.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ImportError: Missing required dependencies
        RuntimeError: Configuration validation failed
    """
    try:
        # Check NumPy
        import numpy as np
        if not hasattr(np, 'float32'):
            raise RuntimeError("NumPy lacks required float32 support")
        
        # Check Rust backend
        if not hasattr(PySearchEngine, '__new__'):
            raise RuntimeError("Rust backend not properly initialized")
        
        # Check file system access
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.bin")
            with open(test_file, 'wb') as f:
                f.write(b"test")
            if not os.path.exists(test_file):
                raise RuntimeError("File system access test failed")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

# Warning filters for known issues
def _setup_warnings():
    """Setup warning filters for common issues"""
    # Filter out specific warnings that are expected
    warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
    
    # Custom warning for large datasets
    def large_dataset_warning(size_mb: float):
        if size_mb > 1000:  # 1GB
            warnings.warn(
                f"Large dataset detected ({size_mb:.1f}MB). "
                "Consider using lower precision levels for memory efficiency.",
                UserWarning,
                stacklevel=3
            )

# Setup warnings on import
_setup_warnings()

# Main exports - what users will import
__all__ = [
    # Main class
    "NSeek",
    
    # Exception classes
    "NSeekError",
    "NSeekValidationError", 
    "NSeekMemoryError",
    "NSeekIndexError",
    
    # Convenience functions (recommended)
    "from_embeddings",
    "load_index", 
    "create_index",
    
    # Utility functions
    "health_check",
    "get_system_info",
    "validate_config",
    
    # Configuration
    "Config",
    
    # Low-level components (advanced users)
    "PySearchEngine",
    "py_prepare_bin_from_embeddings",
    
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

# Initialize logging for the package
def _init_logging():
    """Initialize package logging"""
    # Only add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Default to WARNING level

# Initialize on import
_init_logging()

# Perform basic validation on import
try:
    validate_config()
    logger.debug("NSeekFS initialized successfully")
except Exception as e:
    logger.warning(f"NSeekFS initialization warning: {e}")

# Version compatibility check
def _check_version_compatibility():
    """Check for version compatibility issues"""
    import sys
    
    if sys.version_info < (3, 8):
        warnings.warn(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not officially supported. "
            "Please upgrade to Python 3.8 or later.",
            UserWarning
        )
    
    try:
        import numpy as np
        if hasattr(np, '__version__'):
            major, minor = map(int, np.__version__.split('.')[:2])
            if major == 1 and minor < 21:
                warnings.warn(
                    f"NumPy {np.__version__} is older than recommended (1.21+). "
                    "Some features may not work correctly.",
                    UserWarning
                )
    except (ImportError, ValueError, AttributeError):
        pass

# Check compatibility on import
_check_version_compatibility()