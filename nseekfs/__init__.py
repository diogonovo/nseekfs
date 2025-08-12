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
"""

from .highlevel import NSeek
from .nseekfs import PySearchEngine, py_prepare_bin_from_embeddings

# Export convenience functions at package level
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
    """
    return NSeek.from_embeddings(*args, **kwargs)

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
    """
    return NSeek.load_index(*args, **kwargs)

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
    """
    return NSeek.create_index(*args, **kwargs)

# Main exports - what users will import
__all__ = [
    # Main class
    "NSeek",
    
    # Convenience functions (recommended)
    "from_embeddings",
    "load_index", 
    "create_index",
    
    # Low-level components (advanced users)
    "PySearchEngine",
    "py_prepare_bin_from_embeddings",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Diogo Novo"
__email__ = "diogonovo@outlook.pt"