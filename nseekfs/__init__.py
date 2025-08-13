# nseekfs/__init__.py  (versão segura, lazy)
from importlib import import_module

__version__ = "0.1.0"
__author__ = "Diogo Novo"
__email__ = "diogonovo@outlook.pt"
__license__ = "MIT"
__description__ = "High-performance vector similarity search with Rust backend"

# Lazy proxies: só importam quando usados
def _hl():
    return import_module("nseekfs.highlevel")

def from_embeddings(*args, **kwargs):
    return _hl().NSeek.from_embeddings(*args, **kwargs)

def load_index(*args, **kwargs):
    return _hl().NSeek.load_index(*args, **kwargs)

def create_index(*args, **kwargs):
    return _hl().NSeek.create_index(*args, **kwargs)

def health_check():
    return _hl().health_check()

def get_system_info():
    return _hl().get_system_info()

def validate_config():
    return _hl().validate_config()

# Expor classes/exceções através do highlevel (lazy)
class _Exports:
    @property
    def NSeek(self):
        return _hl().NSeek
    @property
    def NSeekError(self):
        return _hl().NSeekError
    @property
    def NSeekValidationError(self):
        return _hl().NSeekValidationError
    @property
    def NSeekMemoryError(self):
        return _hl().NSeekMemoryError
    @property
    def NSeekIndexError(self):
        return _hl().NSeekIndexError

exports = _Exports()

# Low-level (apenas para utilizadores avançados) — também lazy
def _low():
    return import_module("nseekfs.nseekfs")

def PySearchEngine(*args, **kwargs):               # factory
    return _low().PySearchEngine(*args, **kwargs)

def py_prepare_bin_from_embeddings(*args, **kwargs):
    return _low().py_prepare_bin_from_embeddings(*args, **kwargs)

__all__ = [
    "exports",
    "from_embeddings", "load_index", "create_index",
    "health_check", "get_system_info", "validate_config",
    "PySearchEngine", "py_prepare_bin_from_embeddings",
    "__version__", "__author__", "__license__",
]
