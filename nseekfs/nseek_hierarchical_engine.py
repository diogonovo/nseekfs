import numpy as np
from nseekfs import PySearchEngine

LEVELS_DEFAULT = ["f8", "f16", "f32", "f64"]
LEVEL_PERCENT = {
    "f8": 0.01,
    "f16": 0.1,
    "f32": 1.0,
    "f64": 1.0,
}

class HierarchicalEngine:
    def __init__(self, levels: dict, disable_ann: bool = False):
        """
        levels: dict {"f16": path, "f32": path, ...}
        disable_ann: se True, força busca exaustiva (sem ANN)
        """
        if not levels or not isinstance(levels, dict):
            raise ValueError("You must provide a dictionary with level → path mapping.")

        self.engines = {}
        self.level_order = []
        self.disable_ann = disable_ann

        for level in LEVELS_DEFAULT:
            if level in levels:
                try:
                    self.engines[level] = PySearchEngine(levels[level], use_ann=not disable_ann)
                    self.level_order.append(level)
                except Exception as e:
                    raise RuntimeError(f"Failed to load level {level}: {e}")

        if not self.engines:
            raise ValueError("No valid levels were loaded.")

    def search(self, query_vector, path=None, top_k=10):
        if path is None:
            path = self.level_order

        if not isinstance(query_vector, (np.ndarray, list)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        first_level = path[0]
        if first_level not in self.engines:
            raise ValueError(f"First level '{first_level}' not found in engine.")

        expected_dims = self.engines[first_level].dims()
        if query_vector.shape[-1] != expected_dims:
            raise ValueError(f"Dimensionality mismatch: expected {expected_dims}, got {query_vector.shape[-1]}")

        candidates = None
        for i, level in enumerate(path):
            if level not in self.engines:
                raise ValueError(f"Level '{level}' not loaded.")

            engine = self.engines[level]

            if candidates is None:
                current_k = max(1, int(engine.rows() * LEVEL_PERCENT.get(level, 1.0)))
                results = engine.top_k_query(query_vector, current_k)
            else:
                idxs = [idx for idx, _ in candidates if isinstance(idx, int)]
                current_k = max(1, int(len(idxs) * LEVEL_PERCENT.get(level, 1.0)))
                results = engine.top_k_subset(query_vector, idxs, current_k)

            candidates = results

        return candidates[:top_k] if candidates else []

    def get_vector(self, level: str, idx: int):
        if level not in self.engines:
            raise ValueError(f"Level '{level}' not loaded.")
        return self.engines[level].get_vector(idx)

    def dims(self):
        return {lvl: e.dims() for lvl, e in self.engines.items()}

    def rows(self):
        return {lvl: e.rows() for lvl, e in self.engines.items()}