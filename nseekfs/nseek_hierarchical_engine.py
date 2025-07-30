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
    def __init__(self, levels):
        """
        levels: dict {"f16": path, "f32": path, ...}
        Only levels present will be used.
        """
        self.engines = {}
        self.level_order = []

        for level in LEVELS_DEFAULT:
            if level in levels:
                self.engines[level] = PySearchEngine(levels[level])
                self.level_order.append(level)

    def search(self, query_vector, path=None, top_k=10):
        if path is None:
            path = self.level_order

        candidates = None
        for i, level in enumerate(path):
            engine = self.engines[level]

            if candidates is None:
                current_k = max(1, int(engine.rows() * LEVEL_PERCENT[level]))
                results = engine.top_k_query(query_vector, current_k)
            else:
                idxs = [idx for idx, _ in candidates]
                current_k = max(1, int(len(idxs) * LEVEL_PERCENT[level]))
                results = engine.top_k_subset(query_vector, idxs, current_k)

            candidates = results

        return candidates[:top_k]

    def get_vector(self, level, idx):
        return self.engines[level].get_vector(idx)

    def dims(self):
        return {lvl: e.dims() for lvl, e in self.engines.items()}

    def rows(self):
        return {lvl: e.rows() for lvl, e in self.engines.items()}
