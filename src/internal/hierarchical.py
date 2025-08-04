import numpy as np
import logging
from ...nseekfs.engine_core import PySearchEngine

logger = logging.getLogger(__name__)

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
        levels: dict like {"f16": "path/to/f16.bin", "f32": "path/to/f32.bin", ...}
        disable_ann: if True, force exhaustive search (no ANN index)
        """
        if not isinstance(levels, dict) or not levels:
            raise ValueError("A dictionary with level → path is required.")

        self.engines = {}
        self.level_order = []
        self.disable_ann = disable_ann

        logger.info(f"Initializing HierarchicalEngine with levels: {list(levels.keys())} → ANN={'disabled' if disable_ann else 'enabled'}")

        for level in LEVELS_DEFAULT:
            path = levels.get(level)
            if path:
                try:
                    self.engines[level] = PySearchEngine(path, use_ann=not disable_ann)
                    self.level_order.append(level)
                    logger.info(f"Loaded level '{level}' from {path}")
                except Exception as e:
                    logger.error(f"Failed to load level '{level}' from {path}: {e}")
                    raise RuntimeError(f"Error loading level {level}: {e}")

        if not self.engines:
            raise ValueError("No valid levels were loaded.")

    def search(self, query_vector, path=None, top_k=10):
        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        path = path or self.level_order
        first_level = path[0]
        if first_level not in self.engines:
            raise ValueError(f"Level '{first_level}' is not loaded.")

        expected_dims = self.engines[first_level].dims()
        if query_vector.shape[-1] != expected_dims:
            raise ValueError(f"Invalid vector dimension: expected {expected_dims}, got {query_vector.shape[-1]}")

        logger.debug(f"Starting hierarchical search on path={path} with top_k={top_k}")

        candidates = None

        for level in path:
            engine = self.engines.get(level)
            if engine is None:
                raise ValueError(f"Level '{level}' is not loaded.")

            percent = LEVEL_PERCENT.get(level, 1.0)
            if candidates is None:
                current_k = max(1, int(engine.rows() * percent))
                logger.debug(f"Level {level} → selecting {current_k} candidates from full set")
                results = engine.top_k_query(query_vector.tolist(), current_k)
            else:
                idxs = [idx for idx, _ in candidates]
                current_k = max(1, int(len(idxs) * percent))
                logger.debug(f"Level {level} → refining {len(idxs)} → {current_k} candidates")
                results = engine.top_k_subset(query_vector.tolist(), idxs, current_k)

            candidates = results

        final_results = candidates[:top_k] if candidates else []
        logger.info(f"Search completed → final top_k={len(final_results)} results")
        return final_results

    def get_vector(self, level: str, idx: int):
        engine = self.engines.get(level)
        if engine is None:
            raise ValueError(f"Level '{level}' is not loaded.")
        return engine.get_vector(idx)

    def dims(self):
        return {lvl: eng.dims() for lvl, eng in self.engines.items()}

    def rows(self):
        return {lvl: eng.rows() for lvl, eng in self.engines.items()}

    def __repr__(self):
        loaded = ", ".join(self.level_order)
        return f"<HierarchicalEngine levels=[{loaded}] ann={'disabled' if self.disable_ann else 'enabled'}>"

