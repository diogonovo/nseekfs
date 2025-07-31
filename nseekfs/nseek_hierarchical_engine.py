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
        levels: dict como {"f16": "path/to/f16.bin", "f32": "path/to/f32.bin", ...}
        disable_ann: se True, força busca exaustiva (sem índice ANN)
        """
        if not isinstance(levels, dict) or not levels:
            raise ValueError("É necessário fornecer um dicionário com os níveis → caminho.")

        self.engines = {}
        self.level_order = []
        self.disable_ann = disable_ann

        for level in LEVELS_DEFAULT:
            path = levels.get(level)
            if path:
                try:
                    self.engines[level] = PySearchEngine(path, use_ann=not disable_ann)
                    self.level_order.append(level)
                except Exception as e:
                    raise RuntimeError(f"Erro ao carregar nível {level}: {e}")

        if not self.engines:
            raise ValueError("Nenhum nível válido foi carregado.")

    def search(self, query_vector, path=None, top_k=10):
        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("O vetor de consulta deve ser uma lista ou um array numpy.")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        path = path or self.level_order
        first_level = path[0]
        if first_level not in self.engines:
            raise ValueError(f"Nível '{first_level}' não está carregado.")

        expected_dims = self.engines[first_level].dims()
        if query_vector.shape[-1] != expected_dims:
            raise ValueError(f"Dimensão inválida: esperada {expected_dims}, recebida {query_vector.shape[-1]}")

        candidates = None

        for level in path:
            engine = self.engines.get(level)
            if engine is None:
                raise ValueError(f"Nível '{level}' não está carregado.")

            percent = LEVEL_PERCENT.get(level, 1.0)
            if candidates is None:
                current_k = max(1, int(engine.rows() * percent))
                results = engine.top_k_query(query_vector.tolist(), current_k)
            else:
                idxs = [idx for idx, _ in candidates]
                current_k = max(1, int(len(idxs) * percent))
                results = engine.top_k_subset(query_vector.tolist(), idxs, current_k)

            candidates = results

        return candidates[:top_k] if candidates else []

    def get_vector(self, level: str, idx: int):
        engine = self.engines.get(level)
        if engine is None:
            raise ValueError(f"Nível '{level}' não está carregado.")
        return engine.get_vector(idx)

    def dims(self):
        return {lvl: eng.dims() for lvl, eng in self.engines.items()}

    def rows(self):
        return {lvl: eng.rows() for lvl, eng in self.engines.items()}

    def __repr__(self):
        loaded = ", ".join(self.level_order)
        return f"<HierarchicalEngine níveis=[{loaded}] ann={'desligado' if self.disable_ann else 'ligado'}>"
