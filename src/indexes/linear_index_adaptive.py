import numpy as np
from src.indexes.learned_index_optimized import LearnedIndexOptimized


class LinearIndexAdaptive(LearnedIndexOptimized):
    """
    Learned index with adaptive window selection based on empirical error.
    Window := max(min_window, quantile(|pred - actual|)).
    """
    def __init__(self, quantile: float = 0.995, min_window: int = 16,
                 sample: int = 50_000, use_numpy: bool = True):
        super().__init__(window=min_window, use_numpy=use_numpy)
        self.quantile = quantile
        self.min_window = min_window
        self.sample = sample

    def build_from_sorted_array(self, keys: np.ndarray) -> None:
        super().build_from_sorted_array(keys)
        if self.n == 0:
            return

        # sample for error profiling
        k = min(self.sample, self.n)
        idxs = np.linspace(0, self.n - 1, k, dtype=int)
        sample_keys = keys[idxs]

        preds = np.clip((self.a * sample_keys + self.b).astype(int), 0, self.n - 1)
        abs_err = np.abs(idxs - preds)

        # choose window based on quantile
        q_err = int(np.quantile(abs_err, self.quantile))
        self.window = max(self.min_window, q_err)
