"""
MinMax Adaptive Index - Alternative to quantile-based approach.

NOTE: Testing shows that quantile=0.99 generally performs better than MinMax.
This is provided as an alternative for comparison purposes.
"""

import numpy as np
from src.indexes.learned_index_optimized import LearnedIndexOptimized


class MinMaxAdaptiveIndex(LearnedIndexOptimized):
    """
    Alternative adaptive window using max observed error instead of quantile.
    
    Window := max(min_window, max(|pred - actual|))
    
    Note: Quantile-based approach (LinearIndexAdaptive with quantile=0.99) 
    generally performs better as it's less sensitive to outliers.
    """
    def __init__(self, min_window: int = 16, sample: int = 50_000, use_numpy: bool = True):
        """
        Args:
            min_window: Minimum window size (default: 16)
            sample: Number of samples for error profiling (default: 50,000)
            use_numpy: Must be True for performance (np.searchsorted is 44x faster than bisect)
        """
        super().__init__(window=min_window, use_numpy=use_numpy)
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
        
        # predict positions
        preds = np.clip((self.a * sample_keys + self.b).astype(int), 0, self.n - 1)
        abs_err = np.abs(idxs - preds)
        
        # use maximum error instead of quantile
        max_err = int(np.max(abs_err))
        self.window = max(self.min_window, max_err)
