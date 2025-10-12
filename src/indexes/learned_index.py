"""
===============================================================================
LINEAR MODEL INDEX (LEARNED INDEX BASELINE)
===============================================================================
This module implements a simple learned index using a single linear regression
model. It predicts the position of each key in a sorted array, then performs
a local binary search around that prediction for correction.

Usage:
    from src.indexes.linear_model_index import LearnedIndex
    from src.utils.data_loader import DatasetGenerator

    keys = DatasetGenerator.generate_uniform(10000)
    index = LearnedIndex()
    index.build_from_sorted_array(keys)
    print(index.search(keys[100]))
===============================================================================
"""

import numpy as np
import bisect


class LearnedIndex:
    """Simple learned index using linear regression and local search correction."""

    def __init__(self):
        self.a = 0.0  # slope
        self.b = 0.0  # intercept
        self.keys = None

    # ----------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------
    def build_from_sorted_array(self, keys: np.ndarray):
        """Fit a linear regression model to predict key position."""
        self.keys = keys
        n = len(keys)
        if n == 0:
            return

        # Normalize positions (0 to n-1)
        positions = np.arange(n)

        # Fit simple linear regression: position ≈ a * key + b
        self.a, self.b = np.polyfit(keys, positions, 1)

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(self, key: float, window: int = 64):
        """Predict approximate position, then correct locally."""
        if self.keys is None or len(self.keys) == 0:
            return False, 0

        n = len(self.keys)

        # Predict approximate index
        pred = int(self.a * key + self.b)

        # Clamp to valid range
        pred = max(0, min(n - 1, pred))

        # Define local search window
        left = max(0, pred - window)
        right = min(n, pred + window)

        # Local binary search correction
        idx = bisect.bisect_left(self.keys[left:right], key)
        found = (idx + left < n) and (self.keys[idx + left] == key)
        return found, 1

    # ----------------------------------------------------------------------
    # Memory usage estimate
    # ----------------------------------------------------------------------
    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes."""
        return len(self.keys) * 8 + 16 + 16  # keys + a/b params + overhead
