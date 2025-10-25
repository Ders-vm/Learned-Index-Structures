"""
===============================================================================
LINEAR MODEL INDEX (LEARNED INDEX BASELINE)
===============================================================================
This module implements a simple learned index using a single linear regression
model. It predicts the position of each key in a sorted array, then performs
a local binary search around that prediction for correction.

Usage:
    from src.indexes.linear_index import LearnedIndex
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
        self.keys = None  # the sorted keys
        self.correct_predictions = 0  # tracking correct predictions
        self.fallbacks = 0  # tracking fallbacks to full search
        self.not_found = 0  # tracking not found cases
        self.total_queries = 0  # total queries made

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
    def search(self, key: float, window: int = 64) -> bool:
        """Predict approximate position, then correct locally.
        Args:
            key: The key to search for.
            window: Search window size around predicted position.
        Returns:
            bool indicating whether the key was found.
        """
        if self.keys is None or len(self.keys) == 0:
            return False

        n = len(self.keys)
        self.total_queries += 1

        # Predict approximate index
        pred = int(self.a * key + self.b)

        # Clamp to valid range
        pred = max(0, min(n - 1, pred))

        # Define local search window
        left = max(0, pred - window)
        right = min(n, pred + window)

        # Local binary search correction
        idx = bisect.bisect_left(self.keys[left:right], key)

        # try this TODO
        #idx = np.searchsorted(self.keys[left:right], key, side='left')

        found = (idx + left < n) and (self.keys[idx + left] == key)

        #for debug tracking
        if found and (idx + left) == pred:
            self.correct_predictions += 1 # predicted position was correct
        elif not found:
            self.not_found += 1 # key was not found (no fallbacks so could be false negative)
        return found

    # ----------------------------------------------------------------------
    # Memory usage estimate
    # ----------------------------------------------------------------------
    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes."""
        return len(self.keys) * 8 + 16 + 16  # keys + a/b params + overhead
    

# ----------------------------------------------------------------------
# Quick sanity check / debug test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from ..utils.data_loader import DatasetGenerator

    print("Sanity Check: LearnedIndex\n")

    # Test different dataset types
    for name, gen_func in [
        ("Sequential", DatasetGenerator.generate_sequential),
        ("Uniform", DatasetGenerator.generate_uniform),
        ("Mixed", DatasetGenerator.generate_mixed),
    ]:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        # Generate and build
        keys = gen_func(1000)
        index = LearnedIndex()
        index.build_from_sorted_array(keys)

        # Print model parameters and error window
        print(f"Slope (a): {index.a:.6f}")
        print(f"Intercept (b): {index.b:.6f}")
        print(f"Memory usage: {index.get_memory_usage() / 1024:.2f} KB")

        # Test a few existing keys
        test_keys = [
            keys[0],                      # first
            keys[len(keys)//2],           # middle
            keys[-1],                     # last
            float(keys[len(keys)//2] + 1) # nearby key (probably not in dataset)
        ]

        for k in test_keys:
            found = index.search(k)
            print(f"Key={k:.2f} -> Found={found}")

        # Test a random non-existing key
        q = float(np.random.uniform(keys.min(), keys.max()))
        found = index.search(q)
        print(f"\nRandom query: {q:.2f} -> Found={found}")
    
    print("\nLearnedIndex sanity check complete.\n")
   
