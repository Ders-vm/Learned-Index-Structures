"""
PGM-Index implementation using pygm library.
"""

import numpy as np
from pygm._pygm import PGMIndexDouble


class PGMIndex:
    """Wrapper for pygm PGM-Index implementation."""

    def __init__(self, epsilon: int = 64):
        self.epsilon = epsilon
        self.pgm = None
        self.keys = None
        self.built = False

        # Accuracy tracking
        self.total_queries = 0
        self.correct_predictions = 0   # approximate_rank was exact
        self.fallbacks = 0             # bisect_left used
        self.not_found = 0             # key not present
        self.false_negatives = 0       # approx failed but key exists

        self.segments = []

    def build_from_sorted_array(self, keys: np.ndarray):
        """Build the PGM-Index from a sorted array."""
        self.keys = np.sort(keys).astype(np.float64)
        self.pgm = PGMIndexDouble(iter(self.keys), self.epsilon, False, 0)
        self.segments = list(self.pgm.segments())
        self.built = True

    def search(self, key: float) -> bool:
        """Search for key using PGM-Index."""
        self.total_queries += 1

        if not self.built:
            self.not_found += 1
            return False

        key = float(key)

        # approximate rank
        predicted_pos, _ = self.pgm.approximate_rank(key)

        # direct prediction hit
        if predicted_pos < len(self.keys) and self.keys[predicted_pos] == key:
            self.correct_predictions += 1
            return True

        # binary search fallback
        idx = self.pgm.bisect_left(key)

        if idx < len(self.keys) and self.keys[idx] == key:
            self.fallbacks += 1
            return True

        # key not found
        self.not_found += 1
        return False

    def get_memory_usage(self) -> int:
        """Return memory usage in bytes for keys + PGM structure."""
        if not self.built:
            return 0

        return self.keys.nbytes + self.pgm.size_in_bytes()
