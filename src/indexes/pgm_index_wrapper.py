import numpy as np
from pygm._pygm import PGMIndexDouble


class PGMIndex:
    """Wrapper for pygm PGM-Index implementation."""

    def __init__(self, epsilon: int = 64):
        self.epsilon = epsilon
        self.pgm = None
        self.keys = None
        self.built = False

        # metrics
        self.total_queries = 0
        self.correct_predictions = 0
        self.fallbacks = 0
        self.false_negatives = 0
        self.not_found = 0

        self.segments = []

    def build_from_sorted_array(self, keys: np.ndarray):
        self.keys = np.sort(keys).astype(np.float64)
        self.pgm = PGMIndexDouble(iter(self.keys), self.epsilon, False, 0)
        self.segments = list(self.pgm.segments())
        self.built = True

    def search(self, key: float) -> bool:
        self.total_queries += 1

        if not self.built:
            self.not_found += 1
            return False

        key = float(key)

        # approximate prediction
        predicted_pos, _ = self.pgm.approximate_rank(key)

        if predicted_pos < len(self.keys) and self.keys[predicted_pos] == key:
            self.correct_predictions += 1
            return True

        # fallback binary search
        idx = self.pgm.bisect_left(key)
        if idx < len(self.keys) and self.keys[idx] == key:
            self.fallbacks += 1
            return True

        self.not_found += 1
        return False

    def get_memory_usage(self) -> int:
        if not self.built:
            return 0
        return self.keys.nbytes + self.pgm.size_in_bytes()
