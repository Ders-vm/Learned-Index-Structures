"""
===============================================================================
OPTIMIZED LINEAR MODEL INDEX (LEARNED INDEX)
===============================================================================
Single-stage learned index using linear regression with local binary search.

Key Enhancements:
  • Added use_numpy parameter (searchsorted vs bisect)
  • Added timing breakdown (local vs fallback search)
  • Added timing for polyfit (fit_ms)
  • Added local/fallback call counts
  • Compatible with LinearIndexAdaptive
  • Keeps all original functionality
===============================================================================
"""

import time
import bisect
import numpy as np
from typing import Optional
from src.indexes.index_base import IndexStructure, LearnedIndexMetrics


class LearnedIndexOptimized(IndexStructure):

    DEFAULT_WINDOW = 64
    MIN_WINDOW = 4
    MAX_WINDOW = 100000

    def __init__(self, window: int = DEFAULT_WINDOW, use_numpy: bool = True):
        """
        Args:
            window: Local search window radius
            use_numpy: True = np.searchsorted, False = Python bisect
        """
        super().__init__()

        if not (self.MIN_WINDOW <= window <= self.MAX_WINDOW):
            raise ValueError(f"Window must be in [{self.MIN_WINDOW}, {self.MAX_WINDOW}]")

        self.window = window
        self.use_numpy = use_numpy

        # Linear model params
        self.a = 0.0
        self.b = 0.0

        # Metadata
        self.n = 0
        self.min_key = 0.0
        self.max_key = 0.0

        # Learned-index metrics
        self.learned_metrics = LearnedIndexMetrics()

        # Timing instrumentation
        self.time_local_ns = 0
        self.time_fallback_ns = 0
        self.local_calls = 0
        self.fallback_calls = 0
        self.fit_ms = 0.0

    # ===========================================================================
    # Build
    # ===========================================================================
    def build_from_sorted_array(self, keys: np.ndarray) -> None:
        self.validate_input(keys)
        self.n = len(keys)

        if self.n == 0:
            return

        self.min_key = float(keys[0])
        self.max_key = float(keys[-1])

        positions = np.arange(self.n, dtype=np.float64)

        if self.n == 1:
            self.a = 0
            self.b = 0
        else:
            t0 = time.perf_counter()
            self.a, self.b = np.polyfit(keys, positions, deg=1)
            self.fit_ms = (time.perf_counter() - t0) * 1000

    # ===========================================================================
    # Search
    # ===========================================================================
    def search(self, key: float, keys: np.ndarray) -> bool:
        self.learned_metrics.total_queries += 1
        self.metrics['total_queries'] += 1

        if self.n == 0:
            self.learned_metrics.not_found += 1
            self.metrics['misses'] += 1
            return False

        if key < self.min_key or key > self.max_key:
            self.learned_metrics.not_found += 1
            self.metrics['misses'] += 1
            return False

        # Predict position
        pred = int(self.a * key + self.b)
        pred = np.clip(pred, 0, self.n - 1)

        # Local window bounds
        left = max(0, pred - self.window)
        right = min(self.n, pred + self.window + 1)

        # --------------------------------------------------
        # Local Search
        # --------------------------------------------------
        t0 = time.perf_counter_ns()
        if self.use_numpy:
            idx = np.searchsorted(keys[left:right], key, side='left')
        else:
            idx = bisect.bisect_left(keys[left:right].tolist(), key)
        self.time_local_ns += (time.perf_counter_ns() - t0)
        self.local_calls += 1

        abs_index = idx + left
        found = abs_index < self.n and keys[abs_index] == key

        if found:
            self.learned_metrics.correct_predictions += 1
            self.metrics['hits'] += 1
            return True

        # --------------------------------------------------
        # Fallback Search (only if window didn't cover entire array)
        # --------------------------------------------------
        if left > 0 or right < self.n:
            self.learned_metrics.fallbacks += 1

            t1 = time.perf_counter_ns()
            if self.use_numpy:
                idx2 = np.searchsorted(keys, key, side='left')
            else:
                idx2 = bisect.bisect_left(keys.tolist(), key)
            self.time_fallback_ns += (time.perf_counter_ns() - t1)
            self.fallback_calls += 1

            if idx2 < self.n and keys[idx2] == key:
                self.learned_metrics.false_negatives += 1
                self.metrics['hits'] += 1
                return True

        # Key truly not found
        self.learned_metrics.not_found += 1
        self.metrics['misses'] += 1
        return False

    # ===========================================================================
    # Utilities
    # ===========================================================================
    def get_timing_summary(self):
        """Returns timing breakdown used by benchmark runner."""
        return {
            "fit_ms": self.fit_ms,
            "local_avg_ns": self.time_local_ns / max(1, self.local_calls),
            "fallback_avg_ns": self.time_fallback_ns / max(1, self.fallback_calls),
            "local_calls": self.local_calls,
            "fallback_calls": self.fallback_calls,
        }

    def get_memory_usage(self) -> int:
        return (
            16 +  # a, b
            24 +  # min_key, max_key, n
            8 +   # window
            200   # metrics + overhead
        )

    def reset_metrics(self):
        super().reset_metrics()
        self.learned_metrics.reset()
        self.time_local_ns = 0
        self.time_fallback_ns = 0
        self.local_calls = 0
        self.fallback_calls = 0
        self.fit_ms = 0.0
