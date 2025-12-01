"""
===============================================================================
PURE-PYTHON PGM INDEX (Simplified)
===============================================================================
This implementation mimics the behavior of the official PGM-Index but avoids all
native C++ dependencies so it runs on all platforms including Windows.

    

It is optimized for readability and correctness, not SIGMOD-level performance.
===============================================================================
"""

import bisect
import numpy as np


class PGMIndex:
    """
    Pure-Python piecewise geometric model index.
    """

    def __init__(self, epsilon=64):
        self.epsilon = int(epsilon)
        self.keys = None
        self.segments = []   # list of (slope, intercept, start, end)
        self.total_queries = 0
        self.correct_predictions = 0
        self.fallbacks = 0
        self.false_negatives = 0
        self.not_found = 0

    # ----------------------------------------------------------------------
    # BUILD
    # ----------------------------------------------------------------------
    def build_from_sorted_array(self, keys):
        keys = np.asarray(keys)
        self.keys = keys
        n = len(keys)

        if n == 0:
            return

        positions = np.arange(n, dtype=np.float64)

        # Greedy segment builder
        s = 0
        while s < n:
            # Start a new segment
            a, b = np.polyfit([keys[s], keys[min(s+1, n-1)]],
                              [s, min(s+1, n-1)], 1)

            # Expand segment while error ≤ epsilon
            end = s + 1
            while end < n:
                pred = a * keys[end] + b
                if abs(pred - end) > self.epsilon:
                    break
                end += 1

            # Save this segment
            self.segments.append((float(a), float(b), s, end))
            s = end

    # ----------------------------------------------------------------------
    # SEARCH
    # ----------------------------------------------------------------------
    def search(self, key):
        self.total_queries += 1

        if self.keys is None or len(self.keys) == 0:
            self.not_found += 1
            return False

        # Find the segment using binary search over segment starts
        starts = [seg[2] for seg in self.segments]
        idx = bisect.bisect_right(starts, 0)
        seg = None
        for (a, b, s, e) in self.segments:
            if s <= idx < e:
                seg = (a, b, s, e)
                break

        if seg is None:
            # Fallback to global binary search
            self.fallbacks += 1
            i = bisect.bisect_left(self.keys, key)
            if i < len(self.keys) and self.keys[i] == key:
                self.false_negatives += 1
                self.correct_predictions += 1
                return True
            self.not_found += 1
            return False

        a, b, s, e = seg

        # Predict using segment linear model
        pred = int(a * key + b)
        left = max(s, pred - self.epsilon)
        right = min(e, pred + self.epsilon + 1)

        pos = bisect.bisect_left(self.keys[left:right], key)
        global_pos = pos + left

        if global_pos < len(self.keys) and self.keys[global_pos] == key:
            self.correct_predictions += 1
            return True

        # fallback
        self.fallbacks += 1
        i = bisect.bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            self.false_negatives += 1
            self.correct_predictions += 1
            return True

        self.not_found += 1
        return False

    # ----------------------------------------------------------------------
    # MEMORY USAGE
    # ----------------------------------------------------------------------
    def get_memory_usage(self):
        # rough estimate
        return (
            self.keys.nbytes +
            len(self.segments) * 64 +
            256
        )
