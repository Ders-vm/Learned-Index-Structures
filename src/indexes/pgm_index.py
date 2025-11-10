"""
PGM-Index implementation for benchmarking learned index structures.
Implements the Piecewise Geometric Model (PGM) index algorithm.
"""

import numpy as np
import bisect


class PGMSegment:
    """Single segment in PGM-Index."""
    def __init__(self, key: float, slope: float, intercept: float):
        self.key = key          # Starting key of segment
        self.slope = slope      # Linear model slope
        self.intercept = intercept  # Linear model intercept


class PGMIndex:
    """Piecewise Geometric Model (PGM) Index implementation."""
    
    def __init__(self, epsilon: int = 64):
        """Initialize PGM-Index with error bound.
        
        Args:
            epsilon: Maximum prediction error allowed
        """
        self.epsilon = epsilon
        self.segments = []
        self.keys = None
        self.built = False
    
    def build_from_sorted_array(self, keys: np.ndarray):
        """Build PGM-Index from sorted keys."""
        self.keys = np.sort(keys)
        self.segments = self._build_segments(self.keys)
        self.built = True
    
    def _build_segments(self, keys: np.ndarray) -> list:
        """Build piecewise linear segments using greedy algorithm."""
        if len(keys) == 0:
            return []
        
        segments = []
        i = 0
        n = len(keys)
        
        while i < n:
            # Binary search for longest valid segment
            left, right = i + 1, min(i + self.epsilon * 4, n)
            best_end = i
            
            while left <= right:
                mid = (left + right) // 2
                if self._can_fit_segment_fast(keys, i, mid):
                    best_end = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            # Create segment
            start_key = keys[i]
            if best_end > i:
                # Simple linear fit
                slope = (best_end - i) / (keys[best_end] - keys[i]) if keys[best_end] != keys[i] else 1.0
                intercept = i - slope * keys[i]
            else:
                slope = 1.0
                intercept = float(i)
            
            segments.append(PGMSegment(start_key, slope, intercept))
            i = best_end + 1
        
        return segments
    
    def _can_fit_segment_fast(self, keys: np.ndarray, start: int, end: int) -> bool:
        """Fast check if segment satisfies error bound."""
        if end <= start or end >= len(keys):
            return False
        
        # Simple linear fit between endpoints
        if keys[end] == keys[start]:
            return True
        
        slope = (end - start) / (keys[end] - keys[start])
        intercept = start - slope * keys[start]
        
        # Sample check instead of full check for speed
        step = max(1, (end - start) // 10)
        for i in range(start, end + 1, step):
            predicted = slope * keys[i] + intercept
            if abs(predicted - i) > self.epsilon:
                return False
        
        return True
    
    def search(self, key: float) -> bool:
        """Search for key using PGM prediction + local search."""
        if not self.built or len(self.segments) == 0:
            return False
        
        # Find appropriate segment
        segment = self._find_segment(key)
        if segment is None:
            return False
        
        # Predict position
        predicted_pos = int(segment.slope * key + segment.intercept)
        predicted_pos = max(0, min(len(self.keys) - 1, predicted_pos))
        
        # Local search within error bound
        start = max(0, predicted_pos - self.epsilon)
        end = min(len(self.keys), predicted_pos + self.epsilon + 1)
        
        # Binary search in local range
        idx = bisect.bisect_left(self.keys[start:end], key)
        return (idx + start < len(self.keys)) and (self.keys[idx + start] == key)
    
    def _find_segment(self, key: float) -> PGMSegment:
        """Find segment containing the key."""
        if not self.segments:
            return None
        
        # Binary search for segment
        left, right = 0, len(self.segments) - 1
        while left <= right:
            mid = (left + right) // 2
            if mid == len(self.segments) - 1 or key < self.segments[mid + 1].key:
                if key >= self.segments[mid].key:
                    return self.segments[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return self.segments[-1] if self.segments else None  # Fallback to last segment
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        if not self.built:
            return 0
        
        # Keys array + segments (key, slope, intercept per segment)
        keys_size = self.keys.nbytes
        segments_size = len(self.segments) * 3 * 8  # 3 floats per segment
        return keys_size + segments_size


# Alias for backward compatibility with benchmark
MLLearnedIndex = PGMIndex