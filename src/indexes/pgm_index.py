"""
PGM-Index implementation using pygm library.
"""

import numpy as np
from pygm._pygm import PGMIndexDouble


class PGMIndex:
    """Wrapper for pygm PGM-Index implementation."""
    
    def __init__(self, epsilon: int = 64):
        """Initialize PGM-Index with error bound."""
        self.epsilon = epsilon
        self.pgm = None
        self.keys = None
        self.built = False
        
        # Accuracy tracking
        self.total_queries = 0
        self.correct_predictions = 0
        self.fallbacks = 0
        self.false_negatives = 0
        self.not_found = 0
        self.segments = []
    
    def build_from_sorted_array(self, keys: np.ndarray):
        """Build PGM-Index from sorted keys."""
        self.keys = np.sort(keys).astype(np.float64)
        self.pgm = PGMIndexDouble(iter(self.keys), self.epsilon, False, 0)
        self.segments = [None] * 100  # Placeholder for compatibility
        self.built = True
    
    def search(self, key: float) -> bool:
        """Search for key using pygm PGM-Index."""
        self.total_queries += 1
        
        if not self.built:
            self.not_found += 1
            return False
        
        try:
            # Use pygm's approximate_rank to get predicted position
            rank_result = self.pgm.approximate_rank(float(key))
            predicted_pos = rank_result[0] if isinstance(rank_result, tuple) else rank_result
            
            # Check direct prediction
            if predicted_pos < len(self.keys) and self.keys[predicted_pos] == key:
                self.correct_predictions += 1
                return True
            
            # Use pygm's bisect_left for exact search
            idx = self.pgm.bisect_left(float(key))
            
            if idx < len(self.keys) and self.keys[idx] == key:
                if abs(idx - predicted_pos) <= self.epsilon:
                    self.correct_predictions += 1
                else:
                    self.fallbacks += 1
                return True
            
            self.not_found += 1
            return False
            
        except Exception as e:
            self.not_found += 1
            return False
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        if not self.built:
            return 0
        
        # Keys array + PGM index structure
        keys_size = self.keys.nbytes
        pgm_size = self.pgm.size_in_bytes() if hasattr(self.pgm, 'size_in_bytes') else len(self.segments) * 24
        return keys_size + pgm_size