"""
===============================================================================
ABSTRACT BASE CLASS FOR INDEX STRUCTURES
===============================================================================
Defines a common interface that all index structures must implement.
This ensures consistency across B-Trees, learned indexes, and hybrid structures.
===============================================================================
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional


class IndexStructure(ABC):
    """Abstract base class for all index structures."""
    
    def __init__(self):
        """Initialize common metrics tracking."""
        self.metrics = {
            'total_queries': 0,
            'hits': 0,
            'misses': 0,
            'build_time_ms': 0.0,
        }
    
    @abstractmethod
    def build_from_sorted_array(self, keys: np.ndarray) -> None:
        """
        Build the index structure from sorted keys.
        
        Args:
            keys: Sorted 1D NumPy array of unique keys
            
        Raises:
            TypeError: If keys is not a NumPy array
            ValueError: If keys is not properly formatted
        """
        pass
    
    @abstractmethod
    def search(self, key: float, keys: np.ndarray) -> bool:
        """
        Search for a key in the index.
        
        Args:
            key: The key to search for
            keys: The original sorted key array (for learned indexes)
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Calculate approximate memory usage.
        
        Returns:
            Memory usage in bytes (excluding the keys array itself)
        """
        pass
    
    def validate_input(self, keys: np.ndarray) -> None:
        """
        Validate input keys array.
        
        Args:
            keys: Array to validate
            
        Raises:
            TypeError: If not a NumPy array
            ValueError: If not 1D or not sorted
        """
        if not isinstance(keys, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(keys)}")
        
        if keys.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {keys.shape}")
        
        if len(keys) > 1 and not np.all(keys[:-1] <= keys[1:]):
            raise ValueError("Keys must be sorted in ascending order")
    
    def reset_metrics(self) -> None:
        """Reset all query metrics to zero."""
        for key in self.metrics:
            if isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0
    
    def get_hit_rate(self) -> float:
        """
        Calculate query hit rate.
        
        Returns:
            Fraction of successful queries [0.0, 1.0]
        """
        total = self.metrics['total_queries']
        if total == 0:
            return 0.0
        return self.metrics['hits'] / total


class LearnedIndexMetrics:
    """Extended metrics specific to learned index structures."""
    
    def __init__(self):
        self.total_queries = 0
        self.correct_predictions = 0  # Found within predicted window
        self.fallbacks = 0  # Required full-array search
        self.false_negatives = 0  # Missed by prediction but exists
        self.not_found = 0  # Key doesn't exist
    
    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_queries = 0
        self.correct_predictions = 0
        self.fallbacks = 0
        self.false_negatives = 0
        self.not_found = 0
    
    def get_prediction_accuracy(self) -> float:
        """
        Calculate prediction accuracy rate.
        
        Returns:
            Fraction of queries with correct predictions [0.0, 1.0]
        """
        if self.total_queries == 0:
            return 0.0
        return self.correct_predictions / self.total_queries
    
    def get_fallback_rate(self) -> float:
        """
        Calculate fallback rate.
        
        Returns:
            Fraction of queries requiring full search [0.0, 1.0]
        """
        if self.total_queries == 0:
            return 0.0
        return self.fallbacks / self.total_queries
    
    def to_dict(self) -> Dict[str, float]:
        """Export metrics as dictionary."""
        return {
            'total_queries': self.total_queries,
            'correct_predictions': self.correct_predictions,
            'fallbacks': self.fallbacks,
            'false_negatives': self.false_negatives,
            'not_found': self.not_found,
            'accuracy': self.get_prediction_accuracy(),
            'fallback_rate': self.get_fallback_rate(),
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"Queries: {self.total_queries} | "
            f"Correct: {self.correct_predictions} | "
            f"Fallbacks: {self.fallbacks} | "
            f"Not Found: {self.not_found} | "
            f"Accuracy: {self.get_prediction_accuracy():.2%}"
        )
