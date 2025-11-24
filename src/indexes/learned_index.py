"""
===============================================================================
LEARNED INDEX - BACKWARD COMPATIBILITY WRAPPER
===============================================================================
This is a backward compatibility wrapper that maps to LearnedIndexOptimized.

The original LearnedIndex implementation has been replaced with the optimized
version. This file exists solely to support legacy plot/visualization code.

For new code, import LearnedIndexOptimized directly:
    from src.indexes.learned_index_optimized import LearnedIndexOptimized

===============================================================================
"""

from src.indexes.learned_index_optimized import LearnedIndexOptimized


class LearnedIndex(LearnedIndexOptimized):
    """
    Backward compatibility wrapper for LearnedIndexOptimized.
    
    This class simply inherits all functionality from LearnedIndexOptimized
    with no modifications. It exists to support legacy code that imports
    'LearnedIndex' instead of 'LearnedIndexOptimized'.
    
    Example:
        # Legacy code (now supported):
        from src.indexes.learned_index import LearnedIndex
        index = LearnedIndex()
        
        # Preferred for new code:
        from src.indexes.learned_index_optimized import LearnedIndexOptimized
        index = LearnedIndexOptimized()
    """
    
    def __init__(self, window: int = 128, use_numpy: bool = True):
        """
        Initialize learned index.
        
        Args:
            window: Search window size (default: 128)
            use_numpy: Use NumPy for binary search (default: True, 44x faster)
        """
        super().__init__(window=window, use_numpy=use_numpy)


# For direct script execution or testing
if __name__ == "__main__":
    import numpy as np
    
    print("Testing LearnedIndex backward compatibility wrapper...")
    
    # Generate test data
    keys = np.sort(np.random.uniform(0, 1_000_000, 10_000))
    
    # Test the wrapper
    index = LearnedIndex()
    index.build_from_sorted_array(keys)
    
    # Verify search works
    test_key = keys[5000]
    found = index.search(test_key, keys)
    
    print(f"✓ Created index with {len(keys)} keys")
    print(f"✓ Search for existing key: {found}")
    print(f"✓ Memory usage: {index.get_memory_usage() / 1024:.2f} KB")
    print("\nBackward compatibility wrapper working correctly!")
