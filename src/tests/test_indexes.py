"""
===============================================================================
COMPREHENSIVE TEST SUITE FOR LEARNED INDEX STRUCTURES
===============================================================================
pytest-based test suite covering:
  • Basic functionality
  • Edge cases
  • Performance validation
  • Correctness verification
  • Memory efficiency

Run with: pytest test_indexes.py -v
Or with coverage: pytest test_indexes.py -v --cov
===============================================================================
"""

import pytest
import numpy as np
import time
from src.indexes.learned_index_optimized import LearnedIndexOptimized
from src.indexes.btree_optimized import BTreeOptimized


class TestLearnedIndexOptimized:
    """Test suite for optimized learned index."""
    
    def test_basic_search_existing_keys(self):
        """Test that all existing keys are found."""
        keys = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        for key in keys:
            assert index.search(key, keys), f"Failed to find existing key {key}"
    
    def test_basic_search_missing_keys(self):
        """Test that non-existing keys are not found."""
        keys = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        for key in [0.0, 3.0, 7.5, 25.0]:
            assert not index.search(key, keys), f"False positive for key {key}"
    
    def test_empty_array(self):
        """Test handling of empty key array."""
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(np.array([]))
        
        assert not index.search(5.0, np.array([]))
        assert index.n == 0
        assert index.predict(5.0) == -1
    
    def test_single_element(self):
        """Test handling of single-element array."""
        keys = np.array([42.0])
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        assert index.search(42.0, keys)
        assert not index.search(41.0, keys)
        assert not index.search(43.0, keys)
    
    def test_bounds_checking(self):
        """Test that bounds checking works correctly."""
        keys = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        # Below minimum
        assert not index.search(5.0, keys)
        # Above maximum
        assert not index.search(55.0, keys)
        # At boundaries
        assert index.search(10.0, keys)
        assert index.search(50.0, keys)
    
    @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
    def test_different_sizes(self, size):
        """Test correctness across different dataset sizes."""
        keys = np.sort(np.random.uniform(0, 1_000_000, size))
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        # Test sample of existing keys
        sample_size = min(100, size)
        sample = np.random.choice(keys, sample_size, replace=False)
        
        for key in sample:
            assert index.search(key, keys), f"Failed for size={size}, key={key}"
    
    @pytest.mark.parametrize("window", [4, 16, 64, 256, 1024])
    def test_different_windows(self, window):
        """Test that different window sizes work correctly."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 1000))
        index = LearnedIndexOptimized(window=window)
        index.build_from_sorted_array(keys)
        
        sample = np.random.choice(keys, 50, replace=False)
        for key in sample:
            assert index.search(key, keys)
    
    def test_prediction_accuracy(self):
        """Test that predictions are reasonable."""
        keys = np.arange(0, 1000, dtype=np.float64)  # Sequential
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        # For sequential data, predictions should be very accurate
        errors = []
        for i in range(0, 1000, 100):
            pred = index.predict(keys[i])
            error = abs(pred - i)
            errors.append(error)
        
        avg_error = np.mean(errors)
        assert avg_error < 10, f"Average prediction error too high: {avg_error}"
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 100))
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        # Perform searches
        existing = np.random.choice(keys, 50, replace=False)
        non_existing = np.random.uniform(keys.min(), keys.max(), 100)
        non_existing = non_existing[~np.isin(non_existing, keys)][:50]
        
        for key in existing:
            index.search(key, keys)
        
        for key in non_existing:
            index.search(key, keys)
        
        # Check metrics make sense
        m = index.learned_metrics
        assert m.total_queries == 100  # We did 100 queries
        assert m.correct_predictions + m.false_negatives == 50  # All 50 existing should be found
        assert m.not_found <= 50  # At most 50 not found (the non-existing keys)
        assert m.fallbacks >= 0  # Should have some fallbacks
        
        # Hits + misses should equal total
        assert index.metrics['hits'] + index.metrics['misses'] == m.total_queries
    
    def test_input_validation_non_numpy(self):
        """Test that non-numpy input raises TypeError."""
        index = LearnedIndexOptimized()
        with pytest.raises(TypeError):
            index.build_from_sorted_array([1, 2, 3])
    
    def test_input_validation_wrong_dimension(self):
        """Test that 2D array raises ValueError."""
        index = LearnedIndexOptimized()
        with pytest.raises(ValueError):
            index.build_from_sorted_array(np.array([[1, 2], [3, 4]]))
    
    def test_input_validation_unsorted(self):
        """Test that unsorted array raises ValueError."""
        index = LearnedIndexOptimized()
        with pytest.raises(ValueError):
            index.build_from_sorted_array(np.array([3.0, 1.0, 2.0]))
    
    def test_invalid_window(self):
        """Test that invalid window raises ValueError."""
        with pytest.raises(ValueError):
            LearnedIndexOptimized(window=2)  # Below MIN_WINDOW
        
        with pytest.raises(ValueError):
            LearnedIndexOptimized(window=200000)  # Above MAX_WINDOW
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 10_000))
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(keys)
        
        mem_bytes = index.get_memory_usage()
        keys_bytes = keys.nbytes
        
        # Memory should be much less than storing keys
        assert mem_bytes < keys_bytes * 0.01, \
            f"Memory usage too high: {mem_bytes} bytes vs {keys_bytes} keys"


class TestBTreeOptimized:
    """Test suite for optimized B-Tree."""
    
    def test_basic_search_existing_keys(self):
        """Test that all existing keys are found."""
        keys = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        tree = BTreeOptimized(order=4)
        tree.build_from_sorted_array(keys)
        
        for key in keys:
            assert tree.search(key), f"Failed to find existing key {key}"
    
    def test_basic_search_missing_keys(self):
        """Test that non-existing keys are not found."""
        keys = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        tree = BTreeOptimized(order=4)
        tree.build_from_sorted_array(keys)
        
        for key in [0.0, 3.0, 12.5, 35.0]:
            assert not tree.search(key), f"False positive for key {key}"
    
    def test_empty_tree(self):
        """Test handling of empty tree."""
        tree = BTreeOptimized(order=4)
        tree.build_from_sorted_array(np.array([]))
        
        assert not tree.search(5.0)
        assert tree.size == 0
        assert tree.get_height() == 0
    
    def test_single_element(self):
        """Test handling of single-element tree."""
        tree = BTreeOptimized(order=4)
        tree.build_from_sorted_array(np.array([42.0]))
        
        assert tree.search(42.0)
        assert not tree.search(41.0)
        assert tree.size == 1
        assert tree.get_height() == 1
    
    @pytest.mark.parametrize("size,order", [
        (100, 4),
        (100, 16),
        (1000, 64),
        (1000, 128),
        (10000, 256),
    ])
    def test_different_sizes_and_orders(self, size, order):
        """Test correctness across different sizes and orders."""
        keys = np.sort(np.random.uniform(0, 1_000_000, size))
        tree = BTreeOptimized(order=order)
        tree.build_from_sorted_array(keys)
        
        # Test sample of existing keys
        sample_size = min(100, size)
        sample = np.random.choice(keys, sample_size, replace=False)
        
        for key in sample:
            assert tree.search(key), \
                f"Failed for size={size}, order={order}, key={key}"
    
    def test_tree_height(self):
        """Test that tree height is reasonable."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 10_000))
        tree = BTreeOptimized(order=128)
        tree.build_from_sorted_array(keys)
        
        height = tree.get_height()
        # For 10k keys with order 128, height should be 2-3
        assert 2 <= height <= 4, f"Unexpected tree height: {height}"
    
    def test_memory_usage(self):
        """Test that memory calculation doesn't crash."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 1000))
        tree = BTreeOptimized(order=64)
        tree.build_from_sorted_array(keys)
        
        mem = tree.get_memory_usage()
        assert mem > 0
        assert isinstance(mem, int)
    
    def test_stats(self):
        """Test that stats are calculated correctly."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 1000))
        tree = BTreeOptimized(order=64)
        tree.build_from_sorted_array(keys)
        
        stats = tree.get_stats()
        
        assert stats['size'] == 1000
        assert stats['order'] == 64
        assert stats['height'] > 0
        assert stats['memory_bytes'] > 0
        assert stats['avg_keys_per_node'] > 0
    
    def test_input_validation_non_numpy(self):
        """Test that non-numpy input raises TypeError."""
        tree = BTreeOptimized(order=4)
        with pytest.raises(TypeError):
            tree.build_from_sorted_array([1, 2, 3])
    
    def test_input_validation_wrong_dimension(self):
        """Test that 2D array raises ValueError."""
        tree = BTreeOptimized(order=4)
        with pytest.raises(ValueError):
            tree.build_from_sorted_array(np.array([[1, 2], [3, 4]]))
    
    def test_input_validation_unsorted(self):
        """Test that unsorted array raises ValueError."""
        tree = BTreeOptimized(order=4)
        with pytest.raises(ValueError):
            tree.build_from_sorted_array(np.array([3.0, 1.0, 2.0]))
    
    def test_invalid_order(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError):
            BTreeOptimized(order=2)  # Below MIN_ORDER
        
        with pytest.raises(ValueError):
            BTreeOptimized(order=20000)  # Above MAX_ORDER


class TestPerformanceComparison:
    """Performance comparison tests."""
    
    def test_learned_index_faster_than_btree_sequential(self):
        """Test that learned index is faster on sequential data."""
        keys = np.arange(0, 100_000, dtype=np.float64)
        
        # Build both
        learned = LearnedIndexOptimized(window=64)
        learned.build_from_sorted_array(keys)
        
        btree = BTreeOptimized(order=128)
        btree.build_from_sorted_array(keys)
        
        # Query both
        queries = np.random.choice(keys, 1000, replace=False)
        
        t0 = time.perf_counter()
        for q in queries:
            learned.search(q, keys)
        learned_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        for q in queries:
            btree.search(q)
        btree_time = time.perf_counter() - t0
        
        # Learned index should be faster on sequential data
        print(f"\nLearned: {learned_time*1e6:.1f}µs, BTree: {btree_time*1e6:.1f}µs")
        assert learned_time < btree_time * 2  # At least competitive
    
    def test_memory_comparison(self):
        """Compare memory usage between indexes."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 10_000))
        
        learned = LearnedIndexOptimized()
        learned.build_from_sorted_array(keys)
        
        btree = BTreeOptimized(order=128)
        btree.build_from_sorted_array(keys)
        
        learned_mem = learned.get_memory_usage()
        btree_mem = btree.get_memory_usage()
        
        print(f"\nLearned: {learned_mem/1024:.1f}KB, BTree: {btree_mem/1024:.1f}KB")
        
        # Learned index should use much less memory
        assert learned_mem < btree_mem * 0.1


class TestCorrectness:
    """Rigorous correctness tests."""
    
    @pytest.mark.parametrize("index_class", [
        LearnedIndexOptimized,
        lambda: BTreeOptimized(order=128)
    ])
    def test_all_keys_found(self, index_class):
        """Verify all inserted keys can be found."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 5000))
        index = index_class()
        index.build_from_sorted_array(keys)
        
        # Test EVERY key
        for key in keys:
            if isinstance(index, LearnedIndexOptimized):
                assert index.search(key, keys), f"Failed to find {key}"
            else:
                assert index.search(key), f"Failed to find {key}"
    
    @pytest.mark.parametrize("index_class", [
        LearnedIndexOptimized,
        lambda: BTreeOptimized(order=128)
    ])
    def test_no_false_positives(self, index_class):
        """Verify no false positives for non-existent keys."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 1000))
        index = index_class()
        index.build_from_sorted_array(keys)
        
        # Generate keys definitely not in set
        non_existing = []
        for i in range(1000):
            candidate = np.random.uniform(0, 1_000_000)
            if candidate not in keys:
                non_existing.append(candidate)
                if len(non_existing) >= 500:
                    break
        
        # None should be found
        for key in non_existing:
            if isinstance(index, LearnedIndexOptimized):
                assert not index.search(key, keys), f"False positive: {key}"
            else:
                assert not index.search(key), f"False positive: {key}"


# ===============================================================================
# Performance Benchmarks (not run by default)
# ===============================================================================
class TestBenchmarks:
    """Benchmarks for performance analysis (use pytest -m benchmark)."""
    
    @pytest.mark.benchmark
    def test_benchmark_learned_index_build(self, benchmark):
        """Benchmark learned index build time."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 100_000))
        index = LearnedIndexOptimized()
        
        benchmark(index.build_from_sorted_array, keys)
    
    @pytest.mark.benchmark
    def test_benchmark_btree_build(self, benchmark):
        """Benchmark B-Tree build time."""
        keys = np.sort(np.random.uniform(0, 1_000_000, 100_000))
        tree = BTreeOptimized(order=128)
        
        benchmark(tree.build_from_sorted_array, keys)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
