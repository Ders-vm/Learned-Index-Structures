"""
===============================================================================
QUICK BENCHMARK UTILITY - Single Dataset Testing
===============================================================================
This is a STANDALONE utility for quick benchmarking of a single dataset.

PURPOSE:
    Fast, simple benchmarking for quick tests without the full benchmark suite.
    Useful for development, debugging, and quick performance checks.

RELATIONSHIP TO MAIN WORKFLOW:
    - Main workflow: benchmarks/benchmark.py (comprehensive, multiple datasets)
    - This utility: Quick single-dataset tests (fast, simple)

WHAT IT DOES:
    • Measure build time (ms)
    • Measure average lookup time (ns)
    • Estimate memory usage (MB)
    • Compare B-Tree performance across different orders (page sizes)
    • Compare Learned Index across different window sizes

USAGE:
    from src.benchmarks.benchmark_single import Benchmark
    from src.utils.data_loader import DatasetGenerator

    # Generate data
    keys = DatasetGenerator.generate_uniform(10000)
    
    # Run quick benchmark
    Benchmark.run("Uniform (10k)", keys)

OUTPUT EXAMPLE:
    ======================================================================
    Dataset: Uniform (10k) — 10,000 keys
    ======================================================================

    -- B-Tree (Optimized) --
    Order 32  | Build:     2.34 ms | Lookup:    45.32 ns | Mem:  0.156 MB
    Order 64  | Build:     1.89 ms | Lookup:    38.21 ns | Mem:  0.203 MB
    ...

    -- Learned Index (Optimized) --
    Window 64   | Build:     0.52 ms | Lookup:    12.45 ns | Mem:  0.001 MB
    Window 128  | Build:     0.51 ms | Lookup:    14.23 ns | Mem:  0.001 MB
    ...

WHEN TO USE:
    ✓ Quick tests during development
    ✓ Debugging specific configurations
    ✓ Testing on custom datasets
    ✓ Interactive experimentation
    
    ✗ DON'T use for research/publications (use benchmark.py instead)
    ✗ DON'T use for statistical analysis (no repetition/cycles)

COMPARISON TO FULL BENCHMARK:
    benchmark.py:
        - Tests 5 dataset sizes × 3 distributions × 5 cycles
        - ~30 model configurations
        - Outputs detailed CSV for analysis
        - Takes 5-10 minutes
        - Statistical validity
    
    benchmark_single.py (this file):
        - Tests 1 dataset × 1 run
        - Quick console output
        - Takes seconds
        - Good for development

===============================================================================
"""

import time
import numpy as np

from src.indexes.btree_optimized import BTreeOptimized
from src.indexes.learned_index_optimized import LearnedIndexOptimized


class Benchmark:
    """Benchmark tool for B-Tree + Learned Index performance."""

    @staticmethod
    def measure_build_time(index, keys):
        t0 = time.perf_counter()
        index.build_from_sorted_array(keys)
        return (time.perf_counter() - t0) * 1000  # ms

    @staticmethod
    def measure_lookup_time(index, queries, keys=None):
        # warm-up
        for q in queries[:20]:
            index.search(q, keys) if keys is not None else index.search(q)

        t0 = time.perf_counter()
        for q in queries:
            index.search(q, keys) if keys is not None else index.search(q)
        return (time.perf_counter() - t0) * 1e9 / len(queries)  # ns/query

    @staticmethod
    def run(name: str, keys: np.ndarray, num_queries=1000):
        print(f"\n{'='*70}")
        print(f"Dataset: {name} — {len(keys):,} keys")
        print(f"{'='*70}")

        # Generate query set
        existing = np.random.choice(keys, num_queries // 2, replace=False)
        randoms = np.random.uniform(keys.min(), keys.max(), num_queries // 2)
        queries = np.concatenate([existing, randoms])
        np.random.shuffle(queries)

        # -------------------------
        # B-TREE BENCHMARKS
        # -------------------------
        print("\n-- B-Tree (Optimized) --")

        for order in [32, 64, 128, 256]:
            tree = BTreeOptimized(order=order)
            build = Benchmark.measure_build_time(tree, keys)
            lookup = Benchmark.measure_lookup_time(tree, queries)
            mem = tree.get_memory_usage() / (1024 * 1024)

            print(f"Order {order:<3} | Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")

        # -------------------------
        # LEARNED INDEX BENCHMARKS
        # -------------------------
        print("\n-- Learned Index (Optimized) --")

        for window in [64, 128, 512, 1024, 4096]:
            li = LearnedIndexOptimized(window=window)
            build = Benchmark.measure_build_time(li, keys)
            lookup = Benchmark.measure_lookup_time(li, queries, keys)
            mem = li.get_memory_usage() / (1024 * 1024)

            print(f"Window {window:<5} | Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")
