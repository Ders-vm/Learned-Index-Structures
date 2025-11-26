"""
===============================================================================
BENCHMARKING MODULE
===============================================================================
This module measures the performance of different indexing structures.

The Benchmark class provides utilities to:
    • Measure build time (ms)
    • Measure average lookup time (ns)
    • Estimate memory usage (MB)
    • Compare B-Tree performance across different orders (page sizes)

It can be expanded later to include:
    • Learned index models (Linear Regression, Recursive Model Index)
    • CSV result logging
    • Matplotlib visualization for graphs

Usage:
    from src.benchmarks.benchmark_runner import Benchmark
    from src.utils.data_loader import DatasetGenerator

    keys = DatasetGenerator.generate_uniform(10000)
    Benchmark.run("Uniform (10k)", keys)

Next Steps:
    - Add automatic result saving to CSV.
    - Plot comparisons of build/lookup time across datasets.
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
