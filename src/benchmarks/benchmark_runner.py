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
from src.indexes.btree import BTree
from src.indexes.learned_index import LearnedIndex

class Benchmark:
    """Benchmark tool for B-Tree performance."""

    @staticmethod
    def measure_build_time(index: BTree, keys: np.ndarray) -> float:
        start = time.perf_counter()
        index.build_from_sorted_array(keys)
        end = time.perf_counter()
        return (end - start) * 1000  # ms

    @staticmethod
    def measure_lookup_time(index: BTree, queries: np.ndarray) -> float:
        times = []
        for q in queries:
            start = time.perf_counter()
            index.search(q)
            end = time.perf_counter()
            times.append((end - start) * 1e9)  # ns
        return np.mean(times)

    @staticmethod
    def run(dataset_name: str, keys: np.ndarray, num_queries: int = 1000):
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}  ({len(keys):,} keys)")
        print(f"{'='*70}")

        # Generate random search queries (half existing, half random)
        existing = np.random.choice(keys, num_queries // 2)
        randoms = np.random.uniform(keys.min(), keys.max(), num_queries // 2)
        queries = np.concatenate([existing, randoms])
        np.random.shuffle(queries)

        results = {}

        # ------------------------------------------------------------
        # B-TREE BENCHMARKS
        # ------------------------------------------------------------
        for order in [32, 64, 128, 256]:
            tree = BTree(order=order)
            build = Benchmark.measure_build_time(tree, keys)
            lookup = Benchmark.measure_lookup_time(tree, queries)
            mem = tree.get_memory_usage() / (1024 * 1024)

            print(f"Order {order:<3} | Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")

            results[f"BTree_{order}"] = {
                    "build_ms": build,
                    "lookup_ns": lookup,
                    "memory_mb": mem
            }

        # ------------------------------------------------------------
        # LEARNED INDEX BENCHMARK
        # ------------------------------------------------------------
        print("\n-- Learned Index (Linear Regression) --")

        lm = LearnedIndex()
        build = Benchmark.measure_build_time(lm, keys)
        lookup = Benchmark.measure_lookup_time(lm, queries)
        mem = lm.get_memory_usage() / (1024 * 1024)

        print(f"LinearModel | Build: {build:>8.2f} ms | "
              f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")

        results["LinearModel"] = {
            "build_ms": build,
            "lookup_ns": lookup,
            "memory_mb": mem
        }

        return results