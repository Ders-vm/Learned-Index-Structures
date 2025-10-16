"""
===============================================================================
BENCHMARKING MODULE — PYTHON VS C++ COMPARISON
===============================================================================
This module benchmarks both the Python and C++ implementations of:
    • B-Tree
    • Learned Index
    • Recursive Model Index (RMI)

It will automatically skip any C++ modules that haven’t been built yet.
===============================================================================
"""

import time
import numpy as np

# -------------------------------
# Python Implementations
# -------------------------------
from src.indexes.python.btree import BTree
from src.indexes.python.learned_index import LearnedIndex
from src.indexes.python.rmi import RecursiveModelIndex

# -------------------------------
# Optional C++ Implementations
# -------------------------------
try:
    from src.indexes.cpp import btree_cpp
    has_cpp_btree = True
except ImportError:
    print("⚠️ Could not load C++ BTree module — skipping.")
    has_cpp_btree = False


class Benchmark:
    """Benchmark tool for comparing Python vs C++ index performance."""

    @staticmethod
    def measure_build_time(index, keys: np.ndarray) -> float:
        start = time.perf_counter()
        index.build_from_sorted_array(keys)
        return (time.perf_counter() - start) * 1000  # ms

    @staticmethod
    def measure_lookup_time(index, queries: np.ndarray) -> float:
        times = []
        for q in queries:
            start = time.perf_counter()
            index.search(q)
            times.append((time.perf_counter() - start) * 1e9)  # ns
        return np.mean(times)

    # ------------------------------------------------------------
    # Run benchmark on all available implementations
    # ------------------------------------------------------------
    @staticmethod
    def run(dataset_name: str, keys: np.ndarray, num_queries: int = 1000):
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}  ({len(keys):,} keys)")
        print(f"{'='*70}")

        existing = np.random.choice(keys, num_queries // 2)
        randoms = np.random.uniform(keys.min(), keys.max(), num_queries // 2)
        queries = np.concatenate([existing, randoms])
        np.random.shuffle(queries)

        results = {}

        # ============================================================
        # PYTHON IMPLEMENTATIONS
        # ============================================================
        print("\n🐍 Python Implementations:")

        # --- B-Tree (Python) ---
        for order in [64, 128]:
            tree = BTree(order=order)
            build = Benchmark.measure_build_time(tree, keys)
            lookup = Benchmark.measure_lookup_time(tree, queries)
            mem = tree.get_memory_usage() / (1024 * 1024)
            print(f"  BTree (Order {order}) | Build: {build:>8.2f} ms | Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")
            results[f"Python_BTree_{order}"] = (build, lookup, mem)

        # --- Learned Index (Python) ---
        lm = LearnedIndex()
        build = Benchmark.measure_build_time(lm, keys)
        lookup = Benchmark.measure_lookup_time(lm, queries)
        mem = lm.get_memory_usage() / (1024 * 1024)
        print(f"  LinearModel | Build: {build:>8.2f} ms | Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")

        # --- Recursive Model Index (Python) ---
        rmi = RecursiveModelIndex(fanout=1024)
        build = Benchmark.measure_build_time(rmi, keys)
        lookup = Benchmark.measure_lookup_time(rmi, queries)
        mem = rmi.get_memory_usage() / (1024 * 1024)
        print(f"  RMI_2Stage  | Build: {build:>8.2f} ms | Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")

        # ============================================================
        # C++ IMPLEMENTATIONS
        # ============================================================
        if has_cpp_btree:
            print("\n⚙️  C++ Implementations:")

            for order in [64, 128]:
                tree = btree_cpp.BTree(order=order)
                build = Benchmark.measure_build_time(tree, keys)
                lookup = Benchmark.measure_lookup_time(tree, queries)
                mem = tree.get_memory_usage() / (1024 * 1024)
                print(f"  BTree_CPP (Order {order}) | Build: {build:>8.2f} ms | Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")
                results[f"Cpp_BTree_{order}"] = (build, lookup, mem)

        else:
            print("\n⚠️ Skipping C++ benchmarks — not compiled yet.")

        return results
