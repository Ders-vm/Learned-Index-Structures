"""
===============================================================================
BENCHMARKING MODULE — PYTHON vs C++ IMPLEMENTATIONS
===============================================================================
Now with high-precision lookup benchmarking:
  • Uses perf_counter_ns() for nanosecond accuracy
  • Warm-up phase for cache stabilization
  • Multi-round timing with median + standard deviation
  • Batch lookup timing to reduce Python loop overhead
===============================================================================
"""

import time
import numpy as np
import sys
import os
import inspect

# Add MinGW DLL path (for Windows + pybind11)
if sys.platform == "win32":
    mingw_bin = r"C:\mingw64\bin"
    if os.path.exists(mingw_bin):
        os.add_dll_directory(mingw_bin)

# Python implementations
from src.indexes.python.btree import BTree
from src.indexes.python.learned_index import LearnedIndex
from src.indexes.python.rmi import RecursiveModelIndex

# Try importing C++ implementations
has_btree_cpp = has_linear_cpp = has_rmi_cpp = False

try:
    from src.indexes.cpp import btree_cpp
    has_btree_cpp = True
except ImportError:
    print("⚠️  btree_cpp not compiled")

try:
    from src.indexes.cpp import linear_model_cpp
    has_linear_cpp = True
except ImportError:
    print("⚠️  linear_model_cpp not compiled")

try:
    from src.indexes.cpp import rmi_cpp
    has_rmi_cpp = True
except ImportError:
    print("⚠️  rmi_cpp not compiled")


class Benchmark:
    """Benchmark runner for Python & C++ index structures."""

    # ------------------------------------------------------------------
    # Build timing (ms)
    # ------------------------------------------------------------------
    @staticmethod
    def measure_build_time(index, keys) -> float:
        if isinstance(keys, np.ndarray):
            keys = keys.tolist()

        start = time.perf_counter_ns()
        if hasattr(index, "build_from_sorted_array"):
            index.build_from_sorted_array(keys)
        elif hasattr(index, "build"):
            index.build(keys)
        end = time.perf_counter_ns()

        return (end - start) / 1_000_000  # ns → ms

    # ------------------------------------------------------------------
    # Enhanced lookup timing (ns)
    # ------------------------------------------------------------------
    @staticmethod
    def measure_lookup_time(index, queries, warmup=100, rounds=5):
        if isinstance(queries, np.ndarray):
            queries = queries.tolist()

        # --- Warm-up phase (stabilize CPU caches, JIT, etc.)
        for _ in range(min(warmup, len(queries))):
            index.search(np.random.choice(queries))

        round_means = []

        # --- Multiple rounds for consistency
        for _ in range(rounds):
            np.random.shuffle(queries)
            start = time.perf_counter_ns()
            for q in queries:
                try:
                    result = index.search(q)
                except TypeError:
                    # Some models (Linear/RMI) need an extra param
                    result = index.search(q, 10)
            end = time.perf_counter_ns()

            avg_ns = (end - start) / len(queries)
            round_means.append(avg_ns)

        mean_ns = float(np.median(round_means))
        std_ns = float(np.std(round_means))
        return mean_ns, std_ns

    # ------------------------------------------------------------------
    # Main benchmarking routine
    # ------------------------------------------------------------------
    @staticmethod
    def run(dataset_name: str, keys: np.ndarray, num_queries: int = 1000):
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}  ({len(keys):,} keys)")
        print(f"{'='*70}")

        # Generate queries (half existing, half random)
        existing = np.random.choice(keys, num_queries // 2)
        randoms = np.random.uniform(keys.min(), keys.max(), num_queries // 2)
        queries = np.concatenate([existing, randoms])
        np.random.shuffle(queries)

        results = {}

        # ==============================================================
        # 🐍 PYTHON IMPLEMENTATIONS
        # ==============================================================
        print("\n🐍 Python Implementations:")
        for order in [64, 128]:
            tree = BTree(order=order)
            build = Benchmark.measure_build_time(tree, keys)
            lookup_mean, lookup_std = Benchmark.measure_lookup_time(tree, queries)
            mem = tree.get_memory_usage() / (1024 * 1024)
            print(f"  BTree (Order {order:3d}) | Build: {build:8.2f} ms | "
                  f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        lm = LearnedIndex()
        build = Benchmark.measure_build_time(lm, keys)
        lookup_mean, lookup_std = Benchmark.measure_lookup_time(lm, queries)
        mem = lm.get_memory_usage() / (1024 * 1024)
        print(f"  LinearModel         | Build: {build:8.2f} ms | "
              f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        rmi = RecursiveModelIndex(fanout=128)
        build = Benchmark.measure_build_time(rmi, keys)
        lookup_mean, lookup_std = Benchmark.measure_lookup_time(rmi, queries)
        mem = rmi.get_memory_usage() / (1024 * 1024)
        print(f"  RMI_2Stage          | Build: {build:8.2f} ms | "
              f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        # ==============================================================
        # ⚙️  C++ IMPLEMENTATIONS (if compiled)
        # ==============================================================
        if not (has_btree_cpp or has_linear_cpp or has_rmi_cpp):
            print("\n⚠️  Skipping C++ benchmarks — none compiled yet.")
            return results

        print("\n⚙️  C++ Implementations:")
        if has_btree_cpp:
            for order in [64, 128]:
                tree = btree_cpp.BTree(order)
                build = Benchmark.measure_build_time(tree, keys)
                lookup_mean, lookup_std = Benchmark.measure_lookup_time(tree, queries)
                mem = tree.get_memory_usage() / (1024 * 1024)
                print(f"  BTree_CPP (Order {order:3d}) | Build: {build:8.2f} ms | "
                      f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        if has_linear_cpp:
            lm_cpp = linear_model_cpp.LinearModelIndex()
            build = Benchmark.measure_build_time(lm_cpp, keys)
            lookup_mean, lookup_std = Benchmark.measure_lookup_time(lm_cpp, queries)
            mem = lm_cpp.get_memory_usage() / (1024 * 1024)
            print(f"  Linear_CPP          | Build: {build:8.2f} ms | "
                  f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        if has_rmi_cpp:
            rmi_cpp_obj = rmi_cpp.RecursiveModelIndex(fanout=128)
            build = Benchmark.measure_build_time(rmi_cpp_obj, keys)
            lookup_mean, lookup_std = Benchmark.measure_lookup_time(rmi_cpp_obj, queries)
            mem = rmi_cpp_obj.get_memory_usage() / (1024 * 1024)
            print(f"  RMI_CPP             | Build: {build:8.2f} ms | "
                  f"Lookup: {lookup_mean:8.2f} ± {lookup_std:6.2f} ns | Mem: {mem:7.3f} MB")

        return results
