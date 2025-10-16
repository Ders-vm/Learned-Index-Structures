"""
===============================================================================
BENCHMARKING MODULE — PYTHON vs C++ IMPLEMENTATIONS
===============================================================================
Measures:
    • Build time (ms)
    • Average lookup time (ns)
    • Memory usage (MB)

Automatically benchmarks:
    🐍 Python  → BTree, LearnedIndex, RecursiveModelIndex
    ⚙️  C++     → btree_cpp, linear_model_cpp, rmi_cpp (if compiled)
===============================================================================
"""

import time
import numpy as np
import sys
import os
import inspect

# ----------------------------------------------------------------------------
# Ensure MinGW DLLs are available on Windows for C++ modules
# ----------------------------------------------------------------------------
if sys.platform == "win32":
    mingw_bin = r"C:\mingw64\bin"
    if os.path.exists(mingw_bin):
        os.add_dll_directory(mingw_bin)

# ----------------------------------------------------------------------------
# Import Python implementations
# ----------------------------------------------------------------------------
from src.indexes.python.btree import BTree
from src.indexes.python.learned_index import LearnedIndex
from src.indexes.python.rmi import RecursiveModelIndex

# ----------------------------------------------------------------------------
# Try loading C++ implementations (PyBind11 modules)
# ----------------------------------------------------------------------------
has_btree_cpp = False
has_linear_cpp = False
has_rmi_cpp = False

try:
    from src.indexes.cpp import btree_cpp
    has_btree_cpp = True
    print("✅ Loaded btree_cpp")
except ImportError as e:
    print(f"⚠️  Could not load btree_cpp: {e}")

try:
    from src.indexes.cpp import linear_model_cpp
    has_linear_cpp = True
    print("✅ Loaded linear_model_cpp")
except ImportError as e:
    print(f"⚠️  Could not load linear_model_cpp: {e}")

try:
    from src.indexes.cpp import rmi_cpp
    has_rmi_cpp = True
    print("✅ Loaded rmi_cpp")
except ImportError as e:
    print(f"⚠️  Could not load rmi_cpp: {e}")


# ----------------------------------------------------------------------------
# Benchmark Class
# ----------------------------------------------------------------------------
class Benchmark:
    """Benchmark runner for Python & C++ index structures."""

    # ------------------------------------------------------------------
    @staticmethod
    def measure_build_time(index, keys) -> float:
        """Measure index build time in milliseconds."""
        keys_list = keys.tolist() if isinstance(keys, np.ndarray) else keys
        start = time.perf_counter()
        if hasattr(index, "build_from_sorted_array"):
            index.build_from_sorted_array(keys_list)
        elif hasattr(index, "build"):
            index.build(keys_list)
        end = time.perf_counter()
        return (end - start) * 1000

    # ------------------------------------------------------------------
    @staticmethod
    def measure_lookup_time(index, queries) -> float:
        """Measure average lookup time per query in nanoseconds."""
        if isinstance(queries, np.ndarray):
            queries = queries.tolist()

        times = []
        search_method = getattr(index, "search", None)

        # Detect if the C++ version requires a second arg (like window/safety)
        sig = None
        if search_method:
            try:
                sig = inspect.signature(search_method)
            except (TypeError, ValueError):
                pass

        for q in queries:
            start = time.perf_counter()
            try:
                if sig and len(sig.parameters) > 2:
                    result = index.search(q, 64)  # pass window or safety arg
                else:
                    result = index.search(q)

                if isinstance(result, tuple):
                    found, _ = result
                else:
                    found = result
            except TypeError:
                # fallback: try calling with default window for C++ bindings
                result = index.search(q, 64)
                found = result[0] if isinstance(result, tuple) else result

            end = time.perf_counter()
            times.append((end - start) * 1e9)

        return float(np.mean(times))

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
            lookup = Benchmark.measure_lookup_time(tree, queries)
            mem = tree.get_memory_usage() / (1024 * 1024)
            print(f"  BTree (Order {order:3d}) | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        lm = LearnedIndex()
        build = Benchmark.measure_build_time(lm, keys)
        lookup = Benchmark.measure_lookup_time(lm, queries)
        mem = lm.get_memory_usage() / (1024 * 1024)
        print(f"  LinearModel         | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        rmi = RecursiveModelIndex(fanout=128)
        build = Benchmark.measure_build_time(rmi, keys)
        lookup = Benchmark.measure_lookup_time(rmi, queries)
        mem = rmi.get_memory_usage() / (1024 * 1024)
        print(f"  RMI_2Stage          | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        # ==============================================================
        # ⚙️  C++ IMPLEMENTATIONS (if compiled)
        # ==============================================================
        if not (has_btree_cpp or has_linear_cpp or has_rmi_cpp):
            print("\n⚠️  Skipping C++ benchmarks — none compiled yet.")
            return results

        print("\n⚙️  C++ Implementations:")

        # -- BTree C++ --
        if has_btree_cpp:
            for order in [64, 128]:
                tree = btree_cpp.BTree(order)
                build = Benchmark.measure_build_time(tree, keys)
                lookup = Benchmark.measure_lookup_time(tree, queries)
                mem = tree.get_memory_usage() / (1024 * 1024)
                print(f"  BTree_CPP (Order {order:3d}) | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        # -- LinearModel C++ --
        if has_linear_cpp:
            lm_cpp = linear_model_cpp.LinearModelIndex()
            build = Benchmark.measure_build_time(lm_cpp, keys)
            lookup = Benchmark.measure_lookup_time(lm_cpp, queries)
            mem = lm_cpp.get_memory_usage() / (1024 * 1024)
            print(f"  Linear_CPP          | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        # -- RMI C++ --
        if has_rmi_cpp:
            rmi_cpp_obj = rmi_cpp.RecursiveModelIndex(fanout=128)
            build = Benchmark.measure_build_time(rmi_cpp_obj, keys)
            lookup = Benchmark.measure_lookup_time(rmi_cpp_obj, queries)
            mem = rmi_cpp_obj.get_memory_usage() / (1024 * 1024)
            print(f"  RMI_CPP             | Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")

        return results
