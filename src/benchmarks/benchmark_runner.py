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

# ----------------------------------------------------------
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
    def measure_build_time(index, keys, timeout=10.0, verbose=True) -> float:
        """Measure index build time in milliseconds."""
        index_name = type(index).__name__
        if verbose:
            print(f"    → Building {index_name}...", end='', flush=True)
        
        keys_list = keys.tolist() if isinstance(keys, np.ndarray) else keys
        
        start = time.perf_counter()
        try:
            if hasattr(index, "build_from_sorted_array"):
                index.build_from_sorted_array(keys_list)
            elif hasattr(index, "build"):
                index.build(keys_list)
            else:
                print(f" ❌ No build method found!")
                return 0.0
        except Exception as e:
            print(f" ❌ Build failed: {e}")
            return 0.0
        
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        
        if elapsed > timeout * 1000:
            print(f" ⏱️ Build too slow ({elapsed:.2f}ms)")
        elif verbose:
            print(f" ✓ ({elapsed:.2f}ms)", flush=True)
        
        return elapsed

    # ------------------------------------------------------------------
    @staticmethod
    def measure_lookup_time(index, queries, timeout_per_query=0.1, verbose=True) -> float:
        """
        Measure average lookup time per query in nanoseconds.
        
        Args:
            index: The index structure to benchmark
            queries: Array of query values
            timeout_per_query: Maximum time (seconds) to wait per query before skipping
            verbose: Print progress updates
        """
        index_name = type(index).__name__
        if verbose:
            print(f"    → Starting lookup benchmark for {index_name}...", end='', flush=True)
        
        if isinstance(queries, np.ndarray):
            queries = queries.tolist()

        times = []
        search_method = getattr(index, "search", None)
        
        if not search_method:
            print(f" ❌ No search method!")
            return 0.0

        # Limit queries for testing
        test_queries = queries[:min(100, len(queries))]
        
        # Test first query to determine the correct calling convention
        if not test_queries:
            return 0.0
            
        test_query = test_queries[0]
        search_func = None
        
        # Try without extra arg first
        try:
            start = time.perf_counter()
            result = search_method(test_query)
            elapsed = time.perf_counter() - start
            if elapsed > timeout_per_query:
                print(f" ⏱️ First query too slow ({elapsed:.3f}s), skipping")
                return 0.0
            search_func = lambda q: search_method(q)
            if verbose:
                print(f" using search(q)", end='', flush=True)
        except TypeError:
            # Try with extra arg (window/safety parameter)
            try:
                start = time.perf_counter()
                result = search_method(test_query, 64)
                elapsed = time.perf_counter() - start
                if elapsed > timeout_per_query:
                    print(f" ⏱️ First query too slow ({elapsed:.3f}s), skipping")
                    return 0.0
                search_func = lambda q: search_method(q, 64)
                if verbose:
                    print(f" using search(q, 64)", end='', flush=True)
            except Exception as e:
                print(f" ❌ Cannot call search: {e}")
                return 0.0
        except Exception as e:
            print(f" ❌ Error: {e}")
            return 0.0
        
        if not search_func:
            print(f" ❌ Failed to determine search signature")
            return 0.0

        # Now benchmark all test queries
        progress_interval = max(1, len(test_queries) // 10)
        for i, q in enumerate(test_queries):
            try:
                start = time.perf_counter()
                result = search_func(q)
                end = time.perf_counter()
                
                elapsed = end - start
                if elapsed > timeout_per_query:
                    print(f" ⏱️ Query {i+1} timed out ({elapsed:.3f}s), stopping")
                    break
                
                times.append(elapsed * 1e9)
                
                # Progress indicator
                if verbose and i > 0 and i % progress_interval == 0:
                    print(f".", end='', flush=True)
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted at query {i+1}/{len(test_queries)}")
                break
            except Exception as e:
                print(f" ❌ Query {i+1} error: {e}")
                continue

        if verbose:
            print(f" ✓ ({len(times)} queries)")
        
        if not times:
            return 0.0
            
        return float(np.mean(times))

    # ------------------------------------------------------------------
    @staticmethod
    def run(dataset_name: str, keys: np.ndarray, num_queries: int = 100):
        """
        Run benchmarks on all available index implementations.
        
        Args:
            dataset_name: Name of the dataset being benchmarked
            keys: Sorted array of keys to index
            num_queries: Number of queries to test (default: 100 for faster testing)
        """
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}  ({len(keys):,} keys, {num_queries} queries)")
        print(f"{'='*70}")

        # Ensure keys are sorted
        if not np.all(keys[:-1] <= keys[1:]):
            print("⚠️  Keys are not sorted, sorting now...")
            keys = np.sort(keys)

        # Generate queries (half existing, half random)
        print("Generating queries...", end='', flush=True)
        existing = np.random.choice(keys, num_queries // 2, replace=False)
        randoms = np.random.uniform(keys.min(), keys.max(), num_queries // 2)
        queries = np.concatenate([existing, randoms])
        np.random.shuffle(queries)
        print(" ✓")

        results = {}

        # ==============================================================
        # 🐍 PYTHON IMPLEMENTATIONS
        # ==============================================================
        print("\n🐍 Python Implementations:")

        for order in [64, 128]:
            print(f"\n  BTree (Order {order}):")
            try:
                tree = BTree(order=order)
                build = Benchmark.measure_build_time(tree, keys)
                if build > 0:
                    lookup = Benchmark.measure_lookup_time(tree, queries)
                    mem = tree.get_memory_usage() / (1024 * 1024)
                    print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                    results[f"BTree_{order}"] = {"build": build, "lookup": lookup, "memory": mem}
            except KeyboardInterrupt:
                print(f"    ⚠️  Interrupted by user")
                raise
            except Exception as e:
                print(f"    ❌ Failed: {e}")

        print(f"\n  LinearModel:")
        try:
            lm = LearnedIndex()
            build = Benchmark.measure_build_time(lm, keys)
            if build > 0:
                lookup = Benchmark.measure_lookup_time(lm, queries)
                mem = lm.get_memory_usage() / (1024 * 1024)
                print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                results["LinearModel"] = {"build": build, "lookup": lookup, "memory": mem}
        except KeyboardInterrupt:
            print(f"    ⚠️  Interrupted by user")
            raise
        except Exception as e:
            print(f"    ❌ Failed: {e}")

        print(f"\n  RMI_2Stage:")
        try:
            rmi = RecursiveModelIndex(fanout=128)
            build = Benchmark.measure_build_time(rmi, keys)
            if build > 0:
                lookup = Benchmark.measure_lookup_time(rmi, queries)
                mem = rmi.get_memory_usage() / (1024 * 1024)
                print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                results["RMI_2Stage"] = {"build": build, "lookup": lookup, "memory": mem}
        except KeyboardInterrupt:
            print(f"    ⚠️  Interrupted by user")
            raise
        except Exception as e:
            print(f"    ❌ Failed: {e}")

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
                print(f"\n  BTree_CPP (Order {order}):")
                try:
                    tree = btree_cpp.BTree(order)
                    build = Benchmark.measure_build_time(tree, keys)
                    if build > 0:
                        lookup = Benchmark.measure_lookup_time(tree, queries)
                        mem = tree.get_memory_usage() / (1024 * 1024)
                        print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                        results[f"BTree_CPP_{order}"] = {"build": build, "lookup": lookup, "memory": mem}
                except KeyboardInterrupt:
                    print(f"    ⚠️  Interrupted by user")
                    raise
                except Exception as e:
                    print(f"    ❌ Failed: {e}")

        # -- LinearModel C++ --
        if has_linear_cpp:
            print(f"\n  Linear_CPP:")
            try:
                lm_cpp = linear_model_cpp.LinearModelIndex()
                build = Benchmark.measure_build_time(lm_cpp, keys)
                if build > 0:
                    lookup = Benchmark.measure_lookup_time(lm_cpp, queries)
                    mem = lm_cpp.get_memory_usage() / (1024 * 1024)
                    print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                    results["Linear_CPP"] = {"build": build, "lookup": lookup, "memory": mem}
            except KeyboardInterrupt:
                print(f"    ⚠️  Interrupted by user")
                raise
            except Exception as e:
                print(f"    ❌ Failed: {e}")

        # -- RMI C++ --
        if has_rmi_cpp:
            print(f"\n  RMI_CPP:")
            try:
                rmi_cpp_obj = rmi_cpp.RecursiveModelIndex(fanout=128)
                build = Benchmark.measure_build_time(rmi_cpp_obj, keys)
                if build > 0:
                    lookup = Benchmark.measure_lookup_time(rmi_cpp_obj, queries)
                    mem = rmi_cpp_obj.get_memory_usage() / (1024 * 1024)
                    print(f"    ✅ Build: {build:8.2f} ms | Lookup: {lookup:8.2f} ns | Mem: {mem:7.3f} MB")
                    results["RMI_CPP"] = {"build": build, "lookup": lookup, "memory": mem}
            except KeyboardInterrupt:
                print(f"    ⚠️  Interrupted by user")
                raise
            except Exception as e:
                print(f"    ❌ Failed: {e}")

        print(f"\n{'='*70}\n")
        return results