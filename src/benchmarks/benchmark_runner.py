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
from src.indexes.rmi import RecursiveModelIndex
from src.ml.shallow_nn_rmi import RecursiveModelIndexNN
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.indexes.pgm_index import PGMIndex
class Benchmark:
    """Benchmark tool for B-Tree performance."""

    @staticmethod
    def measure_build_time(index, keys: np.ndarray) -> float:
        start = time.perf_counter()
        index.build_from_sorted_array(keys)
        end = time.perf_counter()
        return (end - start) * 1000  # ms

    @staticmethod
    def measure_lookup_time(index, queries: np.ndarray) -> float:
        # Warmup
        for q in queries[:50]:
            index.search(q)
        start = time.perf_counter()
        for q in queries:
            index.search(q)  # No timing overhead per call
        end = time.perf_counter()
        total_time = (end - start) * 1e9 / len(queries)  # Convert to ns per query
        return total_time

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
        print("\n-- B-Tree Benchmarks --")
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

        for errorWindow in [128, 512, 1024, 2048, 4096, 8192, 16384, 100000]:
            lm = LearnedIndex()
            lm.window = errorWindow
            build = Benchmark.measure_build_time(lm, keys)
            lookup = Benchmark.measure_lookup_time(lm, queries)
            total_queries = lm.total_queries
            correct_predictions = lm.correct_predictions
            fallbacks = lm.fallbacks
            false_negatives = lm.false_negatives
            not_found = lm.not_found
            mem = lm.get_memory_usage() / (1024 * 1024)

            print(f"Window {errorWindow:<4} | Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB | "
                  f"Correct: {correct_predictions}/500 | "                # only half of queries are existing keys
                  f"Fallbacks: {fallbacks} | Not Found: {not_found} | "   # shows how many fallbacks and how many were not found
                  f"False Negatives: {false_negatives}")                  # how many times a key was present but not predicted correctly


        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # LEARNED INDEX (ADAPTIVE)
        # ------------------------------------------------------------
        print("\n-- Learned Index (Adaptive) --")

        adaptive_cfgs = [
            (64,  0.990, 4),
            (128, 0.995, 4),
            (256, 0.995, 4),
            (512, 0.995, 4),
            (512, 0.999, 4),
            (512, 1, 4),
        ]

        for bins, q, min_w in adaptive_cfgs:
            lai = LinearIndexAdaptive(bins=bins, quantile=q, min_window=min_w)
            build = Benchmark.measure_build_time(lai, keys)
            lookup = Benchmark.measure_lookup_time(lai, queries)
            total_queries = lai.total_queries
            correct_predictions = lai.correct_predictions
            fallbacks = lai.fallbacks
            false_negatives = lai.false_negatives
            not_found = lai.not_found
            mem = lai.get_memory_usage() / (1024 * 1024)

            print(f"bins={bins:<4} q={q:<6} minW={min_w:<2} | "
                  f"Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | "
                  f"Mem: {mem:>6.3f} MB | "
                  f"Correct: {correct_predictions}/500 | "
                  f"Fallbacks: {fallbacks} | Not Found: {not_found} | "
                  f"False Negatives: {false_negatives}")



        # ------------------------------------------------------------
        # TWO-STAGE RMI BENCHMARK
        # ------------------------------------------------------------
        print("\n-- Two-Stage RMI --")
        rmi = RecursiveModelIndex(fanout=8192)  # 128 leaf models (change to test different results)
        build = Benchmark.measure_build_time(rmi, keys)
        lookup = Benchmark.measure_lookup_time(rmi, queries)
        mem = rmi.get_memory_usage() / (1024 * 1024)
        print(
            f"RMI_2Stage  | Build: {build:>8.2f} ms | "
            f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB"
        )
        

        # ------------------------------------------------------------
        # TWO-STAGE RMI (SHALLOW NN ROOT)
        # ------------------------------------------------------------
        print("\n-- Two-Stage RMI (Shallow NN Root) --")
        rmi_nn = RecursiveModelIndexNN(fanout=8192, hidden_dim=16, epochs=100, lr=0.01)
        build = Benchmark.measure_build_time(rmi_nn, keys)
        lookup = Benchmark.measure_lookup_time(rmi_nn, queries)
        mem = rmi_nn.get_memory_usage() / (1024 * 1024)
        print(f"RMI_NNRoot | Build: {build:>8.2f} ms | "
              f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB")
        
        # ------------------------------------------------------------
        # PGM-INDEX BENCHMARK
        # ------------------------------------------------------------
        print("\n-- PGM-Index --")
        for epsilon in [32, 64, 128, 256, 512]:
            pgm_index = PGMIndex(epsilon=epsilon)
            build = Benchmark.measure_build_time(pgm_index, keys)
            lookup = Benchmark.measure_lookup_time(pgm_index, queries)
            total_queries = pgm_index.total_queries
            correct_predictions = pgm_index.correct_predictions
            fallbacks = pgm_index.fallbacks
            false_negatives = pgm_index.false_negatives
            not_found = pgm_index.not_found
            mem = pgm_index.get_memory_usage() / (1024 * 1024)
            segments = len(pgm_index.segments) if pgm_index.built else 0
            print(f"ε={epsilon:<3} | Build: {build:>8.2f} ms | "
                  f"Lookup: {lookup:>8.2f} ns | Mem: {mem:>6.3f} MB | "
                  f"Correct: {correct_predictions}/500 | "
                  f"Fallbacks: {fallbacks} | Not Found: {not_found} | "
                  f"False Negatives: {false_negatives} | Segments: {segments}")
        
        return results

if __name__ == "__main__":
    n = 50_000  # number of keys to test
    keys = np.sort(np.random.uniform(0, 1_000_000, n))
    Benchmark.run("Uniform_50k", keys)
