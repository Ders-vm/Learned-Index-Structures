# src/benchmarks/systematic_overnight_runner.py
# Allow running this file directly in VS Code
import os, sys

# Get project root (folder containing src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import csv
import time
import os
from datetime import datetime
import numpy as np

from src.utils.data_loader import DatasetGenerator
from src.indexes.btree_optimized import BTreeOptimized
from src.indexes.learned_index_optimized import LearnedIndexOptimized
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.ml.shallow_nn_rmi import RecursiveModelIndexNN


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
DISTRIBUTIONS = ["seq", "uniform", "mixed"]

BTREE_ORDERS = [32, 64, 128, 256]

FIXED_WINDOWS = [64, 128, 256, 512, 1024]
ADAPTIVE_Q = [0.99, 0.995, 0.999]
ADAPTIVE_MIN_W = [8, 16, 32]

REPEAT_CYCLES = 3  # how many full cycles to run
RESULTS_DIR = "results/overnight"


# ============================================================
# FOLDER SETUP
# ============================================================

def prepare_folders():
    """Creates results directory and nightly run folder."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    run_name = "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_path, exist_ok=True)

    master_csv = os.path.join(run_path, "master.csv")
    return run_path, master_csv


# ============================================================
# CSV HELPERS
# ============================================================

CSV_HEADER = [
    "timestamp",
    "cycle",
    "dataset_size",
    "distribution",
    "model",
    "params",
    "build_ms",
    "lookup_ns",
    "accuracy",
    "fallback_rate",
    "false_neg",
    "not_found",
    "local_avg_ns",
    "fallback_avg_ns",
    "local_calls",
    "fallback_calls",
    "fit_ms",
    "memory_mb",
]


def write_header_if_missing(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def append_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ============================================================
# GENERATE KEYS
# ============================================================

def generate_keys(size, dist):
    if dist == "seq":
        return DatasetGenerator.generate_sequential(size)
    if dist == "uniform":
        return DatasetGenerator.generate_uniform(size)
    return DatasetGenerator.generate_mixed(size)


# ============================================================
# RUN ONE BENCHMARK
# ============================================================

def run_model(size, dist):

    keys = generate_keys(size, dist)

    # queries
    existing = np.random.choice(keys, 500, replace=False)
    randoms = np.random.uniform(keys.min(), keys.max(), 500)
    queries = np.concatenate([existing, randoms])
    np.random.shuffle(queries)

    results = []

    # ------------------------------------------
    # B-TREE
    # ------------------------------------------
    for order in BTREE_ORDERS:
        model = BTreeOptimized(order=order)

        # build
        t0 = time.perf_counter()
        model.build_from_sorted_array(keys)
        build_ms = (time.perf_counter() - t0) * 1000

        # lookup
        t0 = time.perf_counter()
        for q in queries:
            model.search(q)
        lookup_ns = (time.perf_counter() - t0) * 1e9 / len(queries)

        mem_mb = model.get_memory_usage() / (1024 * 1024)

        results.append((
            "btree",
            f"order={order}",
            build_ms,
            lookup_ns,
            None, None, None, None, None, None, None, None, None,
            mem_mb,
        ))

    # ------------------------------------------
    # FIXED WINDOW LEARNED INDEX
    # ------------------------------------------
    for w in FIXED_WINDOWS:
        model = LearnedIndexOptimized(window=w, use_numpy=True)

        t0 = time.perf_counter()
        model.build_from_sorted_array(keys)
        build_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for q in queries:
            model.search(q, keys)
        lookup_ns = (time.perf_counter() - t0) * 1e9 / len(queries)

        m = model.learned_metrics
        t = model.get_timing_summary()
        mem_mb = model.get_memory_usage() / (1024 * 1024)

        results.append((
            "linear_fixed",
            f"window={w}",
            build_ms,
            lookup_ns,
            m.get_prediction_accuracy(),
            m.get_fallback_rate(),
            m.false_negatives,
            m.not_found,
            t["local_avg_ns"],
            t["fallback_avg_ns"],
            t["local_calls"],
            t["fallback_calls"],
            model.fit_ms,
            mem_mb,
        ))
    # ------------------------------------------
    # PGM INDEX
    # ------------------------------------------
    from src.indexes.pgm_index_wrapper import PGMIndex

    for eps in [16, 32, 64, 128, 256]:
        model = PGMIndex(epsilon=eps)

        # Build time
        t0 = time.perf_counter()
        model.build_from_sorted_array(keys)
        build_ms = (time.perf_counter() - t0) * 1000

        # Lookup time
        t0 = time.perf_counter()
        for q in queries:
            model.search(q)
        lookup_ns = (time.perf_counter() - t0) * 1e9 / len(queries)

        # Metrics
        accuracy = (
            model.correct_predictions / model.total_queries
            if model.total_queries else None
        )
        fallback_rate = (
            model.fallbacks / model.total_queries
            if model.total_queries else None
        )

        mem_mb = model.get_memory_usage() / (1024 * 1024)

        results.append((
            "pgm",
            f"eps={eps}",
            build_ms,
            lookup_ns,
            accuracy,
            fallback_rate,
            model.false_negatives,
            model.not_found,
            None,     # local avg ns (not used by PGM)
            None,     # fallback avg ns
            None,     # local calls
            None,     # fallback calls
            None,     # fit_ms
            mem_mb,
        ))

    # ------------------------------------------
    # ADAPTIVE LEARNED INDEX
    # ------------------------------------------
    for qv in ADAPTIVE_Q:
        for mw in ADAPTIVE_MIN_W:
            model = LinearIndexAdaptive(quantile=qv, min_window=mw, use_numpy=True)

            t0 = time.perf_counter()
            model.build_from_sorted_array(keys)
            build_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            for q in queries:
                model.search(q, keys)
            lookup_ns = (time.perf_counter() - t0) * 1e9 / len(queries)

            m = model.learned_metrics
            t = model.get_timing_summary()
            mem_mb = model.get_memory_usage() / (1024 * 1024)

            results.append((
                "linear_adaptive",
                f"quantile={qv},minW={mw}",
                build_ms,
                lookup_ns,
                m.get_prediction_accuracy(),
                m.get_fallback_rate(),
                m.false_negatives,
                m.not_found,
                t["local_avg_ns"],
                t["fallback_avg_ns"],
                t["local_calls"],
                t["fallback_calls"],
                model.fit_ms,
                mem_mb,
            ))

    return results


# ============================================================
# MAIN SYSTEMATIC LOOP
# ============================================================

def run_systematic():
    run_path, master_csv = prepare_folders()
    write_header_if_missing(master_csv)

    print(f"\n🌙 Systematic Overnight Run Started")
    print(f"📁 Saving results under: {run_path}")
    print(f"📄 Master CSV: {master_csv}\n")

    for cycle in range(1, REPEAT_CYCLES + 1):
        cycle_csv = os.path.join(run_path, f"cycle_{cycle:03}.csv")
        write_header_if_missing(cycle_csv)

        print(f"🔁 Starting cycle {cycle}/{REPEAT_CYCLES}")

        for size in DATASET_SIZES:
            for dist in DISTRIBUTIONS:

                print(f"   → size={size:,}, dist={dist}")

                rows = run_model(size, dist)

                timestamp = datetime.now().isoformat()

                for model_data in rows:
                    row = [timestamp, cycle, size, dist] + list(model_data)

                    append_row(master_csv, row)
                    append_row(cycle_csv, row)

        print(f"💾 Cycle {cycle} complete. Saved → {cycle_csv}\n")

    print("\n🎉 All cycles complete. Overnight run finished.\n")


if __name__ == "__main__":
    run_systematic()
