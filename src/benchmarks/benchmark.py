"""
================================================================================
BENCHMARK RUNNER - Learned Index Structures
================================================================================

PURPOSE:
    Comprehensive benchmarking suite that compares your learned index 
    implementations against the Kraska et al. baseline and traditional indexes.

WHAT IT DOES:
    1. Tests multiple index types across various dataset sizes and distributions
    2. Measures lookup time, build time, memory usage, and accuracy
    3. Runs multiple cycles for statistical validity
    4. Outputs results to CSV for analysis and graphing

MODELS TESTED:
    - Your implementations: Linear Fixed, Linear Adaptive
    - Kraska baseline: Single-stage, RMI (Recursive Model Index)
    - Traditional: B-Tree
    
HOW IT WORKS:
    For each (dataset_size, distribution, model, configuration) combination:
        1. Generate dataset (sorted keys)
        2. Build index
        3. Run warmup queries (eliminates cold cache)
        4. Run test queries (500 existing + 500 random)
        5. Record all metrics
        6. Repeat for statistical confidence
    
OUTPUT:
    results/benchmarks/run_YYYY-MM-DD_HH-MM-SS/master.csv
    
TYPICAL RUNTIME:
    5-10 minutes for full benchmark (5 sizes × 3 distributions × ~30 configs × 5 cycles)

USAGE:
    python benchmarks/run_benchmarks.py
    
    Then generate graphs with:
    python benchmarks/generate_graphs.py

AUTHOR: Based on Kraska et al. "The Case for Learned Index Structures" (SIGMOD 2018)
================================================================================
"""

import os
import sys

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import time
import numpy as np
from datetime import datetime

# Import data generator and index implementations
from src.utils.data_loader import DatasetGenerator
from src.indexes.btree_optimized import BTreeOptimized
from src.indexes.learned_index_optimized import LearnedIndexOptimized
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.indexes.learned_index_kraska import SingleStageLearnedIndex, RecursiveModelIndex


# ============================================================================
# CONFIGURATION
# ============================================================================
"""
Benchmark parameters that define what gets tested.

DATASET_SIZES: Number of keys to test with (10K to 1M)
DISTRIBUTIONS: Data patterns (sequential, uniform random, mixed)
REPEAT_CYCLES: How many times to repeat each test (5 = good statistical power)

Testing across multiple sizes shows:
- Small (10K-100K): Where B-Trees excel (low log(n))
- Medium (500K): Transition point
- Large (1M): Where learned indexes shine (O(1) advantage)

Model-specific parameters follow.
"""

DATASET_SIZES = [
    10_000,       # 10K
    50_000,       # 50K
    100_000,      # 100K
    500_000,      # 500K
    1_000_000,    # 1M
    5_000_000,    # 5M
    10_000_000,   # 10M
    50_000_000,   # 50M
    100_000_000   # 100M
]
DISTRIBUTIONS = ["seq", "uniform", "mixed"]

# B-Tree configurations (order = fanout)
BTREE_ORDERS = [32, 64, 128, 256]

# Linear Fixed: Different window sizes to test
FIXED_WINDOWS = [64, 128, 256, 512, 1024]

# Linear Adaptive: Quantile and minimum window combinations
ADAPTIVE_Q = [0.99, 0.995, 0.999]
ADAPTIVE_MIN_W = [8, 16, 32]

# Kraska Single-Stage: Model types
KRASKA_SINGLE_MODELS = ['linear', 'polynomial']

# Kraska RMI: Hierarchy configurations [stages, experts]
KRASKA_RMI_CONFIGS = [
    [1, 10],      # Small: 1 root + 10 experts
    [1, 100],     # Medium: 1 root + 100 experts (paper default)
    [1, 1000],    # Large: 1 root + 1000 experts
]

# Statistical parameters
REPEAT_CYCLES = 15  # Number of times to repeat each test
RESULTS_DIR = "results/benchmarks"


# ============================================================================
# CSV HEADER
# ============================================================================
"""
Defines all metrics that get recorded for each test.

Core metrics:
    - build_ms: Time to construct the index (milliseconds)
    - lookup_ns: Average time to find a key (nanoseconds)
    - memory_mb: Memory footprint (megabytes)
    
Learned index metrics:
    - accuracy: Fraction of predictions within error bound
    - error_bound: Maximum prediction error (Kraska metric)
    - fallback_rate: Fraction requiring full search
    
Operational metrics:
    - local_calls: Searches in predicted window
    - fallback_calls: Full binary searches needed
"""

CSV_HEADER = [
    "timestamp",              # When test was run
    "cycle",                  # Repetition number (1-5)
    "dataset_size",           # Number of keys
    "distribution",           # seq/uniform/mixed
    "model",                  # Index type
    "params",                 # Model configuration
    "build_ms",               # Build time
    "lookup_ns",              # Lookup time
    "accuracy",               # Prediction accuracy
    "error_bound",            # Max prediction error (Kraska)
    "mean_prediction_error",  # Avg prediction error (Kraska)
    "fallback_rate",          # Full search rate
    "false_neg",              # Missed existing keys
    "not_found",              # Missing keys found
    "local_avg_ns",           # Avg local search time
    "fallback_avg_ns",        # Avg fallback time
    "local_calls",            # Count of local searches
    "fallback_calls",         # Count of fallbacks
    "fit_ms",                 # Model training time
    "memory_mb",              # Memory usage
]


# ============================================================================
# INITIALIZATION
# ============================================================================

def setup_results_folder():
    """
    Create timestamped results folder.
    
    Structure:
        results/benchmarks/run_YYYY-MM-DD_HH-MM-SS/master.csv
    
    Returns:
        tuple: (run_folder_path, master_csv_path)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    run_path = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_path, exist_ok=True)
    
    master_csv = os.path.join(run_path, "master.csv")
    
    # Write CSV header
    with open(master_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    
    return run_path, master_csv


# ============================================================================
# WORKLOAD GENERATION
# ============================================================================

def generate_queries(keys, count=1000, hit_rate=0.5):
    """
    Generate query workload: mix of existing keys and random probes.
    
    Args:
        keys: Sorted array of keys in the index
        count: Total number of queries to generate
        hit_rate: Fraction of queries that should hit existing keys (0.0-1.0)
    
    Returns:
        numpy array of query keys
        
    Example:
        If count=1000 and hit_rate=0.5:
            - 500 queries from existing keys (should be found)
            - 500 random queries (may or may not exist)
    """
    num_hits = int(count * hit_rate)
    num_misses = count - num_hits
    
    # Sample existing keys
    hit_queries = np.random.choice(keys, size=num_hits, replace=True)
    
    # Generate random queries in the key range
    min_key, max_key = keys[0], keys[-1]
    miss_queries = np.random.randint(min_key, max_key + 1, size=num_misses)
    
    # Combine and shuffle
    queries = np.concatenate([hit_queries, miss_queries])
    np.random.shuffle(queries)
    
    return queries


def warmup_cache(index, keys, num_queries=100):
    """
    Warm up caches before timing measurements.
    
    WHY: First few queries often slower due to cold CPU cache, lazy
    initialization, Python JIT, etc. Run throwaway queries first.
    
    IMPORTANCE: Critical for fair benchmarking! Without warmup:
    - First query: 50-100% slower (cold cache)
    - After warmup: Consistent, representative timing
    
    For 1M keys, we use 100 warmup queries to ensure:
    - CPU cache fully populated
    - Branch predictors trained
    - Python bytecode cached
    - Memory pages loaded
    
    Args:
        index: Index structure to warm up
        keys: Dataset keys (needed by some indexes like your learned indexes)
        num_queries: Number of warmup queries to run (default: 100)
    """
    warmup_queries = np.random.choice(keys, size=num_queries, replace=True)
    
    # Check search method signature to call correctly
    import inspect
    sig = inspect.signature(index.search)
    params = list(sig.parameters.keys())
    
    for q in warmup_queries:
        if len(params) == 1:  # search(key) - B-Tree, Kraska
            index.search(q)
        else:  # search(key, keys) - Your learned indexes
            index.search(q, keys)


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def benchmark_index(index_instance, keys, queries):
    """
    Run benchmark on a single index instance.
    
    PROCESS:
        1. Build the index
        2. Warm up caches
        3. Run all queries and time them
        4. Calculate metrics
    
    Args:
        index_instance: Index object with build() or build_from_sorted_array() method
        keys: Sorted array of keys
        queries: Array of query keys to test
    
    Returns:
        dict: All benchmark metrics (build time, lookup time, accuracy, etc.)
    """
    # Phase 1: Build index (handle different method names)
    build_start = time.perf_counter()
    if hasattr(index_instance, 'build'):
        index_instance.build(keys)
    elif hasattr(index_instance, 'build_from_sorted_array'):
        index_instance.build_from_sorted_array(keys)
    else:
        raise AttributeError(f"{type(index_instance).__name__} has no build method")
    build_time_ms = (time.perf_counter() - build_start) * 1000
    
    # Phase 2: Warm up (eliminate cold cache effects)
    warmup_cache(index_instance, keys, num_queries=20)
    
    # Phase 3: Run timed queries
    lookup_times = []
    
    # Check search method signature to call correctly
    import inspect
    sig = inspect.signature(index_instance.search)
    params = list(sig.parameters.keys())
    needs_keys = len(params) > 1  # More than just 'key' parameter
    
    for q in queries:
        start = time.perf_counter_ns()
        if needs_keys:
            found = index_instance.search(q, keys)  # Your learned indexes
        else:
            found = index_instance.search(q)  # B-Tree, Kraska
        elapsed = time.perf_counter_ns() - start
        lookup_times.append(elapsed)
    
    # Phase 4: Calculate metrics
    avg_lookup_ns = np.mean(lookup_times)
    
    # Extract learned index specific metrics
    metrics = index_instance.get_metrics() if hasattr(index_instance, 'get_metrics') else {}
    
    result = {
        'build_ms': build_time_ms,
        'lookup_ns': avg_lookup_ns,
        'accuracy': metrics.get('accuracy', None),
        'error_bound': metrics.get('error_bound', None),
        'mean_prediction_error': metrics.get('mean_prediction_error', None),
        'fallback_rate': metrics.get('fallback_rate', None),
        'false_neg': metrics.get('false_neg', 0),
        'not_found': metrics.get('not_found', 0),
        'local_avg_ns': metrics.get('local_avg_ns', None),
        'fallback_avg_ns': metrics.get('fallback_avg_ns', None),
        'local_calls': metrics.get('local_calls', 0),
        'fallback_calls': metrics.get('fallback_calls', 0),
        'fit_ms': metrics.get('fit_ms', None),
        'memory_mb': metrics.get('memory_mb', 0),
    }
    
    return result


# ============================================================================
# MODEL TESTING FUNCTIONS
# ============================================================================
"""
Each function tests one model type across all its configurations.
They all follow the same pattern:
    1. Loop through configurations
    2. Create index instance
    3. Run benchmark
    4. Write results to CSV
"""

def test_btree(csv_writer, cycle, size, dist, keys, queries):
    """Test B-Tree with different orders."""
    for order in BTREE_ORDERS:
        btree = BTreeOptimized(order=order)
        result = benchmark_index(btree, keys, queries)
        
        row = [
            datetime.now().isoformat(),
            cycle,
            size,
            dist,
            "btree",
            f"order={order}",
            result['build_ms'],
            result['lookup_ns'],
            result['accuracy'],
            result['error_bound'],
            result['mean_prediction_error'],
            result['fallback_rate'],
            result['false_neg'],
            result['not_found'],
            result['local_avg_ns'],
            result['fallback_avg_ns'],
            result['local_calls'],
            result['fallback_calls'],
            result['fit_ms'],
            result['memory_mb'],
        ]
        csv_writer.writerow(row)
        print(f"    ✓ B-Tree (order={order}): {result['lookup_ns']:.1f} ns")


def test_linear_fixed(csv_writer, cycle, size, dist, keys, queries):
    """Test Linear Fixed with different window sizes."""
    for window in FIXED_WINDOWS:
        idx = LearnedIndexOptimized(window=window)
        result = benchmark_index(idx, keys, queries)
        
        row = [
            datetime.now().isoformat(),
            cycle,
            size,
            dist,
            "linear_fixed",
            f"window={window}",
            result['build_ms'],
            result['lookup_ns'],
            result['accuracy'],
            result['error_bound'],
            result['mean_prediction_error'],
            result['fallback_rate'],
            result['false_neg'],
            result['not_found'],
            result['local_avg_ns'],
            result['fallback_avg_ns'],
            result['local_calls'],
            result['fallback_calls'],
            result['fit_ms'],
            result['memory_mb'],
        ]
        csv_writer.writerow(row)
        print(f"    ✓ Linear Fixed (W={window}): {result['lookup_ns']:.1f} ns")


def test_linear_adaptive(csv_writer, cycle, size, dist, keys, queries):
    """Test Linear Adaptive with different quantile/min_window combinations."""
    for q in ADAPTIVE_Q:
        for min_w in ADAPTIVE_MIN_W:
            idx = LinearIndexAdaptive(quantile=q, min_window=min_w)
            result = benchmark_index(idx, keys, queries)
            
            row = [
                datetime.now().isoformat(),
                cycle,
                size,
                dist,
                "linear_adaptive",
                f"quantile={q},min_window={min_w}",
                result['build_ms'],
                result['lookup_ns'],
                result['accuracy'],
                result['error_bound'],
                result['mean_prediction_error'],
                result['fallback_rate'],
                result['false_neg'],
                result['not_found'],
                result['local_avg_ns'],
                result['fallback_avg_ns'],
                result['local_calls'],
                result['fallback_calls'],
                result['fit_ms'],
                result['memory_mb'],
            ]
            csv_writer.writerow(row)
            print(f"    ✓ Linear Adaptive (q={q}, min_w={min_w}): {result['lookup_ns']:.1f} ns")


def test_kraska_single(csv_writer, cycle, size, dist, keys, queries):
    """Test Kraska single-stage models (linear, polynomial)."""
    for model_type in KRASKA_SINGLE_MODELS:
        idx = SingleStageLearnedIndex(model_type=model_type)
        result = benchmark_index(idx, keys, queries)
        
        row = [
            datetime.now().isoformat(),
            cycle,
            size,
            dist,
            "kraska_single",
            f"model={model_type}",
            result['build_ms'],
            result['lookup_ns'],
            result['accuracy'],
            result['error_bound'],
            result['mean_prediction_error'],
            result['fallback_rate'],
            result['false_neg'],
            result['not_found'],
            result['local_avg_ns'],
            result['fallback_avg_ns'],
            result['local_calls'],
            result['fallback_calls'],
            result['fit_ms'],
            result['memory_mb'],
        ]
        csv_writer.writerow(row)
        print(f"    ✓ Kraska Single ({model_type}): {result['lookup_ns']:.1f} ns")


def test_kraska_rmi(csv_writer, cycle, size, dist, keys, queries):
    """Test Kraska RMI with different stage configurations."""
    for stages in KRASKA_RMI_CONFIGS:
        idx = RecursiveModelIndex(stages=stages)
        result = benchmark_index(idx, keys, queries)
        
        row = [
            datetime.now().isoformat(),
            cycle,
            size,
            dist,
            "kraska_rmi",
            f"stages={stages}",
            result['build_ms'],
            result['lookup_ns'],
            result['accuracy'],
            result['error_bound'],
            result['mean_prediction_error'],
            result['fallback_rate'],
            result['false_neg'],
            result['not_found'],
            result['local_avg_ns'],
            result['fallback_avg_ns'],
            result['local_calls'],
            result['fallback_calls'],
            result['fit_ms'],
            result['memory_mb'],
        ]
        csv_writer.writerow(row)
        print(f"    ✓ Kraska RMI {stages}: {result['lookup_ns']:.1f} ns")


# ============================================================================
# MAIN BENCHMARK LOOP
# ============================================================================

def main():
    """
    Main benchmark execution loop.
    
    STRUCTURE:
        For each cycle (1 to REPEAT_CYCLES):
            For each dataset size:
                For each distribution:
                    Generate dataset
                    Generate queries
                    Test all models
                    Write results to CSV
    
    This nested structure ensures statistical validity by repeating
    measurements and testing across different data patterns and scales.
    """
    print("\n" + "="*70)
    print("📊 BENCHMARK RUNNER")
    print("="*70)
    
    # Setup results folder
    run_path, master_csv = setup_results_folder()
    
    print(f"\n🚀 Benchmark Runner Started")
    print(f"📁 Saving results: {run_path}")
    print(f"📄 Master CSV: {master_csv}")
    print(f"\nModels tested:")
    print(f"  • Your models (Linear Fixed, Linear Adaptive)")
    print(f"  • Kraska models (Single-stage, RMI)")
    print(f"  • Baselines (B-Tree)")
    print()
    
    # Main benchmark loop
    total_tests = len(DATASET_SIZES) * len(DISTRIBUTIONS) * REPEAT_CYCLES
    test_count = 0
    
    for cycle in range(1, REPEAT_CYCLES + 1):
        print(f"\n{'='*70}")
        print(f"🔄 CYCLE {cycle}/{REPEAT_CYCLES}")
        print(f"{'='*70}")
        
        for size in DATASET_SIZES:
            for dist in DISTRIBUTIONS:
                test_count += 1
                print(f"\n[{test_count}/{total_tests}] Testing: {size:,} keys, {dist} distribution")
                
                # Generate dataset using appropriate static method
                if dist == "seq":
                    keys = DatasetGenerator.generate_sequential(size)
                elif dist == "uniform":
                    keys = DatasetGenerator.generate_uniform(size)
                elif dist == "mixed":
                    keys = DatasetGenerator.generate_mixed(size)
                else:
                    raise ValueError(f"Unknown distribution: {dist}")
                
                queries = generate_queries(keys, count=1000, hit_rate=0.5)
                
                # Open CSV and test all models
                with open(master_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    
                    test_btree(writer, cycle, size, dist, keys, queries)
                    test_linear_fixed(writer, cycle, size, dist, keys, queries)
                    test_linear_adaptive(writer, cycle, size, dist, keys, queries)
                    test_kraska_single(writer, cycle, size, dist, keys, queries)
                    test_kraska_rmi(writer, cycle, size, dist, keys, queries)
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {master_csv}")
    print(f"\n📈 Next steps:")
    print(f"   1. Generate graphs: python benchmarks/generate_graphs.py")
    print(f"   2. Statistical analysis: python benchmarks/statistical_analysis.py")
    print()


if __name__ == "__main__":
    main()
