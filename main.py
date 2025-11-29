"""
MAIN BENCHMARK RUNNER
Run everything from here - easy configuration at the top
"""

# ============================================================================
# CONFIGURATION - EDIT THESE TO CONTROL EVERYTHING
# ============================================================================

# What to run?
RUN_BENCHMARK = True          # Run the main benchmark
RUN_ANALYSIS = True           # Generate statistical analysis  
RUN_GRAPHS = True             # Generate performance graphs from benchmark data
RUN_EXPLORATORY_PLOTS = False # Generate exploratory plots (uses test data, not benchmark results)

# Benchmark settings
DATASET_SIZES = [10_000_000]              # Number of keys to test [100_000, 1_000_000, 100_000_000]
DISTRIBUTIONS = ["s  eq", "uniform", "mixed"]  # Data patterns
REPEAT_CYCLES = 1                        # Times to repeat (higher = better statistics)

# ACCURACY VALIDATION (Critical!)
VALIDATE_ACCURACY = True       # Check if searches return correct results
ACCURACY_HANDLING = "discard"  # Options:
                               #   "discard" - throw out incorrect searches (don't count time)
                               #   "penalty" - add penalty to lookup time for wrong results
PENALTY_MULTIPLIER = 10.0      # If penalty mode, multiply lookup time by this

# Models to test
BTREE_ORDERS = [128]                    # B-Tree configurations
FIXED_WINDOWS = [2048]                   # Linear Fixed window sizes
ADAPTIVE_Q = [0.99]                     # Linear Adaptive quantiles
ADAPTIVE_MIN_W = [64]                   # Linear Adaptive min windows
KRASKA_SINGLE_MODELS = ['linear']       # Kraska single-stage
KRASKA_RMI_CONFIGS = [[1, 100]]         # Kraska RMI [stages, experts]

# Output
RESULTS_DIR = "results/benchmarks"
GRAPHS_DIR = "graphs"

# ============================================================================
# END CONFIGURATION - Code starts here
# ============================================================================

import os
import sys

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import csv
import numpy as np
from datetime import datetime

# Imports
from src.utils.data_loader import DatasetGenerator
from src.indexes.btree_optimized import BTreeOptimized
from src.indexes.learned_index_optimized import LearnedIndexOptimized
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.indexes.learned_index_kraska import SingleStageLearnedIndex, RecursiveModelIndex


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def print_config():
    """Display current configuration."""
    print_header("CONFIGURATION")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Distributions: {DISTRIBUTIONS}")
    print(f"Repeat cycles: {REPEAT_CYCLES}")
    print(f"\nAccuracy validation: {'ENABLED' if VALIDATE_ACCURACY else 'DISABLED'}")
    if VALIDATE_ACCURACY:
        print(f"  Mode: {ACCURACY_HANDLING}")
        if ACCURACY_HANDLING == "penalty":
            print(f"  Penalty: {PENALTY_MULTIPLIER}x slower for wrong results")
        else:
            print(f"  Incorrect searches: DISCARDED (not counted)")
    print(f"\nModels:")
    print(f"  B-Tree: {BTREE_ORDERS}")
    print(f"  Linear Fixed: {FIXED_WINDOWS}")
    print(f"  Linear Adaptive: q={ADAPTIVE_Q}, min_w={ADAPTIVE_MIN_W}")
    print(f"  Kraska Single: {KRASKA_SINGLE_MODELS}")
    print(f"  Kraska RMI: {KRASKA_RMI_CONFIGS}")


def benchmark_index_with_validation(index_instance, keys, queries, validate=True, handling="discard", penalty=10.0):
    """
    Benchmark an index with accuracy validation.
    
    Args:
        index_instance: Index to test
        keys: Sorted array of keys
        queries: Query keys (mix of existing and non-existing)
        validate: Whether to check correctness
        handling: "discard" or "penalty"
        penalty: Penalty multiplier for wrong results
    
    Returns:
        dict: Metrics including corrected lookup times
    """
    import inspect
    
    # Build index
    build_start = time.perf_counter()
    if hasattr(index_instance, 'build'):
        index_instance.build(keys)
    elif hasattr(index_instance, 'build_from_sorted_array'):
        index_instance.build_from_sorted_array(keys)
    build_time_ms = (time.perf_counter() - build_start) * 1000
    
    # Warmup
    warmup_queries = np.random.choice(keys, size=min(20, len(keys)), replace=False)
    sig = inspect.signature(index_instance.search)
    needs_keys = len(list(sig.parameters.keys())) > 1
    
    for q in warmup_queries:
        if needs_keys:
            index_instance.search(q, keys)
        else:
            index_instance.search(q)
    
    # Run queries with validation
    lookup_times = []
    correct_count = 0
    incorrect_count = 0
    
    # Pre-compute expected results for validation
    expected_results = {}
    if validate:
        keys_set = set(keys)
        for q in queries:
            expected_results[q] = q in keys_set
    
    for q in queries:
        start = time.perf_counter_ns()
        if needs_keys:
            result = index_instance.search(q, keys)
        else:
            result = index_instance.search(q)
        elapsed = time.perf_counter_ns() - start
        
        # Handle different return types
        if isinstance(result, tuple):
            # Kraska returns (found, position)
            found = bool(result[0])
        else:
            # Others return boolean directly
            found = bool(result)
        
        # Validate result
        if validate:
            expected = expected_results[q]
            is_correct = (found == expected)
            
            if is_correct:
                correct_count += 1
                lookup_times.append(elapsed)
            else:
                incorrect_count += 1
                if handling == "penalty":
                    # Apply penalty
                    lookup_times.append(elapsed * penalty)
                # If "discard", don't add to lookup_times
        else:
            # No validation, accept all
            lookup_times.append(elapsed)
    
    # Calculate metrics
    if len(lookup_times) == 0:
        avg_lookup_ns = np.nan  # All searches were incorrect!
    else:
        avg_lookup_ns = np.mean(lookup_times)
    
    # Get model-specific metrics
    metrics = index_instance.get_metrics() if hasattr(index_instance, 'get_metrics') else {}
    
    # Calculate search accuracy (different from prediction accuracy)
    total_queries = len(queries)
    search_accuracy = correct_count / total_queries if total_queries > 0 else 0.0
    
    result = {
        'build_ms': build_time_ms,
        'lookup_ns': avg_lookup_ns,
        'search_accuracy': search_accuracy,  # NEW: fraction of correct searches
        'searches_correct': correct_count,    # NEW
        'searches_incorrect': incorrect_count, # NEW
        'searches_counted': len(lookup_times), # NEW: how many contributed to avg
        'accuracy': metrics.get('accuracy', None),  # Model prediction accuracy
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


def run_benchmark():
    """Run the main benchmark."""
    print_header("RUNNING BENCHMARK")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    run_path = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_path, exist_ok=True)
    
    master_csv = os.path.join(run_path, "master.csv")
    
    # CSV Header
    header = [
        "timestamp", "cycle", "dataset_size", "distribution", "model", "params",
        "build_ms", "lookup_ns", "search_accuracy", "searches_correct", 
        "searches_incorrect", "searches_counted",
        "accuracy", "error_bound", "mean_prediction_error", "fallback_rate",
        "false_neg", "not_found", "local_avg_ns", "fallback_avg_ns",
        "local_calls", "fallback_calls", "fit_ms", "memory_mb"
    ]
    
    with open(master_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    print(f"Results will be saved to: {master_csv}\n")
    
    # Generate test configurations
    configs = []
    
    # B-Tree
    for order in BTREE_ORDERS:
        configs.append(('btree', f'order={order}', lambda: BTreeOptimized(order=order)))
    
    # Linear Fixed
    for window in FIXED_WINDOWS:
        configs.append(('linear_fixed', f'window={window}', 
                       lambda w=window: LearnedIndexOptimized(window=w)))
    
    # Linear Adaptive
    for q in ADAPTIVE_Q:
        for min_w in ADAPTIVE_MIN_W:
            configs.append(('linear_adaptive', f'quantile={q},min_window={min_w}',
                           lambda q=q, m=min_w: LinearIndexAdaptive(quantile=q, min_window=m)))
    
    # Kraska Single
    for model_type in KRASKA_SINGLE_MODELS:
        configs.append(('kraska_single', f'model={model_type}',
                       lambda m=model_type: SingleStageLearnedIndex(model_type=m)))
    
    # Kraska RMI
    for stages in KRASKA_RMI_CONFIGS:
        configs.append(('kraska_rmi', f'stages={stages}',
                       lambda s=stages: RecursiveModelIndex(stages=s)))
    
    total_tests = len(DATASET_SIZES) * len(DISTRIBUTIONS) * len(configs) * REPEAT_CYCLES
    current_test = 0
    start_time = time.time()
    
    print(f"Total tests to run: {total_tests}\n")
    
    # Run benchmarks
    for size in DATASET_SIZES:
        for dist in DISTRIBUTIONS:
            # Generate dataset
            print(f"\n[Dataset: {size:,} keys, {dist}]")
            if dist == "seq":
                keys = DatasetGenerator.generate_sequential(size)
            elif dist == "uniform":
                keys = DatasetGenerator.generate_uniform(size)
            else:
                keys = DatasetGenerator.generate_mixed(size)
            
            # Generate queries (50% existing, 50% random)
            num_queries = 1000
            existing = np.random.choice(keys, size=num_queries//2, replace=False)
            random_queries = np.random.uniform(keys.min(), keys.max(), size=num_queries//2)
            queries = np.concatenate([existing, random_queries])
            np.random.shuffle(queries)
            
            for model_name, params, create_index in configs:
                for cycle in range(REPEAT_CYCLES):
                    current_test += 1
                    
                    # Progress
                    pct = (current_test / total_tests) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / current_test) * (total_tests - current_test) if current_test > 0 else 0
                    
                    print(f"  [{current_test}/{total_tests} - {pct:.1f}%] "
                          f"{model_name} {params} (cycle {cycle+1}/{REPEAT_CYCLES}) "
                          f"[ETA: {eta/60:.1f}min]", end='')
                    
                    # Create fresh index instance
                    index = create_index()
                    
                    # Benchmark with validation
                    try:
                        result = benchmark_index_with_validation(
                            index, keys, queries,
                            validate=VALIDATE_ACCURACY,
                            handling=ACCURACY_HANDLING,
                            penalty=PENALTY_MULTIPLIER
                        )
                        
                        # Check for failures
                        if VALIDATE_ACCURACY and result['search_accuracy'] < 1.0:
                            print(f" - WARNING: {result['searches_incorrect']} incorrect searches!", end='')
                        
                        print()  # Newline
                        
                        # Write result
                        row = [
                            datetime.now().isoformat(),
                            cycle + 1,
                            size,
                            dist,
                            model_name,
                            params,
                            result['build_ms'],
                            result['lookup_ns'],
                            result['search_accuracy'],
                            result['searches_correct'],
                            result['searches_incorrect'],
                            result['searches_counted'],
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
                        
                        with open(master_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                            
                    except Exception as e:
                        print(f" - ERROR: {e}")
                        continue
    
    elapsed_total = time.time() - start_time
    print(f"\n\nBenchmark completed in {elapsed_total/60:.1f} minutes")
    print(f"Results: {master_csv}")
    
    return master_csv


def run_analysis(master_csv):
    """Generate statistical analysis."""
    print_header("STATISTICAL ANALYSIS")
    
    import pandas as pd
    
    df = pd.read_csv(master_csv)
    
    # Group by model, size, distribution
    grouped = df.groupby(['model', 'dataset_size', 'distribution'])
    
    stats = grouped.agg({
        'lookup_ns': ['mean', 'std', 'count'],
        'search_accuracy': 'mean',
        'build_ms': 'mean',
        'memory_mb': 'mean',
        'accuracy': 'mean'
    }).round(2)
    
    print("\nResults Summary:")
    print(stats)
    
    # Check for accuracy issues
    print("\n\nSearch Accuracy Check:")
    accuracy_check = df.groupby(['model', 'params'])['search_accuracy'].mean()
    for (model, params), acc in accuracy_check.items():
        if acc < 1.0:
            print(f"  WARNING: {model} {params} has {acc*100:.1f}% search accuracy!")
        else:
            print(f"  OK: {model} {params} - 100% accurate")


def run_graphs(master_csv):
    """Generate graphs from benchmark data."""
    print_header("GENERATING GRAPHS FROM BENCHMARK DATA")
    
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(master_csv)
    
    # Get largest dataset
    max_size = df['dataset_size'].max()
    df_large = df[df['dataset_size'] == max_size]
    
    # Group and average
    summary = df_large.groupby(['model', 'params', 'distribution']).agg({
        'lookup_ns': 'mean',
        'build_ms': 'mean',
        'memory_mb': 'mean',
        'search_accuracy': 'mean',
        'accuracy': 'mean'  # prediction accuracy
    }).reset_index()
    
    # Convert ns to µs
    summary['lookup_us'] = summary['lookup_ns'] / 1000
    
    print(f"\nUsing data from {max_size:,} keys")
    print(f"Plotting {len(summary)} configurations")
    
    # Format size for folder name (e.g., 1M, 10M, 100M)
    if max_size >= 1_000_000:
        size_str = f"{max_size // 1_000_000}M"
    elif max_size >= 1_000:
        size_str = f"{max_size // 1_000}K"
    else:
        size_str = str(max_size)
    
    # Create size-specific folder
    size_graphs_dir = os.path.join(GRAPHS_DIR, size_str)
    os.makedirs(size_graphs_dir, exist_ok=True)
    print(f"Saving to: {size_graphs_dir}/")
    print()
    
    colors = {'btree': '#95A5A6', 'kraska_single': '#E74C3C', 
              'kraska_rmi': '#3498DB', 'linear_fixed': '#1ABC9C',
              'linear_adaptive': '#2ECC71'}
    
    # Plot for each distribution
    for dist in DISTRIBUTIONS:
        dist_data = summary[summary['distribution'] == dist]
        
        if len(dist_data) == 0:
            continue
        
        # === LOOKUP TIME GRAPH ===
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(dist_data))
        
        bars = plt.bar(x_pos, dist_data['lookup_us'], 
                      color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
        
        plt.xlabel('Index Configuration')
        plt.ylabel('Lookup Time (µs)')
        plt.title(f'Lookup Performance - {dist.capitalize()} Distribution ({size_str} keys)')
        plt.xticks(x_pos, [f"{row['model']}\n{row['params']}" 
                           for _, row in dist_data.iterrows()], rotation=45, ha='right')
        plt.tight_layout()
        
        filename = os.path.join(size_graphs_dir, f'lookup_time_{dist}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
        
        # === PREDICTION ACCURACY GRAPH ===
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(dist_data))
        
        # Convert to percentage
        accuracy_pct = dist_data['accuracy'] * 100
        
        bars = plt.bar(x_pos, accuracy_pct,
                      color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
        
        plt.xlabel('Index Configuration')
        plt.ylabel('Prediction Accuracy (%)')
        plt.title(f'Prediction Accuracy - {dist.capitalize()} Distribution ({size_str} keys)')
        plt.xticks(x_pos, [f"{row['model']}\n{row['params']}" 
                           for _, row in dist_data.iterrows()], rotation=45, ha='right')
        plt.ylim(0, 105)  # 0-100% with some headroom
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(size_graphs_dir, f'accuracy_{dist}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # === BUILD TIME COMPARISON ===
    plt.figure(figsize=(14, 6))
    
    for i, dist in enumerate(DISTRIBUTIONS):
        dist_data = summary[summary['distribution'] == dist]
        if len(dist_data) == 0:
            continue
        
        x_pos = np.arange(len(dist_data)) + i * 0.25
        plt.bar(x_pos, dist_data['build_ms'],
               width=0.25, label=dist.capitalize(),
               alpha=0.8)
    
    plt.xlabel('Index Configuration')
    plt.ylabel('Build Time (ms)')
    plt.title(f'Index Build Time Comparison ({size_str} keys)')
    plt.legend()
    plt.xticks(np.arange(len(dist_data)) + 0.25, 
               [f"{row['model']}" for _, row in dist_data.iterrows()], 
               rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(size_graphs_dir, f'build_time.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    # === COMBINED COMPARISON ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, dist in enumerate(DISTRIBUTIONS):
        dist_data = summary[summary['distribution'] == dist]
        if len(dist_data) == 0:
            continue
        
        ax = axes[idx]
        x_pos = np.arange(len(dist_data))
        
        ax.bar(x_pos, dist_data['lookup_us'],
              color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
        
        ax.set_xlabel('Index', fontsize=10)
        ax.set_ylabel('Lookup Time (µs)', fontsize=10)
        ax.set_title(f'{dist.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([row['model'] for _, row in dist_data.iterrows()], 
                           rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Lookup Performance Comparison ({size_str} keys)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(size_graphs_dir, f'comparison_all.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    print(f"\nGraphs saved to {size_graphs_dir}/")


def run_exploratory():
    """Run exploratory visualization plots."""
    print_header("GENERATING EXPLORATORY PLOTS")
    print("(Visualizations using test data)")
    print()
    
    import subprocess
    
    plot_scripts = [
        'src/plots/learned_index_plot.py',
        'src/plots/linear_index_adaptive_plot.py',
        # 'src/plots/rmi_plot.py',  # Uncomment if you want RMI plots
    ]
    
    for script in plot_scripts:
        if os.path.exists(script):
            print(f"Running {script}...")
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print(f"  ✅ {script} completed")
                else:
                    print(f"  ⚠️ {script} had errors:")
                    if result.stderr:
                        print(f"    {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"  ⚠️ {script} timed out")
            except Exception as e:
                print(f"  ⚠️ {script} failed: {e}")
        else:
            print(f"  ⚠️ {script} not found")
    
    print()
    print(f"Exploratory plots saved to {GRAPHS_DIR}/")


def main():
    """Main entry point."""
    print_header("LEARNED INDEX BENCHMARK SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_config()
    
    master_csv = None
    
    # Run benchmark
    if RUN_BENCHMARK:
        try:
            master_csv = run_benchmark()
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            return
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Find latest results
        import glob
        csv_files = glob.glob(os.path.join(RESULTS_DIR, "run_*/master.csv"))
        if csv_files:
            master_csv = max(csv_files, key=os.path.getmtime)
            print(f"\nUsing latest results: {master_csv}")
    
    # Analysis
    if RUN_ANALYSIS and master_csv:
        try:
            run_analysis(master_csv)
        except Exception as e:
            print(f"\n\nAnalysis ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Graphs
    if RUN_GRAPHS and master_csv:
        try:
            run_graphs(master_csv)
        except Exception as e:
            print(f"\n\nGraphs ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Exploratory plots
    if RUN_EXPLORATORY_PLOTS:
        try:
            run_exploratory()
        except Exception as e:
            print(f"\n\nExploratory plots ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print_header("COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if master_csv:
        print(f"\nYour results: {os.path.dirname(master_csv)}/")
    if RUN_GRAPHS:
        print(f"Your graphs: {GRAPHS_DIR}/")
    if RUN_EXPLORATORY_PLOTS:
        print(f"Your exploratory plots: {GRAPHS_DIR}/")


if __name__ == "__main__":
    main()
