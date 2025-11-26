"""
Simplified Statistical Analysis for Benchmark Results
Focuses on lookup times and significance testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os


def analyze_statistical_significance(master_csv_path):
    """
    Perform statistical analysis focusing on lookup time and accuracy.
    Handles missing or invalid accuracy values gracefully.
    """

    df = pd.read_csv(master_csv_path)

    # Handle missing accuracy column
    if "accuracy" not in df.columns:
        print("\n[WARNING]  'accuracy' column not found in CSV — filling with 0.0")
        df["accuracy"] = 0.0
    else:
        # Replace non-numeric or NaN accuracy values with 0.0
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0.0)

    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICAL SUMMARY")
    print("=" * 80)

    # Group by model, dataset size, and distribution
    grouped = df.groupby(["model", "dataset_size", "distribution"])

    # Compute lookup and accuracy statistics
    stats_df = grouped.agg({
        "lookup_ns": ["mean", "std", "count"],
        "accuracy": ["mean", "std"]
    }).reset_index()

    # Flatten multi-level columns
    stats_df.columns = ["_".join(col).strip("_") for col in stats_df.columns.values]

    # Compute coefficient of variation and 95% confidence interval
    stats_df["lookup_cv"] = stats_df["lookup_ns_std"] / stats_df["lookup_ns_mean"]
    stats_df["lookup_ci"] = 1.96 * stats_df["lookup_ns_std"] / np.sqrt(stats_df["lookup_ns_count"])

    # Convert nanoseconds to microseconds
    stats_df["lookup_us_mean"] = stats_df["lookup_ns_mean"] / 1000
    stats_df["lookup_us_ci"] = stats_df["lookup_ci"] / 1000

    # Replace NaN accuracy means with 0.0
    stats_df["accuracy_mean"] = stats_df["accuracy_mean"].fillna(0.0)

    # Save summary
    output_path = master_csv_path.replace("master.csv", "lookup_summary.csv")
    stats_df.to_csv(output_path, index=False)
    print(f"\n[DONE] Summary saved to: {output_path}")

    # Focus on largest dataset (usually 1M keys)
    max_size = stats_df["dataset_size"].max()
    largest = stats_df[stats_df["dataset_size"] == max_size]

    print("\n" + "=" * 80)
    print(f"RESULTS FOR LARGEST DATASET ({max_size:,} keys)")
    print("=" * 80)
    print(f"{'Model':<20} {'Distribution':<12} {'Lookup (µs)':>15} {'±95% CI':>12} {'CV':>8} {'Accuracy':>10}")
    print("-" * 85)

    for _, row in largest.iterrows():
        print(f"{row['model']:<20} {row['distribution']:<12} "
              f"{row['lookup_us_mean']:>15.2f} {row['lookup_us_ci']:>12.2f} "
              f"{row['lookup_cv']:>8.2%} {row['accuracy_mean']:>10.4f}")

    # Statistical significance testing (Uniform distribution only)
    print("\n" + "=" * 80)
    print("PAIRWISE SIGNIFICANCE TESTS (Uniform Distribution)")
    print("=" * 80)

    uniform_large = df[
        (df["dataset_size"] == max_size) & (df["distribution"] == "uniform")
    ]

    models = sorted(uniform_large["model"].unique())

    if len(models) < 2:
        print("\n[WARNING]  Not enough models for significance testing.")
    else:
        print(f"{'Comparison':<40} {'Faster Model':<20} {'Speedup':>8} {'p-value':>10} {'Sig':>5}")
        print("-" * 85)

        for i, model1 in enumerate(models):
            for model2 in models[i + 1:]:
                data1 = uniform_large[uniform_large["model"] == model1]["lookup_ns"]
                data2 = uniform_large[uniform_large["model"] == model2]["lookup_ns"]

                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    mean1, mean2 = data1.mean(), data2.mean()
                    faster = model1 if mean1 < mean2 else model2
                    speedup = max(mean1, mean2) / min(mean1, mean2)
                    sig = (
                        "***" if p_value < 0.001 else
                        "**" if p_value < 0.01 else
                        "*" if p_value < 0.05 else "ns"
                    )

                    print(f"{model1} vs {model2:<30} {faster:<20} "
                          f"{speedup:>7.2f}x {p_value:>10.4f} {sig:>5}")

        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Flag any high-variance results
    print("\n" + "=" * 80)
    print("VARIANCE CHECK (CV > 10%)")
    print("=" * 80)
    high_var = largest[largest["lookup_cv"] > 0.1]
    if not high_var.empty:
        print(f"{'Model':<20} {'Distribution':<12} {'CV':>8}")
        print("-" * 45)
        for _, row in high_var.iterrows():
            print(f"{row['model']:<20} {row['distribution']:<12} {row['lookup_cv']:>8.2%}")
    else:
        print("[DONE] All variance levels acceptable (CV < 10%)")

    print("\n[OK] Analysis complete!\n")
    return stats_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        master_csv = sys.argv[1]
    else:
        # Auto-detect latest benchmark run
        results_root = "results/benchmarks"
        if os.path.exists(results_root):
            subdirs = [
                os.path.join(results_root, d)
                for d in os.listdir(results_root)
                if d.startswith("run_")
            ]
            if subdirs:
                latest = max(subdirs, key=os.path.getmtime)
                master_csv = os.path.join(latest, "master.csv")
            else:
                print("[ERROR] No benchmark results found")
                sys.exit(1)
        else:
            print("[ERROR] results/benchmarks directory not found")
            sys.exit(1)

    print(f"\n[BENCHMARK] Analyzing: {master_csv}")
    analyze_statistical_significance(master_csv)
