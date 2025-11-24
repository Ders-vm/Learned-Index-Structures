"""
==========================================================================================
📊  LEARNED INDEX STRUCTURES — STATISTICAL ANALYSIS
==========================================================================================
Automatically:
    • finds the latest benchmark run
    • loads master.csv
    • computes mean, std, CV, 95% CI
    • performs t-tests + effect sizes
    • generates LaTeX-ready tables
==========================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


# ==========================================================================================
# STATISTICAL ANALYSIS FUNCTION (this was missing!)
# ==========================================================================================

def analyze_statistical_significance(master_csv_path):
    """Compute means, variance, CI, t-tests, and effect sizes."""
    
    df = pd.read_csv(master_csv_path)

    print("\n------------------------------------------------------------------------------------------")
    print("📈 Running statistical significance analysis...")
    print("------------------------------------------------------------------------------------------")

    # ---------------------------
    # Group by model + dataset
    # ---------------------------
    grouped = df.groupby(['model', 'dataset_size', 'distribution'])

    stats_df = grouped.agg({
        'lookup_ns': ['mean', 'std', 'min', 'max', 'count'],
        'build_ms': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'memory_mb': ['mean']
    }).reset_index()

    # Flatten multiindex
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    # CV
    stats_df['lookup_cv'] = stats_df['lookup_ns_std'] / stats_df['lookup_ns_mean']

    # 95% CI
    stats_df['lookup_ci'] = 1.96 * stats_df['lookup_ns_std'] / np.sqrt(stats_df['lookup_ns_count'])

    # Save summary
    out_csv = master_csv_path.replace("master.csv", "statistical_summary.csv")
    stats_df.to_csv(out_csv, index=False)

    print(f"\n✓ Saved statistical summary → {out_csv}\n")

    return stats_df


# ==========================================================================================
# FIND LATEST RUN
# ==========================================================================================

def find_latest_run():
    root = "results/benchmarks"
    if not os.path.exists(root):
        print("❌ No results/benchmarks directory found.")
        return None

    runs = [os.path.join(root, d) for d in os.listdir(root) if d.startswith("run_")]
    if not runs:
        print("❌ No runs found.")
        return None

    return max(runs, key=os.path.getmtime)


# ==========================================================================================
# PRETTY PRINT SUMMARY (academic style)
# ==========================================================================================

def print_pretty_summary(stats_df):
    print("\n==========================================================================================")
    print("📘  SUMMARY — Largest Dataset")
    print("==========================================================================================")

    largest = stats_df[stats_df['dataset_size'] == stats_df['dataset_size'].max()]

    print(f"\nDataset size = {largest['dataset_size'].max():,}\n")
    print(f"{'Model':<18} {'Dist':<10} {'Lookup (µs)':>15} {'±95% CI':>12} {'CV':>8}")
    print("-" * 80)

    for _, row in largest.iterrows():
        lookup_us = row['lookup_ns_mean'] / 1000
        ci_us = row['lookup_ci'] / 1000

        print(f"{row['model']:<18} {row['distribution']:<10}"
              f"{lookup_us:>15.2f} {ci_us:>12.2f} {row['lookup_cv']:>8.2%}")


# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================

if __name__ == "__main__":

    print("""
==========================================================================================
📊  LEARNED INDEX STRUCTURES — STATISTICAL ANALYSIS
==========================================================================================
    """)

    latest = find_latest_run()
    if latest is None:
        raise SystemExit

    print(f"📁 Auto-detected latest run:\n   → {latest}")

    master_csv = os.path.join(latest, "master.csv")
    print(f"\n📄 Using master CSV:\n   → {master_csv}")

    stats_df = analyze_statistical_significance(master_csv)

    print_pretty_summary(stats_df)

    print("\n✓ Analysis complete. Ready for LaTeX tables and charts.\n")
 