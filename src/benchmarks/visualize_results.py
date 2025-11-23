"""
Graph generator for systematic overnight benchmark results.

Runs cleanly inside VS Code by pressing "Run Python File".
Automatically locates results, creates graph folders, and handles missing data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force backend that works in VS Code
plt.switch_backend("Agg")  

plt.style.use("seaborn-v0_8-darkgrid")


# =========================================================
# RESOLVE PROJECT ROOT (VS Code friendly)
# =========================================================

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))

# Allow importing project modules if needed later
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# CONFIG
# =========================================================

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "overnight")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)


# =========================================================
# LOAD LATEST RUN
# =========================================================

def load_latest_run():
    """Find the newest folder inside results/overnight and load master.csv."""

    if not os.path.exists(RESULTS_ROOT):
        raise FileNotFoundError(
            f"❌ No results folder found.\n"
            f"Expected: {RESULTS_ROOT}\n"
            f"Run the overnight benchmark first."
        )

    # list run_* folders
    subdirs = [
        os.path.join(RESULTS_ROOT, d)
        for d in os.listdir(RESULTS_ROOT)
        if d.startswith("run_")
    ]

    if not subdirs:
        raise FileNotFoundError(
            f"❌ No run folders found in {RESULTS_ROOT}\n"
            f"Run systematic_overnight_runner.py first."
        )

    # newest run by timestamp
    latest = max(subdirs, key=os.path.getmtime)
    master_path = os.path.join(latest, "master.csv")

    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"❌ master.csv not found in {latest}\n"
            f"Your overnight runner probably failed early."
        )

    print(f"📄 Loading: {master_path}")
    df = pd.read_csv(master_path)

    if df.empty:
        raise ValueError("❌ master.csv is empty — no benchmark results found.")

    return df, latest


# =========================================================
# PLOTTING HELPERS
# =========================================================

def make_plot(df, x, y, hue, title, ylabel, filename):
    if df.empty:
        print(f"⚠ No data for {title}, skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel)
    plt.xlabel(x)
    plt.tight_layout()

    out_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(out_path)
    plt.close()

    print(f"✓ Saved graph → {out_path}")


# =========================================================
# GRAPH FUNCTIONS
# =========================================================

def graph_latency_vs_size(df):
    make_plot(
        df,
        x="dataset_size",
        y="lookup_ns",
        hue="model",
        title="Lookup Latency vs Dataset Size",
        ylabel="Lookup Time (ns)",
        filename="latency_vs_size.png",
    )


def graph_build_time_vs_size(df):
    make_plot(
        df,
        x="dataset_size",
        y="build_ms",
        hue="model",
        title="Build Time vs Dataset Size",
        ylabel="Build Time (ms)",
        filename="build_time_vs_size.png",
    )


def graph_memory_vs_size(df):
    make_plot(
        df,
        x="dataset_size",
        y="memory_mb",
        hue="model",
        title="Memory Usage vs Dataset Size",
        ylabel="Memory (MB)",
        filename="memory_vs_size.png",
    )


def graph_accuracy(df):
    sub = df[df["model"].isin(["pgm", "linear_fixed", "linear_adaptive"])]
    make_plot(
        sub,
        x="dataset_size",
        y="accuracy",
        hue="model",
        title="Accuracy Across Models",
        ylabel="Accuracy",
        filename="accuracy.png",
    )


def graph_pgm_epsilon(df):
    pgm = df[df["model"] == "pgm"].copy()
    if pgm.empty:
        print("⚠ No PGM rows found.")
        return

    pgm["epsilon"] = pgm["params"].str.extract(r"eps=(\d+)").astype(int)

    make_plot(
        pgm,
        x="epsilon",
        y="lookup_ns",
        hue="distribution",
        title="PGM Lookup Latency vs Epsilon",
        ylabel="Lookup (ns)",
        filename="pgm_epsilon_latency.png",
    )

    make_plot(
        pgm,
        x="epsilon",
        y="accuracy",
        hue="distribution",
        title="PGM Accuracy vs Epsilon",
        ylabel="Prediction Accuracy",
        filename="pgm_epsilon_accuracy.png",
    )


def graph_model_comparison(df):
    biggest = df["dataset_size"].max()
    sub = df[df["dataset_size"] == biggest]

    if sub.empty:
        print("⚠ No rows for largest dataset size.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sub, x="model", y="lookup_ns")
    plt.title(f"Model Comparison at Size {biggest:,}")
    plt.ylabel("Latency (ns)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(GRAPH_DIR, "model_comparison_biggest.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✓ Saved graph → {out_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n🔍 Searching for latest overnight benchmark...")
    df, run_dir = load_latest_run()

    print(f"📊 Loaded data from: {run_dir}")
    print(f"📁 Saving graphs to: {GRAPH_DIR}\n")

    graph_latency_vs_size(df)
    graph_build_time_vs_size(df)
    graph_memory_vs_size(df)
    graph_accuracy(df)
    graph_pgm_epsilon(df)
    graph_model_comparison(df)

    print("\n🎉 All graphs saved successfully!\n")


if __name__ == "__main__":
    main()
