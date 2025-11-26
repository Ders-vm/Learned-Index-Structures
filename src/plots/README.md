# Exploratory Visualization

This folder contains scripts for generating exploratory visualizations that show how learned indexes work internally.

## Quick Start

```bash
# From project root
PYTHONPATH=. python src/plots/generate_exploratory_plots.py

# Or use the provided script
bash src/plots/run_exploratory_plots.sh
```

## What Gets Generated

**4 Exploratory Plots:**

1. **model_fits.png** - Shows how linear models fit different data distributions
   - Sequential: Perfect linear fit
   - Uniform: Good linear fit
   - Mixed: Poor fit (non-linear data)

2. **accuracy_heatmap.png** - Prediction accuracy at different error thresholds
   - Shows % of predictions within 1, 10, 50, 100, 500, 1000 positions
   - Color-coded: green = high accuracy, red = low accuracy

3. **error_distribution.png** - Distribution of prediction errors
   - Histogram of absolute errors
   - Shows mean and median error lines

4. **adaptive_vs_fixed.png** - Compares adaptive vs fixed window sizing
   - Shows how adaptive window adjusts to data patterns

## Output Location

All plots are saved to: `exploratory_plots/`

## Purpose

These visualizations help you:
- Understand how learned indexes work
- Debug model behavior on different distributions
- Explain your approach in presentations
- Show why adaptive window sizing helps

## Note

This is different from `src/benchmarks/generate_graphs.py`:
- **Exploratory plots** (this folder): Show model internals and behavior
- **Benchmark graphs** (benchmarks/): Compare performance across models

Use exploratory plots for understanding and explaining your approach.
Use benchmark graphs for reporting performance results in papers.

## Legacy Files

The `.py` files (`learned_index_plot.py`, `linear_index_adaptive_plot.py`, `rmi_plot.py`) are original Jupyter-style notebooks. They've been superseded by `generate_exploratory_plots.py` which is cleaner and easier to run.
