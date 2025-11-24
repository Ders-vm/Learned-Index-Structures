# Visualization & Analysis Tools

This directory contains **standalone visualization notebooks** for exploratory analysis and visualization of learned index behavior.

## ⚠️ Important Notes

- These are **Jupyter/Jupytext notebooks** meant for interactive exploration
- They are **NOT** part of the main benchmark workflow
- They are **NOT** imported by other code
- Run them manually/interactively for analysis and visualization

## 📊 Available Notebooks

### 1. `learned_index_plot.py`
**Purpose:** Visualize how linear regression fits different data distributions

**What it does:**
- Plots learned model predictions vs actual positions
- Shows accuracy heatmaps across error thresholds
- Compares sequential, uniform, and mixed distributions

**Usage:**
```bash
# As Jupyter notebook (if using jupytext)
jupytext --to notebook learned_index_plot.py
jupyter notebook learned_index_plot.ipynb

# Or run directly in VS Code with Jupyter extension
# Or execute with: python learned_index_plot.py
```

**Generates:**
- Scatter plots of data + regression lines
- Heatmap of prediction accuracy by error threshold
- Comparison plots across distributions

---

### 2. `linear_index_adaptive_plot.py`
**Purpose:** Visualize adaptive window behavior

**What it does:**
- Shows how adaptive windows adjust to data patterns
- Compares different quantile thresholds (0.99, 0.995, 0.999)
- Demonstrates window size adaptation

**Usage:**
```bash
python linear_index_adaptive_plot.py
```

**Generates:**
- Window size distribution plots
- Error profile visualizations
- Adaptive vs fixed window comparisons

---

### 3. `rmi_plot.py`
**Purpose:** Visualize RMI (Recursive Model Index) structure

**What it does:**
- Shows hierarchical model predictions
- Visualizes root model + leaf model decomposition
- Demonstrates how RMI partitions data

**Usage:**
```bash
python rmi_plot.py
```

**Generates:**
- RMI stage visualizations
- Partition boundary plots
- Comparison of single-stage vs multi-stage models

---

## 🔄 Relationship to Main Workflow

```
Main Workflow:
├── benchmarks/benchmark.py ────> Runs comprehensive tests, outputs CSV
├── benchmarks/generate_graphs.py ────> Creates publication-ready graphs
└── benchmarks/statistical_analysis.py ────> Statistical validation

Separate (This Directory):
└── plots/*.py ────> Interactive exploration & visualization tools
```

The **main workflow** uses `generate_graphs.py` for publication-ready graphs.

These **plot scripts** are for:
- Understanding model behavior
- Debugging
- Exploratory analysis
- Creating custom visualizations
- Teaching/presentations

---

## 🛠️ Requirements

These scripts require:
```bash
pip install matplotlib numpy jupytext  # (if using as notebooks)
```

---

## 💡 Tips

**For Publications:** Use `benchmarks/generate_graphs.py` (clean, filtered, publication-ready)

**For Understanding:** Use these scripts (detailed, exploratory, customizable)

**For New Visualizations:** Copy one of these scripts and modify it

---

## 📝 File Format

These files use **Jupytext** format (`.py` with special markers):
- `# %%` marks code cells
- `# %% [markdown]` marks markdown cells
- Can be opened as Jupyter notebooks or run as Python scripts

To convert to `.ipynb`:
```bash
jupytext --to notebook learned_index_plot.py
```

---

## 🚀 Quick Start

```bash
# Navigate to plots directory
cd src/plots

# Run any visualization
python learned_index_plot.py

# Or open in Jupyter/VS Code for interactive exploration
```

---

## ⚡ Common Issues

**Import Error:** Make sure you're running from project root or have `PYTHONPATH` set:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
python src/plots/learned_index_plot.py
```

**No Display:** If running headless, use matplotlib's `Agg` backend:
```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
```

---

**Last Updated:** 2024-11-24  
**Status:** Standalone visualization tools - not part of main benchmark workflow
