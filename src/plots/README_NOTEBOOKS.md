# Plot Notebooks - Original Jupyter Versions

These are the original Jupyter notebook files converted to `.py` format using Jupytext.

## Files

- `learned_index_plot.py` - Visualizes learned index predictions on different distributions
- `linear_index_adaptive_plot.py` - Shows adaptive window behavior
- `rmi_plot.py` - Compares RMI with other approaches

## How to Use

### Option 1: Run in Jupyter (Recommended)

These files are designed to be run as Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Then open the .py files in Jupyter
# They will display as notebooks with cell markers
```

### Option 2: Run as Script (No Display)

If you don't have a display (like on a server), you can still run them:

```bash
cd src/plots
python run_plot_notebook.py learned_index_plot.py
```

**Note:** This won't show plots (since `plt.show()` requires a display), but will execute all the code.

### Option 3: Use the Standalone Versions

If you want scripts that save plots to files instead of displaying them, use the standalone versions:

```bash
python src/plots/generate_all_exploratory_plots.py
```

These will save all plots to `graphs/exploratory/` as PNG files.

## What Changed

**Minimal fixes applied:**
- Import `learned_index_optimized` instead of `learned_index` (module was renamed)
- Import `btree_optimized` instead of `btree` (module was renamed)
- Fixed Windows line endings to Unix line endings

**Everything else is original:**
- All code logic unchanged
- All plots unchanged
- All comments unchanged
- Jupyter notebook structure preserved

## Structure

The files use Jupyter cell markers:
- `# %%` - Code cell
- `# %% [markdown]` - Markdown cell

This allows them to work both as notebooks and as Python scripts.

## If You Want Interactive Plots

Install Jupyter and run them there:

```bash
pip install jupyter matplotlib
jupyter notebook
```

Then open any of the `.py` files and they'll render as notebooks with interactive plots.
