#!/usr/bin/env python3
"""
Runner for Jupyter-style plot notebooks

These files are designed to be run as Jupyter notebooks but can also
be executed as scripts with this runner that sets up the paths correctly.

Usage:
    python run_plot_notebook.py learned_index_plot.py
    python run_plot_notebook.py linear_index_adaptive_plot.py
    python run_plot_notebook.py rmi_plot.py
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now run the notebook file
if len(sys.argv) < 2:
    print("Usage: python run_plot_notebook.py <notebook.py>")
    print("\nAvailable notebooks:")
    print("  - learned_index_plot.py")
    print("  - linear_index_adaptive_plot.py")
    print("  - rmi_plot.py")
    sys.exit(1)

notebook_file = sys.argv[1]
notebook_path = os.path.join(os.path.dirname(__file__), notebook_file)

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_file} not found")
    sys.exit(1)

print(f"Running {notebook_file}...")
print(f"Project root: {project_root}")
print()

# Execute the notebook file
with open(notebook_path, 'r') as f:
    code = f.read()
    exec(code)
