#!/bin/bash
# Runner script for exploratory visualization

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

python src/plots/generate_exploratory_plots.py "$@"
