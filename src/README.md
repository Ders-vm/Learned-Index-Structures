# Learned Index Structures - Benchmarking Suite

A comprehensive benchmarking framework for evaluating learned index structures against traditional baselines, implementing the methodology from Kraska et al. (2018).

## Overview

This project implements and benchmarks learned index structures that use machine learning models to predict data positions, comparing them against traditional B-trees and the Kraska et al. baseline implementations.

### Models Implemented

**Learned Indexes (Novel Contributions):**
- Linear Fixed Window: Single linear regression model with fixed search window
- Linear Adaptive Window: Adaptive window sizing based on error quantiles

**Baselines:**
- B-Tree: Traditional index structure (optimized implementation)
- Kraska Single-Stage: Single linear model from Kraska et al.
- Kraska RMI: Recursive Model Index (2-stage hierarchy)

## Quick Start

### Running Benchmarks

```bash
# Run full benchmark suite
python src/benchmarks/benchmark.py

# Results saved to: results/benchmarks/run_YYYY-MM-DD_HH-MM-SS/master.csv
```

### Analyzing Results

```bash
# Generate statistical summary
python analyze_simple.py

# Generate graphs
python src/benchmarks/generate_graphs.py
```

## Configuration

Edit `src/benchmarks/benchmark.py` to adjust test parameters:

```python
# Dataset sizes to test
DATASET_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# Data distributions
DISTRIBUTIONS = ["seq", "uniform", "mixed"]

# Statistical cycles
REPEAT_CYCLES = 5
```

### Recommended Configurations

**Quick Test (5 minutes):**
```python
DATASET_SIZES = [1_000_000]
REPEAT_CYCLES = 5
```

**Research Scale (8-10 hours):**
```python
DATASET_SIZES = [100_000_000]
REPEAT_CYCLES = 10
```

**Large Scale (10-20 hours):**
```python
DATASET_SIZES = [1_000_000_000]
REPEAT_CYCLES = 2
```

## Benchmark Methodology

### Workload
- Point queries only (single key lookups)
- 50% hit rate (500 existing keys, 500 random queries)
- 1000 queries per test
- Warmup phase to eliminate cold cache effects

### Metrics Collected
- **Lookup Time**: Average query latency (nanoseconds)
- **Build Time**: Index construction time (milliseconds)
- **Memory Usage**: Index footprint (megabytes)
- **Accuracy**: Prediction accuracy for learned indexes

### Statistical Rigor
- Multiple cycles for each configuration (default: 5)
- 95% confidence intervals
- Coefficient of variation tracking
- Outlier detection

## Output

### Statistical Analysis
```
Model                Distribution  Lookup (µs)    ±95% CI   Accuracy
----------------------------------------------------------------------
btree                uniform              8.90       0.90     1.0000
kraska_rmi           uniform             12.35       0.55     1.0000
linear_adaptive      uniform             10.09       0.29     0.5077
linear_fixed         uniform             10.30       0.30     0.3340
```

### Graphs Generated
- `lookup_time_seq.png`: Performance on sequential data
- `lookup_time_uniform.png`: Performance on random data
- `lookup_time_mixed.png`: Performance on clustered data
- `memory_usage.png`: Memory efficiency comparison
- `comparison.png`: Overall performance comparison
- `accuracy_*.png`: Prediction accuracy (learned indexes only)

## Project Structure

```
src/
├── benchmarks/
│   ├── benchmark.py           # Main benchmark runner
│   ├── generate_graphs.py     # Graph generation
│   └── statistical_analysis.py
│
├── indexes/
│   ├── btree_optimized.py           # B-Tree implementation
│   ├── learned_index_optimized.py   # Linear Fixed
│   ├── linear_index_adaptive.py     # Linear Adaptive
│   ├── learned_index_kraska.py      # Kraska baselines
│   └── index_base.py                # Base classes
│
├── utils/
│   └── data_loader.py         # Dataset generation
│
└── tests/
    └── test_indexes.py        # Unit tests

results/
└── benchmarks/
    └── run_YYYY-MM-DD_HH-MM-SS/
        └── master.csv         # Benchmark results

graphs/
├── lookup_time_*.png
├── memory_usage.png
└── comparison.png
```

## Key Features

### Optimizations
- NumPy-based binary search (44x faster than bisect)
- Memory-efficient B-tree with __slots__
- Adaptive error bounds for learned indexes
- Bulk-loading from sorted arrays

### Extensibility
- Clean base class interface for new index types
- Configurable model parameters
- Multiple distribution support
- Customizable workloads

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
```

## Implementation Notes

### B-Tree
- Configurable branch factor (order)
- Bottom-up bulk loading
- Optimized for read-only workloads
- Memory usage: ~16-24 bytes per key

### Learned Indexes
- Linear regression for CDF approximation
- Local binary search within error bounds
- Tracks prediction accuracy and fallback rates
- Memory usage: ~16 bytes (model parameters only)

### Kraska Baselines
- Single-Stage: Direct CDF learning
- RMI: 2-stage model hierarchy [1, 100]
- Error-bounded correction search
- Faithful reproduction of paper methodology

## Performance Expectations

At 100M keys (uniform distribution):
- B-Tree: ~12-15 µs (log(n) overhead)
- Learned indexes: ~9-11 µs (O(1) with small constant)
- Memory: Learned ~1000x smaller than B-Tree

Performance varies by distribution:
- Sequential: Learned indexes excel
- Uniform: Competitive
- Mixed: Tests robustness

## Testing

```bash
# Run unit tests
python -m pytest src/tests/

# Test individual index
python src/tests/test_indexes.py
```

## References

Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018).
The case for learned index structures.
In Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18).

## License

MIT License - See LICENSE file for details

## Authors

Research implementation for learned index structure evaluation.
