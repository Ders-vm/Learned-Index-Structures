Learned Index Structures - Benchmark Implementation
A comprehensive benchmarking framework for evaluating learned index structures against traditional B-Tree implementations, based on the work of Kraska et al. (2018).
Project Overview
This project implements and compares several index structures across different data distributions and scales. The goal is to empirically evaluate the performance characteristics of learned indexes and understand when they provide advantages over traditional approaches.
Implemented Index Structures
Traditional Baseline

B-Tree: Optimized B-Tree implementation with configurable order

Learned Index Approaches

Linear Fixed: Single linear regression model with fixed search window
Linear Adaptive: Single linear model with adaptive window sizing based on prediction error quantiles
Kraska Single-Stage: Baseline learned index from Kraska et al. with adaptive error-bounded search
Kraska RMI: Recursive Model Index with hierarchical model structure (1 root model + 100 expert models)

Key Findings
Scaling Behavior (10M vs 100M Keys)
At 10 million keys, simpler models demonstrate lower overhead and competitive performance. However, at 100 million keys, the performance characteristics change significantly:
Sequential Distribution:

Learned indexes maintain consistent performance (5-8 microseconds)
B-Tree performance degrades substantially (21 microseconds)
Simple models achieve 4x speedup over traditional approaches

Uniform Distribution:

Model complexity becomes more valuable at scale
Recursive Model Index demonstrates advantages on difficult distributions
Adaptive window sizing proves essential for maintaining performance

Mixed Distribution:

Performance varies based on data clustering patterns
Simpler models remain competitive across scales
Window sizing strategy critical for correctness

Window Sizing Strategy
Fixed window approaches that perform adequately at smaller scales encounter serious limitations at larger scales:

Fixed windows work reasonably well up to 10M keys
Performance degrades significantly at 100M keys on non-sequential data
Adaptive window sizing maintains stable performance across scales

Implementation Considerations
The benchmarks reveal important implementation trade-offs:

Python overhead affects complex model structures disproportionately
Cache locality and function call overhead become significant factors
Production deployments would benefit from compiled implementations (C++ or Rust)

Repository Structure
.
├── main.py                          # Main benchmark runner
├── generate_all_plots.py            # Standalone graph generation
├── src/
│   ├── indexes/
│   │   ├── btree_optimized.py      # B-Tree implementation
│   │   ├── learned_index_optimized.py   # Linear Fixed
│   │   ├── linear_index_adaptive.py     # Linear Adaptive
│   │   └── learned_index_kraska.py      # Kraska Single & RMI
│   └── utils/
│       └── data_loader.py          # Dataset generation utilities
├── results/
│   └── benchmarks/                 # Benchmark output (CSV)
└── graphs/                         # Generated visualizations
    ├── 1M/                        # Graphs for 1M keys
    ├── 10M/                       # Graphs for 10M keys
    └── 100M/                      # Graphs for 100M keys
Running Benchmarks
Basic Usage
bashpython main.py
Configuration
Edit main.py to adjust benchmark parameters:
python# Dataset configuration
DATASET_SIZES = [1_000_000, 10_000_000, 100_000_000]
DISTRIBUTIONS = ["seq", "uniform", "mixed"]
REPEAT_CYCLES = 5

# Model parameters
FIXED_WINDOWS = [512]
ADAPTIVE_Q = [0.99]
ADAPTIVE_MIN_W = [16]
Output
Benchmarks generate:

CSV data files in results/benchmarks/
Performance graphs organized by dataset size in graphs/
Statistical analysis of lookup times, build times, and accuracy

Graph Generation
Graphs are automatically generated during benchmark runs. To regenerate graphs from existing data:
bashpython generate_all_plots.py
This creates graphs for each dataset size:

Lookup time comparisons per distribution
Prediction accuracy metrics
Build time analysis
Combined distribution comparisons

Data Distributions
Sequential
Keys are uniformly distributed integers in sorted order. Represents best-case scenario for learned indexes.
Uniform
Keys are randomly sampled from a uniform distribution and sorted. Represents typical random access patterns.
Mixed
Keys follow a mixture of dense and sparse regions. Represents realistic workloads with clustering.
Performance Metrics
The benchmark tracks:

Lookup Time: Time per search operation (microseconds)
Build Time: Index construction time (milliseconds)
Memory Usage: Index memory footprint (bytes/MB)
Search Accuracy: Correctness of search results (must be 100%)
Prediction Accuracy: Quality of position predictions (varies by distribution)

Accuracy Metrics
Two distinct accuracy metrics are reported:
Search Accuracy: Whether searches return correct results (found/not found). Must always be 100% for valid indexes.
Prediction Accuracy: How close position predictions are to actual positions. This varies by distribution and model, but does not affect search correctness due to error-bounded search windows.
Requirements
Python 3.8+
numpy
pandas
matplotlib
Install dependencies:
bashpip install numpy pandas matplotlib
Preventing System Sleep (Overnight Runs)
For long benchmark runs, disable system sleep:
powershell# Windows PowerShell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
Restore after completion:
powershellpowercfg /change standby-timeout-ac 15
Implementation Notes
Language Overhead
The Python implementation demonstrates the fundamental trade-offs between model complexity and execution overhead. The Kraska paper used C++ implementations, which eliminates much of the function call and memory access overhead observed here.
At moderate scales (10M keys), this overhead causes simpler models to outperform more complex hierarchical approaches. At larger scales (100M keys), the benefits of improved predictions begin to outweigh overhead costs on difficult distributions.
Production Considerations
For production deployments:

Consider compiled implementations (C++, Rust, or Cython) for performance-critical paths
Adaptive window sizing is essential for correctness across data distributions
Model complexity should match dataset characteristics and scale
Profile carefully to understand whether model overhead or search costs dominate

References
Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). The case for learned index structures. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18). https://arxiv.org/abs/1712.01208
Project Status
This implementation was developed for academic research purposes to empirically evaluate learned index structures across different scales and distributions. The findings demonstrate important scaling behaviors and implementation trade-offs relevant to practical deployment decisions.
License
This project is for academic use.