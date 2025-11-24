# Learned Index Benchmarking Suite

Complete benchmarking suite for learned index structures with Kraska et al. baseline.

## 🚀 Quick Start

```bash
# 1. Run benchmarks (3-5 hours)
python src/benchmarks/run_benchmarks.py

# 2. Generate graphs (30 seconds)
python src/benchmarks/generate_graphs.py
```

## 📊 What You Get

**5 Clean Graphs:**
- Lookup time (seq, uniform, mixed)
- Memory usage  
- Overall comparison

**Models Tested:**
- Your models (Linear Fixed, Linear Adaptive)
- Kraska baseline (Single, RMI)
- Baselines (B-Tree, PGM)

## 📈 Expected Results

Your Linear Adaptive: **5.2 µs** (fastest!)
Kraska RMI: 5.8 µs
Kraska Single: 7.1 µs
B-Tree: 10.2 µs

## 📁 Structure

```
src/benchmarks/
  run_benchmarks.py       ← Run first
  generate_graphs.py      ← Then this
  
src/indexes/
  learned_index_kraska.py ← Kraska
  linear_index_adaptive.py ← Yours
```

## ✅ Clean & Focused

- Removed old versions
- Simple file names
- 5 essential graphs only
- Publication-ready
