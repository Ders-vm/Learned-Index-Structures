# Benchmark Files - Quick Reference

## 📁 Which File To Use?

### **benchmark.py** ← USE THIS FOR RESEARCH
**Full research benchmark suite**
- Tests ALL models across ALL configurations
- 5 dataset sizes (10K to 1M)
- 3 distributions (seq, uniform, mixed)
- 5 cycles for statistical rigor
- Outputs CSV for analysis
- Runtime: 3-5 hours

**Usage:**
```bash
python benchmarks/benchmark.py
```

**Output:**
```
results/benchmarks/run_YYYY-MM-DD_HH-MM-SS/master.csv
```

---

### **benchmark_single.py** ← FOR QUICK TESTS
**Simple utility for development**
- Test one configuration quickly
- No CSV output
- Quick feedback
- Runtime: seconds

**Usage:**
```python
from benchmarks.benchmark_single import Benchmark
from utils.data_loader import DatasetGenerator

keys = DatasetGenerator.generate_uniform(10000)
Benchmark.run("Quick Test", keys)
```

---

### **generate_graphs.py**
**Creates publication-ready graphs**
- Reads CSV from benchmark.py
- Generates 5 essential graphs
- Filters to key configurations

**Usage:**
```bash
python benchmarks/generate_graphs.py
```

**Output:**
```
graphs/lookup_time_seq.png
graphs/lookup_time_uniform.png
graphs/lookup_time_mixed.png
graphs/memory_usage.png
graphs/comparison.png
```

---

### **statistical_analysis.py**
**Computes p-values and confidence intervals**
- Reads CSV from benchmark.py
- Focuses on 1M keys (best scale)
- Statistical significance testing

**Usage:**
```bash
python benchmarks/statistical_analysis.py
```

---

## 🚀 Typical Workflow

```bash
# 1. Run full benchmarks (3-5 hours, run overnight)
python benchmarks/benchmark.py

# 2. Generate graphs (30 seconds)
python benchmarks/generate_graphs.py

# 3. Statistical analysis (10 seconds)
python benchmarks/statistical_analysis.py

# 4. View results
explorer graphs
```

---

## 📊 What Each Does

| File | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `benchmark.py` | Full research suite | 3-5 hrs | CSV data |
| `generate_graphs.py` | Create visualizations | 30 sec | 5 PNG files |
| `statistical_analysis.py` | Compute stats | 10 sec | Console output |
| `benchmark_single.py` | Quick dev tests | Seconds | Console only |

---

## 💡 Pro Tips

**For your research paper:**
1. Run `benchmark.py` once (overnight)
2. Use `generate_graphs.py` for figures
3. Use `statistical_analysis.py` for numbers

**During development:**
- Use `benchmark_single.py` for quick tests
- Don't wait 5 hours to test one change!

---

## ✅ File Name Clarity

**OLD (confusing):**
- `run_benchmarks.py` - Which one runs what?
- `benchmark_runner.py` - Sounds like it runs benchmarks?

**NEW (clear):**
- `benchmark.py` - THE comprehensive benchmark
- `benchmark_single.py` - Quick utility for single tests

Much better! 🎯
