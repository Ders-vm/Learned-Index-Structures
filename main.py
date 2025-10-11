from src.utils.data_loader import DatasetGenerator
from src.benchmarks.benchmark_runner import Benchmark


def main():
    print("🧠 Phase 1 – B-Tree Benchmark\n")

    sizes = [1_000, 10_000, 100_000]   # dataset sizes to test
    for size in sizes:
        print(f"\n{'#'*70}\nTesting {size:,} keys\n{'#'*70}")

        datasets = {
            "Sequential": DatasetGenerator.generate_sequential(size),
            "Uniform": DatasetGenerator.generate_uniform(size)
        }

        for name, keys in datasets.items():
            Benchmark.run(f"{name} ({size:,})", keys)

    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()
