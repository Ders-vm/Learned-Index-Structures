from src.utils.data_loader import DatasetGenerator
from src.benchmarks.benchmark_runner import Benchmark


def main():
    print("🧠 Phase 1 – B-Tree Benchmark\n")

    #sizes = [10_000, 100_000, 1_000_000]   # dataset sizes to test TODO uncomment at the end
    sizes = [100_000]  # Temporarily limit to 100k for quicker tests
    for size in sizes:
        print(f"\n{'#'*70}\nTesting {size:,} keys\n{'#'*70}")

        # Generate datasets, sequential and uniformly distributed (random)
        datasets = {
            "Sequential": DatasetGenerator.generate_sequential(size),
            "Uniform": DatasetGenerator.generate_uniform(size),
            "Mixed": DatasetGenerator.generate_mixed(size)
        }

        for name, keys in datasets.items():
            Benchmark.run(f"{name} ({size:,})", keys)

    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()
