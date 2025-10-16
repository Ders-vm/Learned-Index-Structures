from src.utils.data_loader import DatasetGenerator
from src.benchmarks.benchmark_runner import Benchmark

def main():
    print("🧠 Phase 2–3 – Learned Index & RMI Benchmarks (Python vs C++)\n")

    # Define dataset sizes to test
    sizes = [10_000, 100_000, 1_000_000]

    for size in sizes:
        print(f"\n{'#' * 70}")
        print(f"Testing {size:,} keys")
        print(f"{'#' * 70}\n")

        # Generate datasets
        datasets = {
            "Sequential": DatasetGenerator.generate_sequential(size),
            "Uniform": DatasetGenerator.generate_uniform(size),
            "Mixed": DatasetGenerator.generate_mixed(size)
        }

        # Run benchmarks for each dataset
        for name, keys in datasets.items():
            Benchmark.run(f"{name} ({size:,})", keys)

    print("\n✅ Phase 2–3 complete. Benchmarks include:")
    print("   • Python BTree, LinearModel, RMI")
    print("   • C++  BTree, LinearModel, RMI (if compiled)\n")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Could not import one or more C++ modules.")
        print("Make sure you've built the C++ targets via CMake (btree_cpp, linear_model_cpp, rmi_cpp).")
        print(f"Details: {e}")
