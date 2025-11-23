from src.utils.data_loader import DatasetGenerator
from src.benchmarks.benchmark_runner import Benchmark
from src.ml.shallow_nn_rmi import RecursiveModelIndexNN


def main():
    print("🧠 Phase 1 – Optimized + Neural Index Benchmarks\n")

    # Dataset sizes you want to test
    sizes = [1_000_000]

    for size in sizes:
        print(f"\n{'#' * 70}")
        print(f"Testing {size:,} keys")
        print(f"{'#' * 70}")

        # Generate datasets
        datasets = {
            "Sequential": DatasetGenerator.generate_sequential(size),
            "Uniform": DatasetGenerator.generate_uniform(size),
            "Mixed": DatasetGenerator.generate_mixed(size)
        }

        # Run standard benchmarks (B-Tree, Linear, Adaptive, PGM, RMI)
        for name, keys in datasets.items():
            Benchmark.run(f"{name} ({size:,})", keys)

            # ------------------------------------------------------
            # NEURAL RMI BENCHMARK
            # ------------------------------------------------------
            print("\n-- Neural RMI (Shallow NN Root) --")
            rmi_nn = RecursiveModelIndexNN(
                fanout=512,      # workable value for 1M keys
                hidden_dim=32,   # slightly larger NN for 1M dataset
                epochs=30,       # fast enough but still learns well
                lr=0.01
            )

            # Build time
            import time
            start = time.perf_counter()
            rmi_nn.build_from_sorted_array(keys)
            build_ms = (time.perf_counter() - start) * 1000

            # Lookup time
            import numpy as np
            existing = np.random.choice(keys, 500, replace=False)
            randoms = np.random.uniform(keys.min(), keys.max(), 500)
            queries = np.concatenate([existing, randoms])
            np.random.shuffle(queries)

            start = time.perf_counter()
            hits = 0
            for q in queries:
                found = rmi_nn.search(q)
                hits += int(found)
            lookup_ns = (time.perf_counter() - start) * 1e9 / len(queries)

            # Accuracy (RMI only returns bool so this is approximate)
            accuracy = hits / len(queries)

            # Memory (RMI has method, NN root does not)
            try:
                mem_mb = rmi_nn.get_memory_usage() / (1024 * 1024)
            except:
                mem_mb = None

            print(
                f"NeuralRMI | Build: {build_ms:8.2f} ms | "
                f"Lookup: {lookup_ns:8.2f} ns | "
                f"Acc: {accuracy:.2%} | "
                f"Mem: {mem_mb:.3f} MB"
            )

    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()
 