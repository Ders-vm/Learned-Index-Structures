import numpy as np

class DatasetGenerator:
    """Generate small test datasets for Phase 1 B-Tree benchmarks."""

    @staticmethod
    def generate_uniform(size: int, min_val: int = 0, max_val: int = 1_000_000) -> np.ndarray:
        """Uniformly distributed random keys."""
        keys = np.random.uniform(min_val, max_val, size)
        return np.sort(keys)

    @staticmethod
    def generate_sequential(size: int, start: int = 0, step: int = 1) -> np.ndarray:
        """Sequential keys (0, 1, 2, …)."""
        return np.arange(start, start + size * step, step, dtype=np.float64)

    @staticmethod
    def generate_mixed(size: int) -> np.ndarray:
        """Mixed distribution: uniform + two clusters."""
        uniform = np.random.uniform(0, 1_000_000, int(size * 0.4))
        cluster1 = np.random.normal(250_000, 10_000, int(size * 0.3))
        cluster2 = np.random.normal(750_000, 10_000, int(size * 0.3))
        keys = np.concatenate([uniform, cluster1, cluster2])
        return np.sort(np.unique(keys))[:size]


# -----------------------------------------------------------------------------
# Quick-run tester (works directly inside VS Code)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Testing DatasetGenerator...\n")

    seq = DatasetGenerator.generate_sequential(10)
    print("Sequential (10):", seq)

    uniform = DatasetGenerator.generate_uniform(10)
    print("\nUniform (10):", uniform)

    mixed = DatasetGenerator.generate_mixed(10)
    print("\nMixed (10):", mixed)

    print("\n✅ DatasetGenerator test complete.")
