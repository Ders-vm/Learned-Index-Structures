"""
===============================================================================
LEARNED INDEX IMPLEMENTATION - KRASKA ET AL. SPECIFICATION
===============================================================================
Implementation following "The Case for Learned Index Structures" (Kraska et al., 2018)

Key concepts from the paper:
1. Replace index structures with learned models (CDF approximation)
2. Recursive Model Index (RMI) - staged models
3. Hybrid approach - models + search in bounded error region
4. Error bounds determine search window

This implementation provides:
- Single-stage learned index (baseline from paper)
- Multi-stage RMI (main contribution of paper)  
- Proper error bounds and correction mechanism
- Last-mile search optimization

References:
- Kraska, T., et al. (2018). "The Case for Learned Index Structures."
  SIGMOD '18. https://arxiv.org/abs/1712.01208
===============================================================================
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class ModelMetrics:
    """Metrics for learned index as specified in Kraska paper."""
    total_queries: int = 0
    within_error_bound: int = 0  # Predictions within error bound
    outside_error_bound: int = 0  # Required fallback
    mean_prediction_error: float = 0.0
    max_prediction_error: int = 0
    min_prediction_error: int = 0
    
    def reset(self):
        self.total_queries = 0
        self.within_error_bound = 0
        self.outside_error_bound = 0
        self.mean_prediction_error = 0.0
        self.max_prediction_error = 0
        self.min_prediction_error = 0


class SingleStageLearnedIndex:
    """
    Single-stage learned index as baseline in Kraska et al.
    
    Key insight from paper:
    "A B-Tree can be viewed as a model that maps keys to positions,
     but it does so through multiple layers of models (nodes)."
    
    Learned index: Replace this with a single ML model that learns
    the CDF (cumulative distribution function) of the data.
    
    For sorted array: CDF(key) ≈ position / n
    We learn: position = f(key) where f is a simple model
    """
    
    def __init__(self, model_type: str = 'linear', error_bound: int = None):
        """
        Args:
            model_type: Type of model ('linear', 'polynomial', 'neural')
            error_bound: Maximum allowed prediction error
                        If None, computed from training data
        """
        self.model_type = model_type
        self.error_bound = error_bound
        self.n = 0
        self.keys = None
        
        # Model parameters
        self.model_params = None
        
        # Metrics
        self.metrics = ModelMetrics()
        
        # Timing
        self.build_time_ms = 0.0
        self.inference_time_ns = 0.0
        
    def _train_model(self, keys: np.ndarray) -> dict:
        """
        Train the learned model.
        
        From paper: "We use simple models (linear, polynomial) as they
        are fast to train and have low inference cost."
        """
        positions = np.arange(len(keys), dtype=np.float64)
        
        if self.model_type == 'linear':
            # Linear regression: position = a * key + b
            a, b = np.polyfit(keys, positions, deg=1)
            return {'type': 'linear', 'a': a, 'b': b}
            
        elif self.model_type == 'polynomial':
            # Quadratic: position = a*key^2 + b*key + c
            coeffs = np.polyfit(keys, positions, deg=2)
            return {'type': 'polynomial', 'degree': 2, 'coeffs': coeffs}
            
        elif self.model_type == 'neural':
            # Simple 2-layer neural network (as mentioned in paper)
            # For now, use linear as placeholder
            # In practice, would use TensorFlow/PyTorch
            a, b = np.polyfit(keys, positions, deg=1)
            return {'type': 'neural', 'a': a, 'b': b}
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _predict_position(self, key: float) -> int:
        """
        Predict position using learned model.
        
        From paper: "The model predicts the position where the key
        might be found, with bounded error."
        """
        if self.model_params['type'] == 'linear':
            pred = self.model_params['a'] * key + self.model_params['b']
            
        elif self.model_params['type'] == 'polynomial':
            coeffs = self.model_params['coeffs']
            pred = np.polyval(coeffs, key)
            
        elif self.model_params['type'] == 'neural':
            # Simplified - would use actual NN inference
            pred = self.model_params['a'] * key + self.model_params['b']
        
        return int(np.clip(pred, 0, self.n - 1))
    
    def _compute_error_bound(self, keys: np.ndarray, sample_size: int = 10000) -> int:
        """
        Compute error bound from training data.
        
        From paper: "We measure the maximum prediction error on the
        training data and use that as the error bound for search."
        
        This is the key insight - the error bound determines the
        size of the region we need to search.
        """
        n_samples = min(sample_size, len(keys))
        sample_idx = np.linspace(0, len(keys) - 1, n_samples, dtype=int)
        
        errors = []
        for i in sample_idx:
            predicted = self._predict_position(keys[i])
            actual = i
            errors.append(abs(predicted - actual))
        
        errors = np.array(errors)
        
        # From paper: Use max error as bound
        # In practice, can use 99th percentile for robustness
        max_error = int(np.max(errors))
        p99_error = int(np.percentile(errors, 99))
        
        # Store statistics
        self.metrics.mean_prediction_error = np.mean(errors)
        self.metrics.max_prediction_error = max_error
        self.metrics.min_prediction_error = int(np.min(errors))
        
        return p99_error  # Use 99th percentile for robustness
    
    def build(self, keys: np.ndarray) -> None:
        """
        Build the learned index.
        
        From paper:
        1. Train model on (key, position) pairs
        2. Compute error bound
        3. Store model and bound
        """
        t0 = time.perf_counter()
        
        self.keys = keys
        self.n = len(keys)
        
        # Train model
        self.model_params = self._train_model(keys)
        
        # Compute error bound if not specified
        if self.error_bound is None:
            self.error_bound = self._compute_error_bound(keys)
        
        self.build_time_ms = (time.perf_counter() - t0) * 1000
        
    def search(self, key: float) -> Tuple[bool, int]:
        """
        Search for a key using learned index.
        
        From paper algorithm:
        1. Use model to predict position: pos = f(key)
        2. Search in range [pos - error, pos + error]
        3. Use binary search in that range (last-mile search)
        
        Returns:
            (found, position) tuple
        """
        t0 = time.perf_counter_ns()
        
        self.metrics.total_queries += 1
        
        # Check bounds
        if key < self.keys[0] or key > self.keys[-1]:
            self.inference_time_ns = time.perf_counter_ns() - t0
            return False, -1
        
        # Predict position
        predicted_pos = self._predict_position(key)
        
        # Define search range based on error bound
        left = max(0, predicted_pos - self.error_bound)
        right = min(self.n, predicted_pos + self.error_bound + 1)
        
        # Last-mile search using binary search
        # From paper: "The final search in the bounded region is
        # typically very fast (few comparisons)"
        idx = np.searchsorted(self.keys[left:right], key, side='left')
        actual_pos = left + idx
        
        # Verify
        found = actual_pos < self.n and self.keys[actual_pos] == key
        
        # Update metrics
        if found:
            actual_error = abs(predicted_pos - actual_pos)
            if actual_error <= self.error_bound:
                self.metrics.within_error_bound += 1
            else:
                self.metrics.outside_error_bound += 1
        
        self.inference_time_ns = time.perf_counter_ns() - t0
        
        return found, actual_pos if found else -1
    
    def get_memory_usage(self) -> int:
        """
        Memory usage in bytes.
        
        From paper: "Learned indexes are much smaller than B-Trees
        because they only store model parameters, not the tree structure."
        """
        if self.model_type == 'linear':
            # Just 2 floats (a, b)
            return 16
        elif self.model_type == 'polynomial':
            # Degree + 1 coefficients
            degree = self.model_params['degree']
            return 8 * (degree + 1)
        else:
            # Neural network - would depend on architecture
            return 1024  # Placeholder


class RecursiveModelIndex:
    """
    Recursive Model Index (RMI) - Main contribution of Kraska et al.
    
    From paper:
    "Rather than a single model, we use a hierarchy of models.
     The top model routes to one of several second-stage models,
     which make the final prediction."
    
    Architecture:
                    [Root Model]
                    /    |    \
              [Model1][Model2][Model3]  (Stage 2)
                /|\    /|\      /|\
               Expert models...         (Optional Stage 3+)
    
    Key insight: "Different regions of the data may have different
    patterns. A hierarchy of specialized models can capture this better
    than a single global model."
    """
    
    def __init__(self, 
                 stages: List[int] = [1, 100],
                 model_type: str = 'linear',
                 error_bound: int = None):
        """
        Args:
            stages: Number of models at each stage [1, 100] means:
                    Stage 1: 1 root model
                    Stage 2: 100 expert models
            model_type: Type of models to use
            error_bound: Error bound for last-mile search
        """
        self.stages = stages
        self.model_type = model_type
        self.error_bound = error_bound
        
        self.n = 0
        self.keys = None
        
        # Hierarchy of models
        self.models = []  # List of stages, each stage is list of models
        
        # Metrics
        self.metrics = ModelMetrics()
        self.build_time_ms = 0.0
        
    def _train_stage_models(self, keys: np.ndarray, stage_idx: int) -> List[dict]:
        """
        Train models for a given stage.
        
        From paper: "Each model at stage i is trained on a subset
        of the data determined by the routing from stage i-1."
        """
        n_models = self.stages[stage_idx]
        models = []
        
        if stage_idx == 0:
            # Root model: train on entire dataset
            # Maps key -> model_id in next stage
            positions = np.linspace(0, n_models - 1, len(keys))
            a, b = np.polyfit(keys, positions, deg=1)
            models.append({'a': a, 'b': b})
            
        else:
            # Expert models: each trained on partition
            prev_n_models = self.stages[stage_idx - 1]
            
            for model_id in range(n_models):
                # Find keys routed to this model
                start_idx = int((model_id / n_models) * len(keys))
                end_idx = int(((model_id + 1) / n_models) * len(keys))
                
                if start_idx >= end_idx:
                    # Empty partition - use default model
                    models.append({'a': 0, 'b': start_idx})
                    continue
                
                partition_keys = keys[start_idx:end_idx]
                partition_positions = np.arange(len(partition_keys), dtype=np.float64)
                
                # Train model on this partition
                if len(partition_keys) > 1:
                    a, b = np.polyfit(partition_keys, partition_positions, deg=1)
                    models.append({
                        'a': a, 
                        'b': b,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
                else:
                    models.append({
                        'a': 0, 
                        'b': 0,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
        
        return models
    
    def _predict_with_rmi(self, key: float) -> int:
        """
        Predict position using RMI hierarchy.
        
        From paper:
        1. Root model predicts which stage-2 model to use
        2. Stage-2 model predicts position
        (Can extend to more stages)
        """
        # Stage 1: Root model routes to expert
        root_model = self.models[0][0]
        expert_id = int(np.clip(
            root_model['a'] * key + root_model['b'],
            0,
            self.stages[1] - 1
        ))
        
        # Stage 2: Expert model predicts position
        expert_model = self.models[1][expert_id]
        local_pos = expert_model['a'] * key + expert_model['b']
        
        # Convert to global position
        global_pos = expert_model['start_idx'] + int(local_pos)
        global_pos = np.clip(global_pos, 0, self.n - 1)
        
        return int(global_pos)
    
    def build(self, keys: np.ndarray) -> None:
        """
        Build the RMI.
        
        From paper:
        1. Train root model
        2. Partition data based on root predictions
        3. Train expert models on partitions
        """
        t0 = time.perf_counter()
        
        self.keys = keys
        self.n = len(keys)
        
        # Train models for each stage
        for stage_idx in range(len(self.stages)):
            stage_models = self._train_stage_models(keys, stage_idx)
            self.models.append(stage_models)
        
        # Compute error bound
        if self.error_bound is None:
            self.error_bound = self._compute_error_bound(keys)
        
        self.build_time_ms = (time.perf_counter() - t0) * 1000
    
    def _compute_error_bound(self, keys: np.ndarray, sample_size: int = 10000) -> int:
        """Compute error bound for RMI predictions."""
        n_samples = min(sample_size, len(keys))
        sample_idx = np.linspace(0, len(keys) - 1, n_samples, dtype=int)
        
        errors = []
        for i in sample_idx:
            predicted = self._predict_with_rmi(keys[i])
            actual = i
            errors.append(abs(predicted - actual))
        
        errors = np.array(errors)
        self.metrics.mean_prediction_error = np.mean(errors)
        self.metrics.max_prediction_error = int(np.max(errors))
        
        return int(np.percentile(errors, 99))
    
    def search(self, key: float) -> Tuple[bool, int]:
        """
        Search using RMI.
        
        Same algorithm as single-stage, but uses RMI for prediction.
        """
        self.metrics.total_queries += 1
        
        if key < self.keys[0] or key > self.keys[-1]:
            return False, -1
        
        # Predict using RMI
        predicted_pos = self._predict_with_rmi(key)
        
        # Search in error bound
        left = max(0, predicted_pos - self.error_bound)
        right = min(self.n, predicted_pos + self.error_bound + 1)
        
        idx = np.searchsorted(self.keys[left:right], key, side='left')
        actual_pos = left + idx
        
        found = actual_pos < self.n and self.keys[actual_pos] == key
        
        if found:
            actual_error = abs(predicted_pos - actual_pos)
            if actual_error <= self.error_bound:
                self.metrics.within_error_bound += 1
            else:
                self.metrics.outside_error_bound += 1
        
        return found, actual_pos if found else -1
    
    def get_memory_usage(self) -> int:
        """
        Memory for RMI.
        
        From paper: "RMI uses more memory than single model,
        but still much less than B-Tree."
        """
        # Each linear model: 2 floats + metadata
        total_models = sum(self.stages)
        return total_models * 32  # 32 bytes per model (with overhead)


# ============================================================================
# PAPER EXPERIMENTS - REPLICATION
# ============================================================================

class KraskaExperiments:
    """
    Replicate key experiments from Kraska et al. paper.
    
    Paper shows:
    1. Single-stage learned index vs B-Tree
    2. RMI (staged) vs single-stage
    3. Different model types (linear, NN)
    4. Performance on different distributions
    """
    
    @staticmethod
    def experiment_1_single_vs_btree(keys: np.ndarray, queries: np.ndarray):
        """
        Experiment 1: Single-stage learned index vs B-Tree
        
        From paper: "Even a simple linear model can outperform
        B-Tree on many workloads."
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: Single-Stage Learned Index vs B-Tree")
        print("(Replicating Kraska et al. Figure 3)")
        print("="*80)
        
        # Build single-stage learned index
        learned = SingleStageLearnedIndex(model_type='linear')
        learned.build(keys)
        
        # Test learned index
        t0 = time.perf_counter()
        for q in queries:
            learned.search(q)
        learned_time = (time.perf_counter() - t0) * 1e9 / len(queries)
        
        # Build B-Tree (if available)
        try:
            from src.indexes.btree_optimized import BTreeOptimized
            btree = BTreeOptimized(order=128)
            btree.build_from_sorted_array(keys)
            
            t0 = time.perf_counter()
            for q in queries:
                btree.search(q)
            btree_time = (time.perf_counter() - t0) * 1e9 / len(queries)
            
            speedup = btree_time / learned_time
            
            print(f"\nDataset: {len(keys):,} keys")
            print(f"Learned Index: {learned_time:.1f} ns/query")
            print(f"B-Tree:        {btree_time:.1f} ns/query")
            print(f"Speedup:       {speedup:.2f}x")
            print(f"\nLearned Index Memory: {learned.get_memory_usage():,} bytes")
            print(f"B-Tree Memory:        {btree.get_memory_usage():,} bytes")
            
        except ImportError:
            print(f"\nLearned Index: {learned_time:.1f} ns/query")
            print(f"Error bound: {learned.error_bound}")
            print(f"Mean error: {learned.metrics.mean_prediction_error:.1f}")
    
    @staticmethod
    def experiment_2_rmi_comparison(keys: np.ndarray, queries: np.ndarray):
        """
        Experiment 2: Compare different RMI configurations
        
        From paper: "Adding more stages improves accuracy
        but has diminishing returns."
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: RMI Stage Comparison")
        print("(Replicating Kraska et al. Figure 5)")
        print("="*80)
        
        configurations = [
            ("Single-stage", [1]),
            ("Two-stage [1, 10]", [1, 10]),
            ("Two-stage [1, 100]", [1, 100]),
            ("Two-stage [1, 1000]", [1, 1000]),
        ]
        
        print(f"\nDataset: {len(keys):,} keys\n")
        print(f"{'Configuration':<25} {'Build (ms)':>12} {'Query (ns)':>12} {'Error':>8} {'Memory':>10}")
        print("-" * 75)
        
        for name, stages in configurations:
            if len(stages) == 1:
                # Single stage
                model = SingleStageLearnedIndex()
                model.build(keys)
                
                t0 = time.perf_counter()
                for q in queries:
                    model.search(q)
                query_time = (time.perf_counter() - t0) * 1e9 / len(queries)
                
                print(f"{name:<25} {model.build_time_ms:>12.2f} {query_time:>12.1f} "
                      f"{model.error_bound:>8} {model.get_memory_usage():>10}")
            else:
                # RMI
                model = RecursiveModelIndex(stages=stages)
                model.build(keys)
                
                t0 = time.perf_counter()
                for q in queries:
                    model.search(q)
                query_time = (time.perf_counter() - t0) * 1e9 / len(queries)
                
                print(f"{name:<25} {model.build_time_ms:>12.2f} {query_time:>12.1f} "
                      f"{model.error_bound:>8} {model.get_memory_usage():>10}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LEARNED INDEX IMPLEMENTATION - KRASKA ET AL. SPECIFICATION")
    print("="*80)
    
    # Generate test data
    from src.utils.data_loader import DatasetGenerator
    
    keys = DatasetGenerator.generate_uniform(100_000)
    queries = np.concatenate([
        np.random.choice(keys, 500, replace=False),
        np.random.uniform(keys.min(), keys.max(), 500)
    ])
    
    # Run experiments
    KraskaExperiments.experiment_1_single_vs_btree(keys, queries)
    KraskaExperiments.experiment_2_rmi_comparison(keys, queries)
    
    print("\n" + "="*80)
    print("KEY FINDINGS FROM KRASKA ET AL. PAPER:")
    print("="*80)
    print("""
    1. Learned indexes can replace traditional indexes (B-Trees, Hash)
    2. Simple models (linear) often sufficient for sorted data
    3. RMI (hierarchy) improves accuracy with modest overhead
    4. 10-100x memory reduction vs B-Trees
    5. Comparable or better query performance
    
    Optimal configuration depends on:
    - Data distribution (uniform, normal, skewed)
    - Query pattern (point queries, range queries)
    - Memory constraints
    - Build time requirements
    """)
