"""
===============================================================================
TWO-STAGE RECURSIVE MODEL INDEX (RMI) WITH SHALLOW NEURAL NETWORK ROOT
===============================================================================
Extension of the baseline RMI:
  - Stage 0 (root): shallow neural network (1 hidden layer)
  - Stage 1 (leaves): per-segment linear models

Build process:
  1) Train a small NN to map key -> normalized position [0, 1].
  2) Sort or partition keys based on predicted position.
  3) Fit a linear model (via np.polyfit) per leaf segment.
  4) Store per-segment max absolute error for bounded local search.

Search process (same as linear RMI):
  - Use root NN to predict approximate position.
  - Map to leaf segment.
  - Refine with that leaf’s linear model and binary search locally.

===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.indexes.rmi import RecursiveModelIndex


# ------------------------------------------------------------------------------
# Root model: shallow neural network
# ------------------------------------------------------------------------------
class ShallowNNRoot(nn.Module):
    """Simple feed-forward NN: 1 hidden layer, ReLU activation."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------------------
# Recursive Model Index with Shallow NN root
# ------------------------------------------------------------------------------
class RecursiveModelIndexNN(RecursiveModelIndex):
    """RMI variant that uses a shallow NN as the root model."""

    def __init__(
        self,
        fanout: int = 128,
        hidden_dim: int = 16,
        epochs: int = 200,
        lr: float = 0.01
    ):
        super().__init__(fanout=fanout)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.root_nn = ShallowNNRoot(hidden_dim)

    # --------------------------------------------------------------------------
    # Build
    # --------------------------------------------------------------------------
    def build_from_sorted_array(self, keys: np.ndarray):
        """Fit the shallow NN as the Stage-0 model, then train leaf models."""
        if keys is None or len(keys) == 0:
            super().build_from_sorted_array(np.array([]))
            return

        # Normalize keys and positions
        self.keys = np.asarray(keys)
        self.n = len(self.keys)
        positions = np.arange(self.n, dtype=np.float32) / self.n

        # Convert to tensors
        X = torch.tensor(self.keys.reshape(-1, 1), dtype=torch.float32)
        y = torch.tensor(positions.reshape(-1, 1), dtype=torch.float32)

        # ----------------------------
        # Train shallow NN root model
        # ----------------------------
        opt = optim.Adam(self.root_nn.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            opt.zero_grad()
            pred = self.root_nn(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        # ----------------------------
        # Partition and fit leaves
        # ----------------------------
        with torch.no_grad():
            preds = self.root_nn(X).numpy().flatten()

        # Sort keys by predicted position (coarse approximation)
        order = np.argsort(preds)
        sorted_keys = self.keys[order]

        # Use base RMI’s leaf fitting logic
        super().build_from_sorted_array(sorted_keys)

    # --------------------------------------------------------------------------
    # Search (inherits from base RMI)
    # --------------------------------------------------------------------------
    # Uses same `search()` and `get_memory_usage()` methods as base class.


# ------------------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    keys = np.sort(rng.uniform(0, 1_000_000, 50_000))

    rmi_nn = RecursiveModelIndexNN(fanout=128, hidden_dim=16, epochs=100)
    rmi_nn.build_from_sorted_array(keys)

    # Evaluate on in-distribution queries
    queries = np.concatenate([
        rng.choice(keys, 5_000, replace=True),
        rng.uniform(keys.min(), keys.max(), 5_000),
    ])

    hits = 0
    for q in queries:
        found, _ = rmi_nn.search(q)
        hits += int(found)

    print(f"Sample accuracy: {hits}/{queries.size}")
    print(f"Approx. memory usage: {rmi_nn.get_memory_usage() / 1024:.2f} KB")
