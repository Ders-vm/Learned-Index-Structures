#!/usr/bin/env python3
"""
Exploratory Visualization Runner
Generates exploratory plots showing how learned indexes work on different datasets
"""

import sys
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from src.indexes.learned_index_optimized import LearnedIndexOptimized
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.indexes.btree_optimized import BTreeOptimized
from src.utils.data_loader import DatasetGenerator

# Output directory
OUTPUT_DIR = os.path.join(project_root, "exploratory_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_SIZE = 100_000

print("="*60)
print("EXPLORATORY VISUALIZATION GENERATOR")
print("="*60)
print(f"\nGenerating plots for {DATASET_SIZE:,} keys...")
print(f"Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# 1. LINEAR MODEL FIT VISUALIZATION
# ============================================================================

def plot_model_fits():
    """Show how linear models fit different data distributions."""
    print("Generating model fit visualizations...")
    
    datasets = {
        'Sequential': DatasetGenerator.generate_sequential(DATASET_SIZE),
        'Uniform': DatasetGenerator.generate_uniform(DATASET_SIZE),
        'Mixed': DatasetGenerator.generate_mixed(DATASET_SIZE)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, data) in zip(axes, datasets.items()):
        # Build learned index
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(data)
        
        # Plot data points
        ax.scatter(data, np.arange(len(data)), s=1, alpha=0.5, label='Data')
        
        # Plot learned model
        predicted = index.a * data + index.b
        ax.plot(data, predicted, 'r--', linewidth=2, label='Learned Model')
        
        ax.set_title(f'{name} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Keys', fontsize=12)
        ax.set_ylabel('Positions', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'model_fits.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: model_fits.png")


# ============================================================================
# 2. PREDICTION ACCURACY HEATMAP
# ============================================================================

def plot_accuracy_heatmap():
    """Show prediction accuracy at different error thresholds."""
    print("\nGenerating accuracy heatmap...")
    
    datasets = {
        'Sequential': DatasetGenerator.generate_sequential(DATASET_SIZE),
        'Uniform': DatasetGenerator.generate_uniform(DATASET_SIZE),
        'Mixed': DatasetGenerator.generate_mixed(DATASET_SIZE)
    }
    
    thresholds = [1, 10, 50, 100, 500, 1000]
    matrix = np.zeros((len(datasets), len(thresholds)))
    
    for i, (name, data) in enumerate(datasets.items()):
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(data)
        
        predicted = index.a * data + index.b
        errors = np.abs(predicted - np.arange(len(data)))
        
        for j, threshold in enumerate(thresholds):
            accuracy = np.sum(errors <= threshold) / len(data) * 100
            matrix[i, j] = accuracy
    
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([f'<={t}' for t in thresholds])
    ax.set_yticklabels(datasets.keys())
    
    # Add percentages
    for i in range(len(datasets)):
        for j in range(len(thresholds)):
            ax.text(j, i, f'{matrix[i, j]:.1f}%',
                   ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Error Threshold (positions)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Prediction Accuracy by Error Threshold', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'accuracy_heatmap.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: accuracy_heatmap.png")


# ============================================================================
# 3. ERROR DISTRIBUTION
# ============================================================================

def plot_error_distribution():
    """Show distribution of prediction errors."""
    print("\nGenerating error distribution plots...")
    
    datasets = {
        'Sequential': DatasetGenerator.generate_sequential(DATASET_SIZE),
        'Uniform': DatasetGenerator.generate_uniform(DATASET_SIZE),
        'Mixed': DatasetGenerator.generate_mixed(DATASET_SIZE)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, data) in zip(axes, datasets.items()):
        index = LearnedIndexOptimized()
        index.build_from_sorted_array(data)
        
        predicted = index.a * data + index.b
        errors = np.abs(predicted - np.arange(len(data)))
        
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.1f}')
        ax.axvline(np.median(errors), color='g', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(errors):.1f}')
        
        ax.set_title(f'{name} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Error (positions)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'error_distribution.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: error_distribution.png")


# ============================================================================
# 4. ADAPTIVE VS FIXED WINDOW
# ============================================================================

def plot_adaptive_vs_fixed():
    """Compare adaptive and fixed window sizing."""
    print("\nGenerating adaptive vs fixed comparison...")
    
    data = DatasetGenerator.generate_mixed(DATASET_SIZE)
    
    # Fixed window
    fixed = LearnedIndexOptimized(window=512)
    fixed.build_from_sorted_array(data)
    
    # Adaptive window
    adaptive = LinearIndexAdaptive(quantile=0.99, min_window=16)
    adaptive.build_from_sorted_array(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot both
    for ax, index, name in [(ax1, fixed, 'Fixed Window (512)'), 
                             (ax2, adaptive, f'Adaptive Window ({adaptive.window})')]:
        ax.scatter(data, np.arange(len(data)), s=1, alpha=0.5, label='Data')
        predicted = index.a * data + index.b
        ax.plot(data, predicted, 'r--', linewidth=2, label='Model')
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Keys', fontsize=12)
        ax.set_ylabel('Positions', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'adaptive_vs_fixed.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: adaptive_vs_fixed.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    plot_model_fits()
    plot_accuracy_heatmap()
    plot_error_distribution()
    plot_adaptive_vs_fixed()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nExploratory plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - model_fits.png")
    print("  - accuracy_heatmap.png")
    print("  - error_distribution.png")
    print("  - adaptive_vs_fixed.png")
    print()
