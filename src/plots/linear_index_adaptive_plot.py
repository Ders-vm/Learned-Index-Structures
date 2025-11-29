# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: database-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# Linear Index Adaptive Visualization Module
# ===============================================================================
# This module will generate visualizations in order to illustrate the accuracy
# of our implemented linear index adaptive structure.

# %%
## This section will contain all the nessecary imports for the notebook

import sys, os

# Allow for absolute imported paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from src.indexes.linear_index_adaptive import LinearIndexAdaptive
from src.utils.data_loader import DatasetGenerator


# %%
# Constants to be used throughout the notebook

# Same size for the datasets to keep consistency
DATASET_SIZE = 100_000

# Output directory for plots
OUTPUT_DIR = os.path.join(project_root, 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %% [markdown]
# ## This section will be regarding the uniform dataset

# %%
def plot_uniform():
    # Get the data from our loader
    data_loader_uniform = DatasetGenerator.generate_uniform(DATASET_SIZE)

    # Build our learned index that has been created
    linear_index_uniform = LinearIndexAdaptive()
    linear_index_uniform.build_from_sorted_array(data_loader_uniform)

    # Plot the dataset points with the learned index
    plt.figure(figsize=(10, 6))
    plt.scatter(data_loader_uniform, np.arange(len(data_loader_uniform)), s=1, label='Uniform Data', alpha=0.5)
    predicted_positions = linear_index_uniform.a * data_loader_uniform + linear_index_uniform.b
    plt.plot(data_loader_uniform, predicted_positions, color='red', label='Linear Index Adaptive Model', linewidth=2, linestyle='--')
    plt.title('Linear Index Adaptive Model on Uniform Dataset')
    plt.xlabel('Keys')
    plt.ylabel('Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_uniform.png"), dpi=150, bbox_inches="tight"); plt.close()
plot_uniform()


# %% [markdown]
# ### Results from the uniform dataset:
# The learned model is able to accurately predict the positions of the keys in the uniform dataset. The uniform dataset contains randomly distributed keys across the range 0 to 1,000,000, which creates a generally linear relationship between keys and positions.

# %% [markdown]
# ## This section will be regarding the sequential dataset

# %%
def plot_sequential():
    # Generate the sequential dataset
    data_loader_sequential = DatasetGenerator.generate_sequential(DATASET_SIZE)

    # Build our learned index that has been created
    linear_index_sequential = LinearIndexAdaptive()
    linear_index_sequential.build_from_sorted_array(data_loader_sequential)

    # Plot the dataset points with the learned index
    plt.figure(figsize=(10, 6))
    plt.scatter(data_loader_sequential, np.arange(len(data_loader_sequential)), s=1, label='Sequential Data', alpha=0.5)
    predicted_positions = linear_index_sequential.a * data_loader_sequential + linear_index_sequential.b
    plt.plot(data_loader_sequential, predicted_positions, color='red', label='Linear Index Adaptive Model', linewidth=2, linestyle='--')
    plt.title('Linear Index Adaptive Model on Sequential Dataset')
    plt.xlabel('Keys')
    plt.ylabel('Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_uniform.png"), dpi=150, bbox_inches="tight"); plt.close()
plot_sequential()


# %% [markdown]
# ### Results from the sequential dataset:
#
# The learned index fits the sequential data almost perfectly. The sequential dataset contains keys 0, 1, 2, 3, ..., 99,999 - perfectly evenly spaced values. This creates an ideal linear relationship (slope ≈ 1.0) between keys and positions, resulting in near-zero prediction error.

# %% [markdown]
# ## This section will be regarding the mixed dataset

# %%
def plot_mixed():
    # Generate the mixed dataset
    data_loader_mixed = DatasetGenerator.generate_mixed(DATASET_SIZE)

    # Build our learned index that has been created
    linear_index_mixed = LinearIndexAdaptive()
    linear_index_mixed.build_from_sorted_array(data_loader_mixed)

    # Plot the dataset points with the learned index
    plt.figure(figsize=(10, 6))
    plt.scatter(data_loader_mixed, np.arange(len(data_loader_mixed)), s=1, label='Mixed Data', alpha=0.5)
    predicted_positions = linear_index_mixed.a * data_loader_mixed + linear_index_mixed.b
    plt.plot(data_loader_mixed, predicted_positions, color='red', label='Linear Index Adaptive Model', linewidth=2, linestyle='--')
    plt.title('Linear Index Adaptive Model on Mixed Dataset')
    plt.xlabel('Keys')
    plt.ylabel('Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_uniform.png"), dpi=150, bbox_inches="tight"); plt.close()
plot_mixed()


# %% [markdown]
# ### Results from the mixed dataset:
#
# The learned index does not fit the mixed data well. The mixed dataset has a non-linear, clustered distribution with two normal distributions centered at different points. A single linear model cannot accurately capture this complex pattern, resulting in larger prediction errors.

# %% [markdown]
# ## This section will go over the accuracy and error rate of the Linear Index Adaptive Model

# %%
def heatmap():
    # Data
    datasets = {
        'Sequential': DatasetGenerator.generate_sequential(DATASET_SIZE),
        'Uniform': DatasetGenerator.generate_uniform(DATASET_SIZE),
        'Mixed': DatasetGenerator.generate_mixed(DATASET_SIZE)
    }

    thresholds = [1, 10, 50, 100, 500, 1000]

    # Build the matrix
    matrix = np.zeros((len(datasets), len(thresholds)))

    for i, (name, data) in enumerate(datasets.items()):

        # Build the index
        linear_index = LinearIndexAdaptive()
        linear_index.build_from_sorted_array(data)

        # Call the predict function from our index
        predicted_positions = linear_index.a * data + linear_index.b

        # Now we can calc the errorr
        errors = np.abs(predicted_positions - np.arange(len(data)))
        for j, threshold in enumerate(thresholds):
            accuracy = np.sum(errors <= threshold) / len(data) * 100
            matrix[i, j] = accuracy

    # Generate the heatmap now that we have the data
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([f'<={t}' for t in thresholds])
    ax.set_yticklabels(datasets.keys())
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(thresholds)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Error Threshold\nMeaning what % of predictions are wiithin <threshold number> of the actual position', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Linear Index Adaptive Prediction Accuracy', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_uniform.png"), dpi=150, bbox_inches="tight"); plt.close()

heatmap()
