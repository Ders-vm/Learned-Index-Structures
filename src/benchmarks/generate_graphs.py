"""
Graph Generator for Learned Index Benchmarks
Generates publication-ready graphs from benchmark results
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
plt.switch_backend("Agg")  
plt.style.use("seaborn-v0_8-darkgrid")

# Add project root to Python path
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "benchmarks")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================================
# MODEL DISPLAY CONFIGURATION
# ============================================================================

def get_display_name(row):
    """
    Convert raw model names to clean display names.
    Returns None for configurations to filter out.
    """
    model = row['model']
    params = row.get('params', '')
    
    # Kraska Single-Stage: Keep only linear
    if model == 'kraska_single':
        if 'linear' in params:
            return "Kraska Single"
        return None
    
    # Kraska RMI: Keep only [1,100]
    elif model == 'kraska_rmi':
        if '[1,100]' in params or '[1, 100]' in params:
            return "Kraska RMI"
        return None
    
    # Linear Adaptive: Keep only best config
    elif model == 'linear_adaptive':
        if 'quantile=0.99' in params and 'min_window=16' in params:
            return "Linear Adaptive (Ours)"
        return None
    
    # Linear Fixed: Keep only window=512
    elif model == 'linear_fixed':
        if 'window=512' in params:
            return "Linear Fixed (Ours)"
        return None
    
    # B-Tree: Keep only order=128
    elif model == 'btree':
        if 'order=128' in params:
            return "B-Tree"
        return None
    
    # Unknown model: filter out
    return None


def get_model_color(model_name):
    """Define color scheme for graphs."""
    colors = {
        "Linear Adaptive (Ours)": "#2ECC71",  # Green
        "Linear Fixed (Ours)": "#1ABC9C",      # Teal
        "Kraska Single": "#E74C3C",            # Red
        "Kraska RMI": "#3498DB",               # Blue
        "B-Tree": "#95A5A6",                   # Gray
    }
    return colors.get(model_name, "#34495E")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_latest_run():
    """Find and load the most recent benchmark run."""
    print("Loading benchmark results...")
    
    # Find latest run directory
    if not os.path.exists(RESULTS_ROOT):
        raise FileNotFoundError(f"Results directory not found: {RESULTS_ROOT}")
    
    runs = [d for d in os.listdir(RESULTS_ROOT) if d.startswith("run_")]
    if not runs:
        raise FileNotFoundError("No benchmark runs found")
    
    latest = max(runs)
    master_path = os.path.join(RESULTS_ROOT, latest, "master.csv")
    
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"master.csv not found in {latest}")
    
    print(f"  Loading: {master_path}")
    
    # Load and filter data
    df = pd.read_csv(master_path)
    df['display_name'] = df.apply(get_display_name, axis=1)
    df = df.dropna(subset=['display_name'])
    
    print(f"  Loaded {len(df)} data points")
    print(f"  Models: {', '.join(sorted(df['display_name'].unique()))}")
    
    return df, latest


# ============================================================================
# GRAPH 1: LOOKUP TIME
# ============================================================================

def graph_lookup_time(df):
    """Generate lookup time graphs (one per distribution)."""
    print("\nGenerating lookup time graphs...")
    
    # Average across cycles
    agg = df.groupby(['dataset_size', 'distribution', 'display_name'])['lookup_ns'].mean().reset_index()
    agg['lookup_us'] = agg['lookup_ns'] / 1000
    
    # Generate one graph per distribution
    for dist in sorted(agg['distribution'].unique()):
        sub = agg[agg['distribution'] == dist]
        
        plt.figure(figsize=(10, 6))
        
        # Plot each model
        for model in sorted(sub['display_name'].unique()):
            data = sub[sub['display_name'] == model]
            color = get_model_color(model)
            lw = 3 if "Ours" in model else 2
            
            plt.plot(
                data['dataset_size'], 
                data['lookup_us'], 
                marker='o',
                linewidth=lw,
                markersize=8,
                label=model,
                color=color,
                alpha=0.9
            )
        
        plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
        plt.ylabel('Lookup Time (µs)', fontsize=12, fontweight='bold')
        plt.title(f'Lookup Performance - {dist.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"lookup_time_{dist}.png"
        plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: {filename}")


# ============================================================================
# GRAPH 2: MEMORY USAGE
# ============================================================================

def graph_memory(df):
    """Generate memory usage graph."""
    print("\nGenerating memory usage graph...")
    
    # Average across cycles and distributions
    agg = df.groupby(['dataset_size', 'display_name'])['memory_mb'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot each model
    for model in sorted(agg['display_name'].unique()):
        data = agg[agg['display_name'] == model]
        color = get_model_color(model)
        lw = 3 if "Ours" in model else 2
        
        plt.plot(
            data['dataset_size'], 
            data['memory_mb'], 
            marker='s',
            linewidth=lw,
            markersize=8,
            label=model,
            color=color,
            alpha=0.9
        )
    
    plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
    plt.ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    plt.title('Memory Efficiency', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = "memory_usage.png"
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


# ============================================================================
# GRAPH 3: COMPARISON BAR CHART
# ============================================================================

def graph_comparison(df):
    """Generate overall comparison bar chart at largest dataset."""
    print("\nGenerating comparison chart...")
    
    # Get largest dataset only
    biggest = df['dataset_size'].max()
    sub = df[df['dataset_size'] == biggest]
    
    # Average across distributions and cycles
    agg = sub.groupby('display_name')['lookup_ns'].mean().reset_index()
    agg['lookup_us'] = agg['lookup_ns'] / 1000
    
    # Sort by performance
    agg = agg.sort_values('lookup_us')
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar chart
    colors = [get_model_color(name) for name in agg['display_name']]
    bars = plt.barh(agg['display_name'], agg['lookup_us'], 
                    color=colors, alpha=0.85)
    
    # Highlight your models
    for i, name in enumerate(agg['display_name']):
        if "Ours" in name:
            bars[i].set_edgecolor('#000000')
            bars[i].set_linewidth(2.5)
    
    plt.xlabel('Lookup Time (µs)', fontsize=12, fontweight='bold')
    plt.title(f'Performance at {biggest:,} Keys', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(agg.iterrows()):
        plt.text(
            row['lookup_us'] + 0.2,
            i,
            f"{row['lookup_us']:.1f}", 
            va='center', 
            fontsize=10, 
            fontweight='bold'
        )
    
    plt.tight_layout()
    filename = "comparison.png"
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


# ============================================================================
# GRAPH 4: ACCURACY (OPTIONAL)
# ============================================================================

def graph_accuracy(df):
    """Generate accuracy graph for learned indexes."""
    print("\nGenerating accuracy graph...")
    
    # Check if accuracy data exists
    if 'accuracy' not in df.columns or df['accuracy'].isna().all():
        print("  Skipping: No accuracy data available")
        return
    
    # Filter to only models that have meaningful accuracy metrics
    # (exclude exact structures like B-tree)
    learned_models = df[df['display_name'].str.contains('Linear|Kraska', na=False)]
    
    if len(learned_models) == 0:
        print("  Skipping: No learned index models found")
        return
    
    # Average across cycles
    agg = learned_models.groupby(['dataset_size', 'distribution', 'display_name'])['accuracy'].mean().reset_index()
    
    # Generate one graph per distribution
    for dist in sorted(agg['distribution'].unique()):
        sub = agg[agg['distribution'] == dist]
        
        if sub['accuracy'].isna().all():
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Plot each model
        for model in sorted(sub['display_name'].unique()):
            data = sub[sub['display_name'] == model]
            
            # Skip if all NaN
            if data['accuracy'].isna().all():
                continue
            
            color = get_model_color(model)
            lw = 3 if "Ours" in model else 2
            
            plt.plot(
                data['dataset_size'], 
                data['accuracy'], 
                marker='o',
                linewidth=lw,
                markersize=8,
                label=model,
                color=color,
                alpha=0.9
            )
        
        plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
        plt.ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'Prediction Accuracy - {dist.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.ylim([0, 1.05])
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"accuracy_{dist}.png"
        plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution: Load data and generate all graphs."""
    print("="*60)
    print("GRAPH GENERATOR")
    print("="*60)
    
    # Load data
    df, _ = load_latest_run()
    
    # Generate graphs
    graph_lookup_time(df)   # 3 files
    graph_memory(df)        # 1 file
    graph_comparison(df)    # 1 file
    graph_accuracy(df)      # 0-3 files (if accuracy data exists)
    
    print("\n" + "="*60)
    print("COMPLETE - Graphs saved to: graphs/")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
