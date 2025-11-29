"""
Generate All Plots from Latest Benchmark Data

Run this anytime to regenerate all plots from your most recent benchmark.
Uses the latest CSV in results/benchmarks/

Usage:
    python generate_all_plots.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
RESULTS_DIR = "results/benchmarks"
GRAPHS_DIR = "graphs"
DISTRIBUTIONS = ["seq", "uniform", "mixed"]

def find_latest_csv():
    """Find the most recent benchmark CSV."""
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "run_*/master.csv"))
    if not csv_files:
        print("Error: No benchmark results found!")
        print(f"Run 'python main.py' first to generate benchmark data.")
        sys.exit(1)
    
    latest = max(csv_files, key=os.path.getmtime)
    return latest

def generate_all_plots(csv_file):
    """Generate all plots from CSV file."""
    
    print("="*70)
    print("GENERATING ALL PLOTS FROM BENCHMARK DATA")
    print("="*70)
    print(f"\nUsing: {csv_file}")
    print()
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Get all dataset sizes
    sizes = sorted(df['dataset_size'].unique())
    
    print(f"Found {len(sizes)} dataset sizes: {', '.join([f'{s:,}' for s in sizes])}")
    print()
    
    colors = {
        'btree': '#95A5A6',
        'kraska_single': '#E74C3C', 
        'kraska_rmi': '#3498DB',
        'linear_fixed': '#1ABC9C',
        'linear_adaptive': '#2ECC71'
    }
    
    # Generate plots for each dataset size
    for size in sizes:
        df_size = df[df['dataset_size'] == size]
        
        # Group and average across cycles
        summary = df_size.groupby(['model', 'params', 'distribution']).agg({
            'lookup_ns': 'mean',
            'build_ms': 'mean',
            'memory_mb': 'mean',
            'search_accuracy': 'mean',
            'accuracy': 'mean'
        }).reset_index()
        
        summary['lookup_us'] = summary['lookup_ns'] / 1000
        
        # Format size for folder name
        if size >= 1_000_000:
            size_str = f"{size // 1_000_000}M"
        elif size >= 1_000:
            size_str = f"{size // 1_000}K"
        else:
            size_str = str(size)
        
        # Create size-specific folder
        size_graphs_dir = os.path.join(GRAPHS_DIR, size_str)
        os.makedirs(size_graphs_dir, exist_ok=True)
        
        print(f"Generating plots for {size:,} keys → {size_graphs_dir}/")
        
        # === 1. LOOKUP TIME PER DISTRIBUTION ===
        for dist in DISTRIBUTIONS:
            dist_data = summary[summary['distribution'] == dist]
            
            if len(dist_data) == 0:
                continue
            
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(dist_data))
            
            bars = plt.bar(x_pos, dist_data['lookup_us'], 
                          color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
            
            plt.xlabel('Index Configuration', fontsize=12)
            plt.ylabel('Lookup Time (µs)', fontsize=12)
            plt.title(f'Lookup Performance - {dist.capitalize()} Distribution ({size_str} keys)', 
                     fontsize=14, fontweight='bold')
            plt.xticks(x_pos, [f"{row['model']}\n{row['params']}" 
                               for _, row in dist_data.iterrows()], rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(size_graphs_dir, f'lookup_time_{dist}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ lookup_time_{dist}.png")
        
        # === 2. PREDICTION ACCURACY PER DISTRIBUTION ===
        for dist in DISTRIBUTIONS:
            dist_data = summary[summary['distribution'] == dist]
            
            if len(dist_data) == 0:
                continue
            
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(dist_data))
            
            accuracy_pct = dist_data['accuracy'] * 100
            
            bars = plt.bar(x_pos, accuracy_pct,
                          color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
            
            plt.xlabel('Index Configuration', fontsize=12)
            plt.ylabel('Prediction Accuracy (%)', fontsize=12)
            plt.title(f'Prediction Accuracy - {dist.capitalize()} Distribution ({size_str} keys)',
                     fontsize=14, fontweight='bold')
            plt.xticks(x_pos, [f"{row['model']}\n{row['params']}" 
                               for _, row in dist_data.iterrows()], rotation=45, ha='right')
            plt.ylim(0, 105)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(size_graphs_dir, f'accuracy_{dist}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ accuracy_{dist}.png")
        
        # === 3. BUILD TIME COMPARISON ===
        plt.figure(figsize=(14, 6))
        
        # Get unique models
        models = summary['model'].unique()
        x_labels = []
        
        for i, dist in enumerate(DISTRIBUTIONS):
            dist_data = summary[summary['distribution'] == dist]
            if len(dist_data) == 0:
                continue
            
            x_pos = np.arange(len(dist_data)) + i * 0.25
            plt.bar(x_pos, dist_data['build_ms'],
                   width=0.25, label=dist.capitalize(),
                   alpha=0.8)
            
            if i == 0:
                x_labels = [f"{row['model']}" for _, row in dist_data.iterrows()]
        
        plt.xlabel('Index Configuration', fontsize=12)
        plt.ylabel('Build Time (ms)', fontsize=12)
        plt.title(f'Index Build Time Comparison ({size_str} keys)',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        if x_labels:
            plt.xticks(np.arange(len(x_labels)) + 0.25, x_labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(size_graphs_dir, f'build_time.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ build_time.png")
        
        # === 4. COMBINED COMPARISON (ALL DISTRIBUTIONS) ===
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, dist in enumerate(DISTRIBUTIONS):
            dist_data = summary[summary['distribution'] == dist]
            if len(dist_data) == 0:
                continue
            
            ax = axes[idx]
            x_pos = np.arange(len(dist_data))
            
            ax.bar(x_pos, dist_data['lookup_us'],
                  color=[colors.get(m, '#7F8C8D') for m in dist_data['model']])
            
            ax.set_xlabel('Index', fontsize=10)
            ax.set_ylabel('Lookup Time (µs)', fontsize=10)
            ax.set_title(f'{dist.capitalize()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([row['model'] for _, row in dist_data.iterrows()], 
                               rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Lookup Performance Comparison ({size_str} keys)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = os.path.join(size_graphs_dir, f'comparison_all.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ comparison_all.png")
        
        print()
    
    print("="*70)
    print("ALL PLOTS GENERATED!")
    print("="*70)
    print(f"\nGraphs saved to: {GRAPHS_DIR}/")
    print("\nFolder structure:")
    for folder in sorted(os.listdir(GRAPHS_DIR)):
        folder_path = os.path.join(GRAPHS_DIR, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            print(f"\n  {folder}/ ({len(files)} graphs)")
            for f in sorted(files):
                print(f"    - {f}")

if __name__ == '__main__':
    latest_csv = find_latest_csv()
    generate_all_plots(latest_csv)
