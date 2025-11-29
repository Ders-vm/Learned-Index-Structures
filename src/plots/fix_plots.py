"""
Fix plot scripts to:
1. Work correctly (fix __line__ error)
2. Pull data from latest benchmark run
3. Minimal changes to structure
"""

import re

def fix_adaptive_plot():
    """Fix linear_index_adaptive_plot.py"""
    with open('linear_index_adaptive_plot.py', 'r') as f:
        content = f.read()
    
    # Fix __line__ errors - replace with actual function names
    content = content.replace(
        'plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_plot_{}.png".format(__line__)),',
        'plt.savefig(os.path.join(OUTPUT_DIR, "adaptive_uniform.png"),'
    )
    
    # Actually, let's be smarter - find which function and name appropriately
    lines = content.split('\n')
    new_lines = []
    current_function = 'unknown'
    
    for line in lines:
        if 'def plot_' in line:
            match = re.search(r'def (plot_\w+)', line)
            if match:
                current_function = match.group(1).replace('plot_', '')
        
        if '__line__' in line:
            # Replace with proper filename based on function
            line = line.replace(
                '"adaptive_plot_{}.png".format(__line__)',
                f'"adaptive_{current_function}.png"'
            )
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open('linear_index_adaptive_plot.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed linear_index_adaptive_plot.py")

def fix_generate_exploratory():
    """Fix generate_exploratory_plots.py path issue"""
    with open('generate_exploratory_plots.py', 'r') as f:
        content = f.read()
    
    # Check if it has the path fix
    if 'sys.path.insert' not in content:
        # Add path setup at the beginning
        lines = content.split('\n')
        
        # Find where imports start
        import_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_idx = i
                break
        
        # Insert path setup before imports
        path_setup = """
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
"""
        lines.insert(import_idx, path_setup)
        content = '\n'.join(lines)
        
        with open('generate_exploratory_plots.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed generate_exploratory_plots.py")
    else:
        print("✅ generate_exploratory_plots.py already has path fix")

if __name__ == '__main__':
    print("Fixing plot scripts...")
    print()
    fix_adaptive_plot()
    fix_generate_exploratory()
    print()
    print("Done! Now run: python src/plots/linear_index_adaptive_plot.py")
