#!/usr/bin/env python3
"""
Parse GOSPEL log files and extract key metrics
"""

import re
import json
import pandas as pd
from pathlib import Path
import argparse
import numpy as np


def parse_timer_summary(content):
    """
    Parse Timer Summary section from log content.
    Returns a dictionary of timer labels and their total times.
    
    Example section:
    ======================== Timer Summary ========================
    Label                                    | Total (s) | Calls
    ----------------------------------------------------------------
    GOSPEL.calculate                         |    32.836 |     1
    Davidson.diagonalize                     |    25.123 |    17
    ...
    """
    timer_data = {}
    in_summary = False
    
    for line in content.split('\n'):
        if '======================== Timer Summary ========================' in line:
            in_summary = True
            continue
        elif in_summary:
            # Skip separator lines
            if re.match(r'-{20,}', line):
                continue
            # End of timer summary
            elif 'Elapsed time' in line or line.strip() == '':
                break
            else:
                # Parse timer line: Label | Total (s) | Calls
                match = re.match(r'^(.*?)\s+\|\s+([\d\.]+)\s+\|\s+(\d+)', line)
                if match:
                    label = match.group(1).strip()
                    total_time = float(match.group(2))
                    timer_data[label] = total_time
    
    return timer_data


def get_category_map():
    """
    Define timer categories for SCF calculations.
    Returns a dictionary mapping category names to lists of timer labels.
    """
    category_map = {
        "Diagonalization": ["Davidson.diagonalize"],
        "Initialization": ["GOSPEL.initialize"],
        "Calc. potential and energy": [
            "Hamiltonian.update",
            "Hamiltonian.calc_and_print_energies",
        ],
    }
    return category_map


def aggregate_timing_by_category(breakdown, category_map):
    """
    Group raw timing data into defined categories.
    
    Args:
        breakdown (dict): Original timing data from log (label -> time).
        category_map (dict): Category name -> list of labels to include.
    
    Returns:
        dict: Aggregated timing by category.
    """
    aggregated = {key: 0.0 for key in category_map}
    
    for label, time in breakdown.items():
        for category, label_list in category_map.items():
            if label in label_list:
                aggregated[category] += time
                break
    
    # Add ETC time (everything not categorized)
    sum_of_categories = sum(aggregated.values())
    total_time = breakdown.get("GOSPEL.calculate", 0) + breakdown.get("GOSPEL.initialize", 0)
    etc_time = total_time - sum_of_categories
    aggregated["ETC"] = etc_time
    
    return aggregated


def parse_log_file(log_path):
    """
    Parse a single GOSPEL log file and extract:
    - Total Energy (Ha)
    - SCF Converged iteration count
    - Elapsed time (s)
    """
    result = {
        'file': log_path.name,
        'system': None,
        'etol': None,
        'dtol': None,
        'seed': None,
        'method': None,
        'nelec': None,
        'total_energy': None,
        'energy_per_electron': None,
        'scf_iters': None,
        'elapsed_time': None,
        'timer_breakdown': {}
    }
    
    # Parse filename: 1_1_{n}_{method}.log
    filename_pattern = r'1_1_(\d+)_(.+)\.log'
    match = re.match(filename_pattern, log_path.name)
    if match:
        result['method'] = match.group(2)
    
    # Parse directory path: expt.{system}.etol{etol}.seed{seed}
    parent_dir = log_path.parent.name
    # dir_pattern = r'expt\.(.+?)\.etol([^.]+)\.seed(\d+)'
    dir_pattern = r'expt\.(.+?)\.etol([^.]+)\.dtol([^.]+)\.seed(\d+)'
    dir_match = re.match(dir_pattern, parent_dir)
    if dir_match:
        result['system'] = dir_match.group(1)
        result['etol'] = dir_match.group(2)
        result['dtol'] = dir_match.group(3)
        result['seed'] = dir_match.group(4)
    
    # Read and parse log file
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract nelec
        nelec_match = re.search(r'\*\s+nelec\s+:\s+([\d.]+)', content)
        if nelec_match:
            result['nelec'] = float(nelec_match.group(1))
        
        # Extract Total Energy
        energy_match = re.search(r'Total Energy:\s+([-\d.]+)\s+Ha', content)
        if energy_match:
            result['total_energy'] = float(energy_match.group(1))
            
            # Calculate energy per electron
            if result['nelec'] is not None and result['nelec'] > 0:
                result['energy_per_electron'] = result['total_energy'] / result['nelec']
        
        # Extract SCF iterations
        scf_match = re.search(r'SCF CONVERGED with (\d+) iters', content)
        if scf_match:
            result['scf_iters'] = int(scf_match.group(1))
        
        # Extract elapsed time
        time_match = re.search(r'\[Time: GOSPEL\.calculate\]:\s+([\d.]+)\s+s', content)
        if time_match:
            result['elapsed_time'] = float(time_match.group(1))
        
        # Extract timer breakdown
        result['timer_breakdown'] = parse_timer_summary(content)
    
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return result


def collect_log_files(base_dir, pattern='**/1_1_*_*.log'):
    """
    Collect all log files matching the pattern
    """
    base_path = Path(base_dir)
    log_files = list(base_path.glob(pattern))
    return sorted(log_files)


def calculate_energy_errors(df, ref_method='DP_ref'):
    """
    Calculate |ΔE| per electron for each method compared to reference method
    Groups by (system, etol, dtol, seed) and calculates error within each group
    Uses energy_per_electron instead of total_energy
    """
    if ref_method not in df['method'].values:
        print(f"\nWarning: Reference method '{ref_method}' not found in data")
        return df
    
    df = df.copy()
    df['energy_error_per_electron'] = np.nan
    
    # Group by system, etol, dtol, seed
    group_cols = ['system', 'etol', 'dtol', 'seed']
    for group_key, group_df in df.groupby(group_cols):
        # Find reference energy in this group
        ref_row = group_df[group_df['method'] == ref_method]
        
        if len(ref_row) == 0:
            continue
        
        ref_energy_per_electron = ref_row['energy_per_electron'].iloc[0]
        
        if pd.isna(ref_energy_per_electron):
            continue
        
        # Calculate |ΔE| per electron for all methods in this group
        for idx in group_df.index:
            if pd.notna(df.loc[idx, 'energy_per_electron']):
                df.loc[idx, 'energy_error_per_electron'] = abs(
                    df.loc[idx, 'energy_per_electron'] - ref_energy_per_electron
                )
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Parse GOSPEL log files and extract key metrics'
    )
    parser.add_argument(
        'base_dir',
        nargs='?',
        default='.',
        help='Base directory to search for log files (default: current directory)'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='**/1_1_*_*.log',
        help='Glob pattern for log files (default: **/1_1_*_*.log)'
    )
    parser.add_argument(
        '-r', '--ref-method',
        default='DP_ref',
        help='Reference method for energy error calculation (default: DP_ref)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file (optional)'
    )
    parser.add_argument(
        '--timer-output',
        help='Output CSV file for timer breakdown data (optional)'
    )
    parser.add_argument(
        '--timer-json',
        help='Output JSON file for aggregated timer breakdown by method (optional)'
    )
    parser.add_argument(
        '--show-timers',
        action='store_true',
        help='Show detailed timer breakdown in output'
    )
    
    args = parser.parse_args()
    
    # Collect log files
    log_files = collect_log_files(args.base_dir, args.pattern)
    
    if not log_files:
        print(f"No log files found in {args.base_dir} with pattern {args.pattern}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Parse all log files
    results = []
    for log_file in log_files:
        result = parse_log_file(log_file)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate energy errors relative to reference method
    df = calculate_energy_errors(df, ref_method=args.ref_method)
    
    # Reorder columns for better readability
    columns = ['system', 'etol', 'dtol', 'seed', 'method', 'nelec', 
               'total_energy', 'energy_per_electron', 'energy_error_per_electron', 
               'scf_iters', 'elapsed_time', 'file']
    df = df[columns]
    
    # Sort by system, method
    df = df.sort_values(['system', 'method'])
    
    # Display results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6e}' if abs(x) < 1e-3 or abs(x) > 1e3 else f'{x:.6f}')
    
    print("\n" + "="*100)
    print("GOSPEL Log File Analysis")
    print("="*100 + "\n")
    print(df.to_string(index=False))
    
    # Summary statistics
    if len(df) > 0:
        print("\n" + "="*100)
        print("Summary Statistics by Method")
        print("="*100 + "\n")
        
        # Group by method
        if 'method' in df.columns and df['method'].notna().any():
            summary = df.groupby('method').agg({
                'energy_error_per_electron': ['mean', 'median', 'std', 'min', 'max'],
                'scf_iters': ['mean', 'median', 'std', 'min', 'max'],
                'elapsed_time': ['mean', 'median', 'std', 'min', 'max']
            })
            
            # Apply different rounding for different columns
            # Energy errors: keep more precision (10 decimal places)
            for col in ['energy_error_per_electron']:
                for stat in ['mean', 'median', 'std', 'min', 'max']:
                    summary[(col, stat)] = summary[(col, stat)].round(10)
            
            # SCF iters and time: moderate precision (4 decimal places)
            for col in ['scf_iters', 'elapsed_time']:
                for stat in ['mean', 'median', 'std', 'min', 'max']:
                    summary[(col, stat)] = summary[(col, stat)].round(4)
            
            # Rename columns for clarity
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            
            print(summary)
            
            # Show energy error in scientific notation for better readability
            print("\n" + "-"*100)
            print("Energy Error per Electron (|ΔE|/nelec) Statistics (in Ha/electron)")
            print("-"*100 + "\n")
            
            energy_stats = df.groupby('method')['energy_error_per_electron'].agg(['mean', 'median', 'std', 'min', 'max'])
            # Filter out reference method (should have 0 or NaN error)
            energy_stats = energy_stats[energy_stats['mean'] > 1e-10]
            
            for col in energy_stats.columns:
                energy_stats[col] = energy_stats[col].apply(lambda x: f'{x:.6e}')
            
            print(energy_stats)
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    # Process and save timer breakdown if requested
    if args.timer_output or args.show_timers or args.timer_json:
        timer_rows = []
        for idx, row in df.iterrows():
            timer_breakdown = results[idx]['timer_breakdown']
            for timer_label, timer_value in timer_breakdown.items():
                timer_rows.append({
                    'system': row['system'],
                    'etol': row['etol'],
                    'dtol': row['dtol'],
                    'seed': row['seed'],
                    'method': row['method'],
                    'timer_label': timer_label,
                    'time': timer_value
                })
        
        if timer_rows:
            timer_df = pd.DataFrame(timer_rows)
            
            if args.show_timers:
                # print("\n" + "="*100)
                # print("Timer Breakdown Summary (Raw Timers)")
                # print("="*100 + "\n")
                # 
                # # Show average time for each timer across all runs, grouped by method
                # timer_summary = timer_df.groupby(['method', 'timer_label'])['time'].agg(['mean', 'std', 'min', 'max']).round(4)
                # print(timer_summary)
                
                # Show categorized breakdown
                print("\n" + "="*100)
                print("Timer Breakdown Summary (Categorized)")
                print("="*100 + "\n")
                
                category_map = get_category_map()
                categorized_data = []
                
                for method in sorted(df['method'].unique()):
                    method_timers = timer_df[timer_df['method'] == method]
                    # Average timer breakdown for this method
                    avg_breakdown = method_timers.groupby('timer_label')['time'].mean().to_dict()
                    aggregated = aggregate_timing_by_category(avg_breakdown, category_map)
                    aggregated['method'] = method
                    categorized_data.append(aggregated)
                
                categorized_df = pd.DataFrame(categorized_data)
                categorized_df = categorized_df.set_index('method')
                print(categorized_df.round(4))
            
            if args.timer_output:
                timer_df.to_csv(args.timer_output, index=False)
                print(f"\nTimer breakdown saved to {args.timer_output}")
            
            # Generate JSON output if requested (following plot_breakdown.py pattern)
            if args.timer_json:
                category_map = get_category_map()
                
                # Aggregate timer data by method
                aggregated_results = []
                
                for method in sorted(df['method'].unique()):
                    method_timers = timer_df[timer_df['method'] == method]
                    # Average timer breakdown for this method across all seeds
                    avg_breakdown = method_timers.groupby('timer_label')['time'].mean().to_dict()
                    aggregated = aggregate_timing_by_category(avg_breakdown, category_map)
                    
                    # Add total_time and acc_fold (similar to plot_breakdown.py)
                    total_time = sum(aggregated.values())
                    aggregated['total_time'] = total_time
                    
                    # Add method name for reference
                    result_dict = {'method': method}
                    result_dict.update(aggregated)
                    aggregated_results.append(result_dict)
                
                # Calculate acc_fold (speedup relative to first method)
                if aggregated_results:
                    base_time = aggregated_results[0]['total_time']
                    for result in aggregated_results:
                        result['acc_fold'] = base_time / result['total_time']
                
                # Save to JSON
                with open(args.timer_json, 'w') as f:
                    json.dump(aggregated_results, f, indent=2)
                print(f"\nAggregated timer data saved to {args.timer_json}")


if __name__ == '__main__':
    main()
