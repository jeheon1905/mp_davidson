import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import argparse
import os
import re
from typing import List, Dict, Any

# ==============================
# Configuration
# ==============================
dtype_list = ["SP", "TF32", "BF16", "HP"]
operations = ["projection", "kinetic", "local", "nonlocal", "tensordot"]

# Color scheme for different dtypes
colors = {
    "SP": "#1f77b4",    # blue
    "TF32": "#ff7f0e",  # orange
    "BF16": "#2ca02c",  # green
    "HP": "#d62728",    # red
}

# Marker styles
markers = {
    "SP": "o",
    "TF32": "s",
    "BF16": "^",
    "HP": "D",
}

# natoms info
natoms_dict = {
    "CNT_6_0": 24,
    "MgO_1x1x2": 16,
    "Si_diamond_2x2x1": 32,
}

# Default font sizes (used and merged with CLI args)
FONT_SIZES_DEFAULT = {
    'title': 16,
    'subtitle': 14,
    'label': 12,
    'tick': 10
}

# ==============================
# Utilities
# ==============================
def _merge_font_sizes(font_sizes: Dict[str, int] | None) -> Dict[str, int]:
    """Merge user-provided font sizes with defaults and return a safe dict."""
    merged = FONT_SIZES_DEFAULT.copy()
    if font_sizes:
        merged.update({k: int(v) for k, v in font_sizes.items() if k in merged and v is not None})
    return merged

def _apply_axis_fonts(ax: plt.Axes, font_sizes: Dict[str, int]):
    """Apply tick and label font sizes to an axis."""
    ax.tick_params(axis='both', labelsize=font_sizes['tick'])
    # Note: label font sizes are applied at set_xlabel/set_ylabel calls

def calculate_natoms(supercell_size, base_natoms=5):
    """
    Calculate number of atoms based on supercell size
    Default base_natoms=5 for BaTiO3 unit cell
    """
    return base_natoms * supercell_size

# ==============================
# Parsing
# ==============================
def parse_timer_summaries(log_content: str) -> pd.DataFrame:
    """
    Parse 'Timer Summary' blocks from the log and return a DataFrame.
    This version uses a robust splitter on the header line to avoid brittle regex.
    """
    # Split the log by lines that look like "====... Timer Summary ...===="
    sections = re.split(r"^=+\s*Timer Summary\s*=+\s*$", log_content, flags=re.MULTILINE)
    # sections[0] is preamble before first header; blocks are sections[1:]
    results = []

    for block in sections[1:]:
        # Example line we expect inside a block (adjust to your real format as needed):
        # Operation: projection (DP) | 0.123 | 45
        operation_pattern = r"Operation:\s*(\w+)\s*\((\w+)\)\s*\|\s*([\d.]+)\s*\|\s*(\d+)"
        operations_found = re.findall(operation_pattern, block)

        # Metadata (take the last appearances if multiple)
        supercell_match = re.search(r"supercell:\s*\[([\d,\s]+)\]", block)
        dtype_match = re.search(r"dtype:\s*(\w+)", block)
        dense_proj_match = re.search(r"use_dense_proj:\s*(\w+)", block)
        dense_kinetic_match = re.search(r"use_dense_kinetic:\s*(\w+)", block)

        supercell_size = None
        if supercell_match:
            supercell_numbers = [int(x.strip()) for x in supercell_match.group(1).split(",")]
            supercell_size = supercell_numbers[-1]  # use the last value

        for op_name, op_dtype, time, count in operations_found:
            record = {
                "operation": op_name,
                "operation_dtype": op_dtype,
                "time_msec": float(time) * 1000.0,   # sec -> msec
                "count": int(count),
                "supercell_size": supercell_size,
                "dtype": dtype_match.group(1) if dtype_match else None,
                "use_dense_proj": (
                    dense_proj_match.group(1).lower() == "true" if dense_proj_match else None
                ),
                "use_dense_kinetic": (
                    dense_kinetic_match.group(1).lower() == "true" if dense_kinetic_match else None
                ),
            }
            results.append(record)

    return pd.DataFrame(results)

# ==============================
# Data prep for plotting
# ==============================
def prepare_plot_data(df: pd.DataFrame, use_dense_proj=False, material_name=None):
    """
    Prepare data per operation for plotting speedup curves and DP baseline times.
    """
    df_filtered = df[df["use_dense_proj"] == use_dense_proj].copy()

    df_dp = df_filtered[df_filtered["dtype"] == "DP"].copy()
    df_others = df_filtered[df_filtered["dtype"] != "DP"].copy()

    dp_pivot = df_dp.pivot_table(
        values="time_msec", index="operation", columns="supercell_size", aggfunc="sum"
    )

    other_pivot = df_others.pivot_table(
        values="time_msec", index=["operation", "dtype"], columns="supercell_size", aggfunc="sum"
    )

    supercell_sizes = sorted(dp_pivot.columns.tolist())

    # Determine base_natoms
    if material_name and material_name in natoms_dict:
        base_natoms = natoms_dict[material_name]
    else:
        base_natoms = 5  # default for BaTiO3

    natoms = [calculate_natoms(size, base_natoms) for size in supercell_sizes]

    plot_data = {}
    for operation in operations:
        if operation not in dp_pivot.index:
            continue

        plot_data[operation] = {
            'natoms': natoms,
            'speedup': {},
            'dp_time': dp_pivot.loc[operation].values.tolist()
        }

        # Speedup = DP time / other dtype time
        for dtype in dtype_list:
            if (operation, dtype) in other_pivot.index:
                other_times = other_pivot.loc[(operation, dtype)]
                dp_times = dp_pivot.loc[operation]

                # align indices just in case
                aligned = dp_times.align(other_times, join='inner')
                if aligned[0].empty:
                    speedup = [np.nan] * len(natoms)
                else:
                    sp = (aligned[0] / aligned[1]).values.tolist()
                    # expand/trim to length of natoms
                    speedup = sp + [np.nan] * (len(natoms) - len(sp))
            else:
                speedup = [np.nan] * len(natoms)

            plot_data[operation]['speedup'][dtype] = speedup

    return plot_data

# ==============================
# Plotting
# ==============================
def create_performance_plots(plot_data,
                             save_prefix="",
                             title_suffix="",
                             show_dp_time=True,
                             create_legend_figure=True,
                             xscale='log',
                             font_sizes=None):
    """
    Create performance plots for each operation. Font sizes are applied consistently.
    """
    font_sizes = _merge_font_sizes(font_sizes)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_idx = 0
    for operation in operations:
        if operation not in plot_data:
            continue

        ax = axes[plot_idx]
        op_data = plot_data[operation]

        natoms_array = np.array(op_data['natoms'], dtype=float)

        # Primary y-axis: Speedup curves
        for dtype in dtype_list:
            speedup_array = np.array(op_data['speedup'][dtype], dtype=float)

            # Trim to same length
            min_len = min(len(natoms_array), len(speedup_array))
            x = natoms_array[:min_len]
            y = speedup_array[:min_len]

            valid = ~np.isnan(y)
            if np.any(valid):
                ax.plot(
                    x[valid], y[valid],
                    marker=markers[dtype],
                    color=colors[dtype],
                    label=dtype,
                    linewidth=2,
                    markersize=8
                )

        # Formatting (use provided font sizes)
        ax.set_xlabel('Number of atoms', fontsize=font_sizes['label'])
        ax.set_ylabel('Speedup', fontsize=font_sizes['label'])
        ax.set_title(f'({chr(97+plot_idx)}) {operation.capitalize()}',
                     fontsize=font_sizes['subtitle'], fontweight='bold')
        ax.grid(True, alpha=0.3)

        # X scale
        if xscale not in ('linear', 'log'):
            xscale = 'log'
        ax.set_xscale(xscale)

        # Apply tick font sizes
        _apply_axis_fonts(ax, font_sizes)

        # Secondary y-axis: DP time (optional)
        if show_dp_time and 'dp_time' in op_data:
            dp_time_array = np.array(op_data['dp_time'], dtype=float)
            min_len = min(len(natoms_array), len(dp_time_array))
            x2 = natoms_array[:min_len]
            y2 = dp_time_array[:min_len]
            valid2 = ~np.isnan(y2)

            if np.any(valid2):
                ax2 = ax.twinx()
                ax2.plot(x2[valid2], y2[valid2], 'k--', linewidth=1.5, alpha=0.7, label='DP time')
                ax2.set_ylabel('DP time (ms)', fontsize=font_sizes['label'], color='black')
                ax2.tick_params(axis='y', labelcolor='black', labelsize=font_sizes['tick'])

        plot_idx += 1

    # Remove unused subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    # Suptitle
    main_title = 'Performance Comparison'
    if title_suffix:
        main_title = f'{main_title} {title_suffix}'
    fig.suptitle(main_title, fontsize=font_sizes['title'], fontweight='bold')

    # Avoid overlap with suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    if save_prefix:
        fig.savefig(f'{save_prefix}_performance_plots.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_prefix}_performance_plots.pdf', bbox_inches='tight')
        print(f"Saved: {save_prefix}_performance_plots.png/pdf")

    # Separate legend figure (optional)
    if create_legend_figure:
        create_legend_only(save_prefix, font_sizes)

    return fig

def create_legend_only(save_prefix="", font_sizes=None):
    """
    Create a separate figure containing only the legend, respecting font sizes.
    """
    font_sizes = _merge_font_sizes(font_sizes)

    fig_legend = plt.figure(figsize=(6, 2))

    legend_elements = []
    for dtype in dtype_list:
        legend_elements.append(
            plt.Line2D([0], [0],
                       marker=markers[dtype],
                       color='w',
                       markerfacecolor=colors[dtype],
                       markersize=10,
                       label=dtype,
                       linewidth=2,
                       markeredgewidth=1.5,
                       markeredgecolor=colors[dtype])
        )

    # DP time line
    legend_elements.append(
        plt.Line2D([0], [0],
                   color='black',
                   linewidth=1.5,
                   linestyle='--',
                   alpha=0.7,
                   label='DP time (right axis)')
    )

    legend = fig_legend.legend(
        handles=legend_elements,
        loc='center',
        ncol=len(dtype_list) + 1,
        frameon=True,
        fontsize=font_sizes['tick'],
        title='Precision Types',
        title_fontsize=font_sizes['subtitle']
    )

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.2)

    if save_prefix:
        fig_legend.savefig(f'{save_prefix}_legend.png', dpi=300, bbox_inches='tight')
        fig_legend.savefig(f'{save_prefix}_legend.pdf', bbox_inches='tight')
        print(f"Saved: {save_prefix}_legend.png/pdf")

    return fig_legend

def plot_operation_comparison(plot_data, operation, save_name="",
                              xscale='log', font_sizes=None):
    """
    Create a single plot for a specific operation with consistent font sizes.
    """
    font_sizes = _merge_font_sizes(font_sizes)

    if operation not in plot_data:
        print(f"Operation '{operation}' not found in data")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    op_data = plot_data[operation]

    natoms_array = np.array(op_data['natoms'], dtype=float)
    has_data = False

    for dtype in dtype_list:
        speedup_array = np.array(op_data['speedup'][dtype], dtype=float)
        min_len = min(len(natoms_array), len(speedup_array))
        x = natoms_array[:min_len]
        y = speedup_array[:min_len]
        valid = ~np.isnan(y)
        if np.any(valid):
            has_data = True
            ax.plot(
                x[valid], y[valid],
                marker=markers[dtype],
                color=colors[dtype],
                label=dtype,
                linewidth=2,
                markersize=8
            )

    ax.set_xlabel('Number of atoms', fontsize=font_sizes['label'])
    ax.set_ylabel('Speedup', fontsize=font_sizes['label'])
    ax.set_title(f'{operation.capitalize()} Performance', fontsize=font_sizes['title'], fontweight='bold')
    ax.grid(True, alpha=0.3)

    if xscale not in ('linear', 'log'):
        xscale = 'log'
    ax.set_xscale(xscale)

    _apply_axis_fonts(ax, font_sizes)

    if has_data:
        ax.legend(loc='best', fontsize=font_sizes['tick'])
    else:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=font_sizes['label'], color='gray')

    # Secondary axis: DP time
    if 'dp_time' in op_data:
        dp_time_array = np.array(op_data['dp_time'], dtype=float)
        min_len = min(len(natoms_array), len(dp_time_array))
        x2 = natoms_array[:min_len]
        y2 = dp_time_array[:min_len]
        valid2 = ~np.isnan(y2)
        if np.any(valid2):
            ax2 = ax.twinx()
            ax2.plot(x2[valid2], y2[valid2], 'k--', linewidth=1.5, alpha=0.7)
            ax2.set_ylabel('DP time (ms)', fontsize=font_sizes['label'], color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=font_sizes['tick'])

    plt.tight_layout()

    if save_name:
        fig.savefig(f'{save_name}_{operation}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_name}_{operation}.pdf', bbox_inches='tight')
        print(f"Saved: {save_name}_{operation}.png/pdf")

    return fig

# ==============================
# CLI
# ==============================
def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate performance comparison plots from log file'
    )

    parser.add_argument('log_file', type=str, help='Path to the log file containing Timer Summary data')

    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output files (default: basename of log file)')

    parser.add_argument('--material', type=str, default=None, choices=list(natoms_dict.keys()) + [None],
                        help='Material name for natoms calculation (default: BaTiO3 with 5 atoms/unit cell)')

    parser.add_argument('--base-natoms', type=int, default=5, help='Base number of atoms per unit cell (default: 5 for BaTiO3)')

    parser.add_argument('--use-dense-proj', action='store_true', help='Use dense projection data (default: sparse)')

    parser.add_argument('--no-dp-time', action='store_true', help='Do not show DP time on secondary y-axis')

    parser.add_argument('--no-legend', action='store_true', help='Do not create separate legend figure')

    parser.add_argument('--individual-plots', action='store_true', help='Create individual plots for each operation')

    parser.add_argument('--title-suffix', type=str, default='', help='Additional text to add to the main title')

    parser.add_argument('--xscale', type=str, default='log', choices=['linear', 'log'], help='X-axis scale type (default: log)')

    parser.add_argument('--font-size-title', type=int, default=16, help='Font size for main/supertitle (default: 16)')
    parser.add_argument('--font-size-subtitle', type=int, default=14, help='Font size for subplot titles (default: 14)')
    parser.add_argument('--font-size-label', type=int, default=12, help='Font size for axis labels (default: 12)')
    parser.add_argument('--font-size-tick', type=int, default=10, help='Font size for tick labels (default: 10)')

    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    return parser.parse_args()

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found!")
        exit(1)

    if not args.output_prefix:
        args.output_prefix = os.path.splitext(os.path.basename(args.log_file))[0]

    # Prepare font sizes dictionary
    font_sizes = _merge_font_sizes({
        'title': args.font_size_title,
        'subtitle': args.font_size_subtitle,
        'label': args.font_size_label,
        'tick': args.font_size_tick
    })

    try:
        with open(args.log_file, 'r') as f:
            log_content = f.read()

        print(f"Parsing log file: {args.log_file}")
        df = parse_timer_summaries(log_content)

        if df.empty:
            print("No timer summary data found in the log file!")
            exit(1)

        print(f"Found {len(df)} records")
        print(f"Operations: {sorted(df['operation'].dropna().unique().tolist())}")
        print(f"Dtypes: {sorted(df['dtype'].dropna().unique().tolist())}")
        print(f"Supercell sizes: {sorted([x for x in df['supercell_size'].dropna().unique().tolist()])}")

        print(f"\nPreparing plot data (use_dense_proj={args.use_dense_proj})...")
        plot_data = prepare_plot_data(
            df,
            use_dense_proj=args.use_dense_proj,
            material_name=args.material
        )

        if not plot_data:
            print("No data to plot!")
            exit(1)

        if not args.title_suffix:
            args.title_suffix = "(Dense Projection)" if args.use_dense_proj else "(Sparse Projection)"

        print(f"Creating performance plots (xscale={args.xscale})...")
        fig = create_performance_plots(
            plot_data,
            save_prefix=args.output_prefix,
            title_suffix=args.title_suffix,
            show_dp_time=not args.no_dp_time,
            create_legend_figure=not args.no_legend,
            xscale=args.xscale,
            font_sizes=font_sizes
        )

        if args.individual_plots:
            print("\nCreating individual operation plots...")
            for operation in operations:
                if operation in plot_data:
                    plot_operation_comparison(
                        plot_data,
                        operation,
                        save_name=f"{args.output_prefix}_individual",
                        xscale=args.xscale,
                        font_sizes=font_sizes
                    )

        if args.show:
            plt.show()

        print(f"\nAll plots have been generated successfully!")
        print(f"Output files saved with prefix: {args.output_prefix}")

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
