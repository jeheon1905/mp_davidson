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
dtype_list = ["SP", "TF32", "BF16", "HP"]  # full set kept for data prep
# Only plot these dtypes (HP disabled as requested)
PLOT_DTYPES = ["SP", "TF32", "BF16"]
# PLOT_DTYPES = ["SP", "TF32", "BF16", "HP"]  # <-- enable HP (FP16) later if needed

operations = ["projection", "kinetic", "local", "nonlocal", "tensordot"]

# Color scheme for different dtypes (used in speedup subplots)
colors = {
    "SP": "#1f77b4",    # blue (displayed as FP32)
    "TF32": "#ff7f0e",  # orange
    "BF16": "#2ca02c",  # green
    "HP": "#d62728",    # red (displayed as FP16)  # currently not plotted
}

# Marker styles per dtype (speedup subplots)
markers = {
    "SP": "o",
    "TF32": "s",
    "BF16": "^",
    "HP": "D",
}

# ===== Distinct styles for DP comparison panel (avoid confusion with dtype colors) =====
# Clear, vivid palette different from first 3 dtype colors
OP_MARKERS = {
    "projection": "o",
    "kinetic": "s",
    "local": "^",
    "nonlocal": "D",
    "tensordot": "v",
}

OP_COLORS = {
    "projection": "#8c1d18",  # 진한 다크 레드 (강조)
    "kinetic":    "#6f42c1",  # 퍼플
    "local":      "#006d77",  # 딥 틸 (green과 구분)
    "nonlocal":   "#5f6a6a",  # 슬레이트 그레이
    "tensordot":  "#6ec5e9",  # 라이트 시안(살짝 연함)
}

OP_LINESTYLES = {
    "projection": "-", 
    "kinetic": "--",
    "local": "-.",
    "nonlocal": (0, (5,2)),
    "tensordot": ":",
}
# Marker face/edge styles (projection filled, tensordot hollow)
OP_MARKERFACE = {
    "projection": None,      # -> 선색으로 채움(진하게 보임)
    "kinetic":    None,
    "local":      None,
    "nonlocal":   None,
    "tensordot":  "none",   # -> 속 빈 마커
}
OP_MARKEREDGE = {
    "projection": None,
    "kinetic":    None,
    "local":      None,
    "nonlocal":   None,
    "tensordot":  None,
}
OP_MARKEREDGEWIDTH = {
    "projection": 0.8,
    "kinetic":    1.2,
    "local":      1.2,
    "nonlocal":   1.2,
    "tensordot":  1.8,       # 조금 더 두껍게
}

# Display names for dtypes (UI/legend labels)
DISPLAY_NAME = {"SP": "FP32", "TF32": "TF32", "BF16": "BF16", "HP": "FP16"}

# Operation display names for titles/legends
OP_DISPLAY = {
    "projection": "Projection",
    "kinetic": "Kinetic",
    "local": "Local",
    "nonlocal": "Nonlocal",
    "tensordot": "n-mode product",  # rename for figure
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

def _sanitize_scale(scale_value: str, default_value: str) -> str:
    """Validate scale string; fallback to default if invalid."""
    return scale_value if scale_value in ('linear', 'log') else default_value

def calculate_natoms(supercell_size, base_natoms=5):
    """Calculate number of atoms based on supercell size."""
    return base_natoms * supercell_size

# ==============================
# Parsing
# ==============================
def parse_timer_summaries(log_content: str) -> pd.DataFrame:
    """Parse 'Timer Summary' blocks from the log and return a DataFrame."""
    sections = re.split(r"^=+\s*Timer Summary\s*=+\s*$", log_content, flags=re.MULTILINE)
    results = []

    for block in sections[1:]:
        # Expected:
        # Operation: projection (DP) | 0.123 | 45
        operation_pattern = r"Operation:\s*(\w+)\s*\((\w+)\)\s*\|\s*([\d.]+)\s*\|\s*(\d+)"
        operations_found = re.findall(operation_pattern, block)

        supercell_match = re.search(r"supercell:\s*\[([\d,\s]+)\]", block)
        dtype_match = re.search(r"dtype:\s*(\w+)", block)
        dense_proj_match = re.search(r"use_dense_proj:\s*(\w+)", block)
        dense_kinetic_match = re.search(r"use_dense_kinetic:\s*(\w+)", block)

        supercell_size = None
        if supercell_match:
            supercell_numbers = [int(x.strip()) for x in supercell_match.group(1).split(",")]
            supercell_size = supercell_numbers[-1]

        for op_name, op_dtype, time, count in operations_found:
            record = {
                "operation": op_name,
                "operation_dtype": op_dtype,
                "time_msec": float(time) * 1000.0,   # sec -> msec
                "count": int(count),
                "supercell_size": supercell_size,
                "dtype": dtype_match.group(1) if dtype_match else None,
                "use_dense_proj": (dense_proj_match and dense_proj_match.group(1).lower() == "true"),
                "use_dense_kinetic": (dense_kinetic_match and dense_kinetic_match.group(1).lower() == "true"),
            }
            results.append(record)

    return pd.DataFrame(results)

# ==============================
# Data prep for plotting
# ==============================
def prepare_plot_data(df: pd.DataFrame, use_dense_proj=False, material_name=None):
    """Prepare data per operation for plotting speedup curves and DP baseline times."""
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

        for dtype in dtype_list:
            if (operation, dtype) in other_pivot.index:
                other_times = other_pivot.loc[(operation, dtype)]
                dp_times = dp_pivot.loc[operation]
                aligned = dp_times.align(other_times, join='inner')
                if aligned[0].empty:
                    speedup = [np.nan] * len(natoms)
                else:
                    sp = (aligned[0] / aligned[1]).values.tolist()
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
                             include_dp_comparison=False,
                             xscale='log',
                             yscale_speedup='linear',
                             yscale_dp_time='linear',
                             font_sizes=None):
    """
    Create performance plots for each operation.
    If include_dp_comparison=True, add DP comparison as panel (f).
    """
    font_sizes = _merge_font_sizes(font_sizes)
    xscale = _sanitize_scale(xscale, 'log')
    yscale_speedup = _sanitize_scale(yscale_speedup, 'linear')
    yscale_dp_time = _sanitize_scale(yscale_dp_time, 'linear')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_idx = 0
    # Speedup panels (a)-(e)
    for operation in operations:
        if operation not in plot_data:
            continue

        ax = axes[plot_idx]
        op_data = plot_data[operation]
        natoms_array = np.array(op_data['natoms'], dtype=float)

        # Primary y-axis: Speedup curves
        for dtype in PLOT_DTYPES:  # only FP32/TF32/BF16
            speedup_array = np.array(op_data['speedup'][dtype], dtype=float)
            min_len = min(len(natoms_array), len(speedup_array))
            x = natoms_array[:min_len]
            y = speedup_array[:min_len]
            valid = ~np.isnan(y)
            if np.any(valid):
                ax.plot(
                    x[valid], y[valid],
                    marker=markers[dtype],
                    color=colors[dtype],
                    label=DISPLAY_NAME.get(dtype, dtype),
                    linewidth=2,
                    markersize=8
                )

        # Formatting
        ax.set_xlabel('Number of atoms', fontsize=font_sizes['label'])
        ax.set_ylabel('Speedup', fontsize=font_sizes['label'])
        disp_name = OP_DISPLAY.get(operation, operation.capitalize())
        ax.set_title(f'({chr(97+plot_idx)}) {disp_name}',
                     fontsize=font_sizes['subtitle'], fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Axis scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale_speedup)
        _apply_axis_fonts(ax, font_sizes)

        # Secondary y-axis: FP64 time (optional)
        if show_dp_time and 'dp_time' in op_data:
            dp_time_array = np.array(op_data['dp_time'], dtype=float)
            min_len = min(len(natoms_array), len(dp_time_array))
            x2 = natoms_array[:min_len]
            y2 = dp_time_array[:min_len]
            valid2 = ~np.isnan(y2)
            if np.any(valid2):
                ax2 = ax.twinx()
                ax2.plot(x2[valid2], y2[valid2], 'k--', linewidth=1.6, alpha=0.8, label='FP64 time')
                ax2.set_ylabel('FP64 time (ms)', fontsize=font_sizes['label'], color='black')
                ax2.tick_params(axis='y', labelcolor='black', labelsize=font_sizes['tick'])
                ax2.set_yscale(yscale_dp_time)

        plot_idx += 1
        if plot_idx == 5:  # up to (e)
            break

    # Panel (f): DP comparison panel
    if include_dp_comparison:
        axf = axes[5]
        _plot_dp_comparison_panel(
            axf, plot_data, xscale, yscale_dp_time, font_sizes, panel_idx=5, title_suffix=title_suffix
        )
    else:
        fig.delaxes(axes[5])

    # Suptitle
    main_title = 'Performance Comparison'
    if title_suffix:
        main_title = f'{main_title} {title_suffix}'
    fig.suptitle(main_title, fontsize=font_sizes['title'], fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    if save_prefix:
        suffix = "_with_dp_comp" if include_dp_comparison else ""
        fig.savefig(f'{save_prefix}_performance_plots{suffix}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_prefix}_performance_plots{suffix}.pdf', bbox_inches='tight')
        print(f"Saved: {save_prefix}_performance_plots{suffix}.png/pdf")

    # Legend-only (respect show_dp_time and exclude HP)
    if create_legend_figure:
        create_legend_only(save_prefix, font_sizes, show_dp_time=show_dp_time)

    return fig

def _plot_dp_comparison_panel(ax: plt.Axes,
                              plot_data: Dict[str, Any],
                              xscale: str,
                              yscale_dp_time: str,
                              font_sizes: Dict[str, int],
                              panel_idx: int = 5,
                              title_suffix: str = ""):
    """
    DP baseline time comparison as panel (f) with in-axes legend.
    Uses OP_* dictionaries for per-operation styling (color/marker/linestyle/marker face & edge).
    - projection: filled marker (facecolor=color), black edge (thinner)
    - tensordot:  hollow marker (facecolor='white'), colored edge (thicker)
    """
    has_any = False

    for op in operations:
        if op not in plot_data:
            continue
        op_data = plot_data[op]
        if 'dp_time' not in op_data or op_data['dp_time'] is None:
            continue

        x = np.array(op_data['natoms'], dtype=float)
        y = np.array(op_data['dp_time'], dtype=float)
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        valid = ~np.isnan(y)
        if not np.any(valid):
            continue

        has_any = True

        # ---- 스타일 Lookup (fallback 포함) ----
        color     = OP_COLORS.get(op, 'C0')
        marker    = OP_MARKERS.get(op, 'o')
        linestyle = OP_LINESTYLES.get(op, '-')

        # facecolor: None -> 선색(color)로 채움(=filled), 'white' 등 명시 -> 해당 색 사용
        mfc_cfg = OP_MARKERFACE.get(op, 'white')
        markerfacecolor = color if mfc_cfg is None else mfc_cfg

        # edgecolor: None -> 선색(color), 그 외 명시된 색 사용
        mec_cfg = OP_MARKEREDGE.get(op, None)
        markeredgecolor = color if mec_cfg is None else mec_cfg

        # edgewidth: 기본 1.6
        markeredgewidth = OP_MARKEREDGEWIDTH.get(op, 1.6)

        ax.plot(
            x[valid], y[valid],
            marker=marker,
            linewidth=2.2,
            markersize=7.5,
            color=color,
            linestyle=linestyle,
            alpha=0.95,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth,
            label=OP_DISPLAY.get(op, op.capitalize()),
            zorder=3
        )

    # ---- 축/레이블/스케일/범례 ----
    ax.set_xlabel('Number of atoms', fontsize=font_sizes['label'])
    ax.set_ylabel('FP64 time (ms)', fontsize=font_sizes['label'])
    title = f'({chr(97 + panel_idx)}) DP Baseline Time'
    ax.set_title(title, fontsize=font_sizes['subtitle'], fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)

    ax.set_xscale(_sanitize_scale(xscale, 'log'))
    ax.set_yscale(_sanitize_scale(yscale_dp_time, 'linear'))
    _apply_axis_fonts(ax, font_sizes)

    if has_any:
        leg = ax.legend(loc='best', fontsize=font_sizes['tick'], frameon=True)
        leg.get_frame().set_alpha(0.9)
    else:
        ax.text(0.5, 0.5, 'No DP-time data available',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=font_sizes['label'], color='gray')


def create_legend_only(save_prefix="", font_sizes=None, show_dp_time=True):
    """
    Legend-only figure reflecting visible dtypes (FP32/TF32/BF16) and FP64 time (optional).
    Shows line + marker in legend.
    """
    font_sizes = _merge_font_sizes(font_sizes)
    fig_legend = plt.figure(figsize=(6, 2))

    legend_elements = []
    for dtype in PLOT_DTYPES:  # exclude HP from legend
        legend_elements.append(
            plt.Line2D([0], [0],
                       marker=markers[dtype],
                       color=colors[dtype],
                       markerfacecolor=colors[dtype],
                       markersize=10,
                       label=DISPLAY_NAME.get(dtype, dtype),
                       linewidth=2,
                       linestyle='-',
                       markeredgewidth=1.5,
                       markeredgecolor=colors[dtype])
        )

    if show_dp_time:
        legend_elements.append(
            plt.Line2D([0], [0],
                       color='black',
                       linewidth=1.6,
                       linestyle='--',
                       alpha=0.8,
                       label='FP64 time (right axis)')
        )

    legend = fig_legend.legend(
        handles=legend_elements,
        loc='center',
        ncol=(len(PLOT_DTYPES) + (1 if show_dp_time else 0)),
        frameon=True,
        fontsize=font_sizes['tick'],
        # title='Precision Types',
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
                              xscale='log',
                              yscale_speedup='linear',
                              yscale_dp_time='linear',
                              font_sizes=None,
                              show_dp_time=True):
    """
    Single operation plot with separate y-scale options and optional FP64 time on right axis.
    """
    font_sizes = _merge_font_sizes(font_sizes)
    xscale = _sanitize_scale(xscale, 'log')
    yscale_speedup = _sanitize_scale(yscale_speedup, 'linear')
    yscale_dp_time = _sanitize_scale(yscale_dp_time, 'linear')

    if operation not in plot_data:
        print(f"Operation '{operation}' not found in data")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    op_data = plot_data[operation]
    natoms_array = np.array(op_data['natoms'], dtype=float)
    has_data = False

    for dtype in PLOT_DTYPES:  # exclude HP from plot
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
                label=DISPLAY_NAME.get(dtype, dtype),
                linewidth=2,
                markersize=8
            )

    ax.set_xlabel('Number of atoms', fontsize=font_sizes['label'])
    ax.set_ylabel('Speedup', fontsize=font_sizes['label'])
    disp_name = OP_DISPLAY.get(operation, operation.capitalize())
    ax.set_title(f'{disp_name} Performance', fontsize=font_sizes['title'], fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale_speedup)
    _apply_axis_fonts(ax, font_sizes)

    if has_data:
        ax.legend(loc='best', fontsize=font_sizes['tick'])
    else:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=font_sizes['label'], color='gray')

    if show_dp_time and 'dp_time' in op_data:
        dp_time_array = np.array(op_data['dp_time'], dtype=float)
        min_len = min(len(natoms_array), len(dp_time_array))
        x2 = natoms_array[:min_len]
        y2 = dp_time_array[:min_len]
        valid2 = ~np.isnan(y2)
        if np.any(valid2):
            ax2 = ax.twinx()
            ax2.plot(x2[valid2], y2[valid2], 'k--', linewidth=1.6, alpha=0.8)
            ax2.set_ylabel('FP64 time (ms)', fontsize=font_sizes['label'], color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=font_sizes['tick'])
            ax2.set_yscale(yscale_dp_time)

    plt.tight_layout()

    if save_name:
        fig.savefig(f'{save_name}_{operation}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_name}_{operation}.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}_{operation}.png/pdf")

    return fig

# ==============================
# CLI
# ==============================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate performance comparison plots from log file'
    )

    parser.add_argument('log_file', type=str, help='Path to the log file containing Timer Summary data')
    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output files (default: basename of log file)')
    parser.add_argument('--material', type=str, default=None, choices=list(natoms_dict.keys()) + [None],
                        help='Material name for natoms calculation (default: BaTiO3 with 5 atoms/unit cell)')
    parser.add_argument('--base-natoms', type=int, default=5, help='Base number of atoms per unit cell (default: 5 for BaTiO3)')
    parser.add_argument('--use-dense-proj', action='store_true', help='Use dense projection data (default: sparse)')

    parser.add_argument('--no-dp-time', action='store_true', help='Do not show FP64 time on secondary y-axis')
    parser.add_argument('--no-legend', action='store_true', help='Do not create separate legend figure')
    parser.add_argument('--individual-plots', action='store_true', help='Create individual plots for each operation')
    parser.add_argument('--dp-comparison', action='store_true', help='Add DP-time comparison as panel (f) in the main figure')

    parser.add_argument('--title-suffix', type=str, default='', help='Additional text to add to the main title')
    parser.add_argument('--xscale', type=str, default='log', choices=['linear', 'log'], help='X-axis scale (default: log)')
    parser.add_argument('--yscale-speedup', type=str, default='linear', choices=['linear', 'log'],
                        help='Y-axis scale for speedup curves (default: linear)')
    parser.add_argument('--yscale-dp-time', type=str, default='linear', choices=['linear', 'log'],
                        help='Y-axis scale for FP64 time axes (right axis & panel f) (default: linear)')

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

        print(f"Creating performance plots (xscale={args.xscale}, yscale_speedup={args.yscale_speedup}, yscale_dp_time={args.yscale_dp_time})...")
        fig = create_performance_plots(
            plot_data,
            save_prefix=args.output_prefix,
            title_suffix=args.title_suffix,
            show_dp_time=not args.no_dp_time,
            create_legend_figure=not args.no_legend,
            include_dp_comparison=args.dp_comparison,
            xscale=args.xscale,
            yscale_speedup=args.yscale_speedup,
            yscale_dp_time=args.yscale_dp_time,
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
                        yscale_speedup=args.yscale_speedup,
                        yscale_dp_time=args.yscale_dp_time,
                        font_sizes=font_sizes,
                        show_dp_time=not args.no_dp_time
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
