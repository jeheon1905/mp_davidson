"""
Performance enhancement of MP1*.

x-axis: the number of atoms
y-axis: speedups
legend: different GPUs
subfigure: systems
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Metadata & naming -------------------------------------------------------

# natoms info
natoms_dict = {
    "CNT_6_0": 24,
    "MgO_1x1x2": 16,
    "Si_diamond_2x2x1": 32,
}

# Global font config cache
CURRENT_FONT_CONFIG = {
    "base": 14,
    "title": 18.2,
    "label": 16.1,
    "tick": 14,
    "legend": 11.9,
}

# Centralized GPU metadata: display label, color, and directory candidates.
# - label: what appears in legends/plots
# - dir_candidates: possible folder suffixes after "csv_results_"
#   (we'll auto-resolve the first that exists; if none exist, we’ll still try the first)
GPU_META = {
    "A100": {
        "label": "A100",
        "color": "#1f77b4",  # blue
        "dir_candidates": ["A100"],
    },
    "RTX A6000": {
        "label": "RTX A6000",
        "color": "#2ca02c",  # green
        "dir_candidates": ["A6000", "RTX_A6000", "RTXA6000"],
    },
    "L40S": {
        "label": "L40S",
        "color": "#ff7f0e",  # orange
        "dir_candidates": ["L40S"],
    },
    "RTX 4090": {
        "label": "RTX 4090",
        "color": "#9467bd",  # purple
        "dir_candidates": ["RTX4090", "RTX_4090", "4090", "GeForce_RTX_4090", "RTX-4090"],
    },
}

def resolve_gpu_dir(gpu_key, base_dir="."):
    """
    Resolve a concrete directory name for csv_results_* based on candidates.
    Returns e.g. 'csv_results_A100' or 'csv_results_RTX4090'.
    If no candidate exists on disk, falls back to the first candidate.
    """
    meta = GPU_META[gpu_key]
    for cand in meta["dir_candidates"]:
        candidate = os.path.join(base_dir, f"csv_results_{cand}")
        if os.path.isdir(candidate):
            return f"csv_results_{cand}"
    # Fallback: use the first candidate to construct the path anyway
    return f"csv_results_{meta['dir_candidates'][0]}"

# For system colors (used in type3 plot)
system_colors = {
    "CNT_6_0": "#e377c2",      # pink
    "MgO_1x1x2": "#17becf",    # cyan
    "Si_diamond_2x2x1": "#bcbd22",  # yellow-green
}

# --- Styling -----------------------------------------------------------------

def setup_plot_style(base_font_size=14):
    """Set global Matplotlib styles with proportional scaling."""
    scale_factors = {
        "title": 1.3,
        "label": 1.15,
        "tick": 1.0,
        "legend": 0.85,
        "subtitle": 1.1,
    }

    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.titlesize": base_font_size * scale_factors["title"],
            "figure.titlesize": base_font_size * scale_factors["title"],
            "axes.labelsize": base_font_size * scale_factors["label"],
            "xtick.labelsize": base_font_size * scale_factors["tick"],
            "ytick.labelsize": base_font_size * scale_factors["tick"],
            "legend.fontsize": base_font_size * scale_factors["legend"],
            "legend.title_fontsize": base_font_size * scale_factors["legend"],
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": True,
            "lines.linewidth": base_font_size / 10,
            "lines.markersize": base_font_size * 0.4,
            "grid.linewidth": base_font_size / 20,
            "grid.alpha": 0.3,
        }
    )
    global CURRENT_FONT_CONFIG
    CURRENT_FONT_CONFIG = {
        "base": base_font_size,
        "title": base_font_size * scale_factors["title"],
        "label": base_font_size * scale_factors["label"],
        "tick": base_font_size * scale_factors["tick"],
        "legend": base_font_size * scale_factors["legend"],
    }

# --- IO helpers --------------------------------------------------------------

def get_results_dict(
    gpu_key,
    system_name,
    method,
    supercell_list=range(1, 11),
    target="speedup",  # options=["speedup", "baseline_time", "time"]
    base_dir="."
):
    """
    Read CSV rows for a given GPU/system/method and return {natoms: value}.
    GPU directory is resolved from GPU_META; GPU label is detached from directory naming.
    """
    results_dict = {}
    csv_dir = resolve_gpu_dir(gpu_key, base_dir=base_dir)

    for supercell in supercell_list:
        filename = os.path.join(csv_dir, f"{system_name}.1_1_{supercell}.csv")

        try:
            df = pd.read_csv(filename)
            data = df[df["Label"] == "davidson"]

            if target == "speedup":
                label = f"1_1_{supercell}_{method}.speed_Speedup"
            elif target == "baseline_time":
                label = "Baseline_Time"
            elif target == "time":
                label = f"1_1_{supercell}_{method}.speed_Time"
            else:
                raise ValueError("Invalid target: use 'speedup', 'baseline_time', or 'time'.")

            value = data[label].iloc[0]
            natoms = supercell * natoms_dict[system_name]
            results_dict[natoms] = value

        except FileNotFoundError:
            print(f"[WARN] File not found: {filename}")
        except Exception as e:
            print(f"[WARN] Error processing {filename}: {e}")

    return results_dict


def create_combined_dataframe(gpu_keys, system_name_list, supercell_list, method, base_dir="."):
    """Merge all results into a single dataframe, using *labels* for GPU names."""
    all_results = []

    for gpu_key in gpu_keys:
        label = GPU_META[gpu_key]["label"]  # pretty label for plotting
        for system_name in system_name_list:
            speedup_dict = get_results_dict(gpu_key, system_name, method, supercell_list, "speedup", base_dir)
            baseline_dict = get_results_dict(gpu_key, system_name, method, supercell_list, "baseline_time", base_dir)
            time_dict = get_results_dict(gpu_key, system_name, method, supercell_list, "time", base_dir)

            for supercell in supercell_list:
                natoms = supercell * natoms_dict[system_name]
                if natoms in speedup_dict:
                    row = {
                        "GPU": label,  # store label (e.g., "RTX 4090")
                        "System": system_name,
                        "Supercell": supercell,
                        "Cell_Name": f"1_1_{supercell}",
                        "Natoms": natoms,
                        "Speedup": speedup_dict.get(natoms, None),
                        "Baseline_Time": baseline_dict.get(natoms, None),
                        "Method_Time": time_dict.get(natoms, None),
                        "Method": method,
                    }
                    all_results.append(row)

    df = pd.DataFrame(all_results)
    # df = df.sort_values(["GPU", "System", "Supercell"]).reset_index(drop=True)
    df = df.sort_values(["System", "Supercell"]).reset_index(drop=True)
    return df

# --- Plot helpers ------------------------------------------------------------

def get_short_system_name(system_name):
    mapping = {"CNT_6_0": "CNT", "MgO_1x1x2": "MgO", "Si_diamond_2x2x1": "Si"}
    return mapping.get(system_name, system_name)

def plot_type1_speedup_by_gpu(
    df, system_name, save_path=None, legend_path=None,
    custom_font_config=None, figsize=(9, 6), ncol=4, show=True,
):
    """Type 1: For a single system, x=natoms, y=speedup, legend=GPU.
       - If legend_path is None: do NOT draw or save any legend.
       - If legend_path is not None: save a standalone legend box to legend_path.
    """
    font_config = custom_font_config or CURRENT_FONT_CONFIG
    fig, ax = plt.subplots(figsize=figsize)
    alpha = 0.85

    system_data = df[df["System"] == system_name].copy()
    handles, labels = [], []

    for gpu_label in system_data["GPU"].unique():
        gpu_data = (
            system_data[system_data["GPU"] == gpu_label]
            .sort_values("Natoms")
            .dropna(subset=["Speedup"])
        )
        # resolve color by label
        color = None
        for k, meta in GPU_META.items():
            if meta["label"] == gpu_label:
                color = meta["color"]
                break

        h, = ax.plot(
            gpu_data["Natoms"], gpu_data["Speedup"],
            marker="o",
            alpha=alpha,
            label=gpu_label,
            color=color or "gray",
        )
        handles.append(h); labels.append(gpu_label)

    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Speedup", fontsize=font_config["label"])

    # y=1 reference line (thicker)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.9, linewidth=2.0)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.tick_params(axis="both", labelsize=font_config["tick"])
    ax.set_ylim(bottom=0.5)

    # 🔑 Force y-ticks
    # Get current y-limit
    ymin, ymax = ax.get_ylim()
    # Generate ticks from 1 to ymax with step 2
    yticks = np.arange(1, ymax + 1, 2)
    ax.set_yticks(yticks)

    # No legend on the main figure when legend_path is None
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)   # free memory when not showing

    # Save a separate legend box only when legend_path is provided
    if legend_path:
        fig_leg = plt.figure(figsize=(4, 2))
        fig_leg.legend(
            handles, labels, loc="center", frameon=True, ncol=ncol,
            fontsize=font_config["legend"], handlelength=3.0, handleheight=1.0
        )
        fig_leg.tight_layout()
        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)


def plot_type2_time_comparison(
    df, system_name, save_path=None, legend_path=None,
    custom_font_config=None, figsize=(9, 6), ncol=4, show=True,
):
    """Type 2: For a single system, x=natoms, y=time, legend=FP64/MP1* per GPU."""
    font_config = custom_font_config or CURRENT_FONT_CONFIG
    fig, ax = plt.subplots(figsize=figsize)

    system_data = df[df["System"] == system_name].copy()
    line_styles = {"FP64": "-", "MP1*": "--"}
    alpha = 0.85

    handles, labels = [], []

    for gpu_label in system_data["GPU"].unique():
        gpu_data = system_data[system_data["GPU"] == gpu_label].sort_values("Natoms")
        # resolve color by label
        color = None
        for k, meta in GPU_META.items():
            if meta["label"] == gpu_label:
                color = meta["color"]
                break

        dp_data = gpu_data.dropna(subset=["Baseline_Time"])
        if not dp_data.empty:
            h1, = ax.plot(
                dp_data["Natoms"], dp_data["Baseline_Time"],
                marker="o", linestyle=line_styles["FP64"],
                markersize=plt.rcParams["lines.markersize"] * 0.8,
                alpha=alpha,
                label=f"FP64 ({gpu_label})",
                color=color or "gray",
            )
            handles.append(h1); labels.append(f"FP64 ({gpu_label})")

        mp1_data = gpu_data.dropna(subset=["Method_Time"])
        if not mp1_data.empty:
            h2, = ax.plot(
                mp1_data["Natoms"], mp1_data["Method_Time"],
                # marker="s", linestyle=line_styles["MP1*"],
                marker="^", linestyle=line_styles["MP1*"],
                alpha=alpha,
                label=f"MP1* ({gpu_label})",
                color=color or "gray",
            )
            handles.append(h2); labels.append(f"MP1* ({gpu_label})")

    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Diag. time (sec)", fontsize=font_config["label"])
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.tick_params(axis="both", labelsize=font_config["tick"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Only show interactively if requested
    if show:
        plt.show()
    else:
        plt.close(fig)   # free memory when not showing

    # Save a separate legend box if requested
    if legend_path and len(handles) > 0:
        fig_leg = plt.figure(figsize=(4, 2))
        fig_leg.legend(
            handles, labels, loc="center", frameon=True, ncol=ncol,
            fontsize=font_config["legend"], handlelength=3.0, handleheight=1.0
        )
        fig_leg.tight_layout()
        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)

def plot_type3_speedup_by_system(
    df, gpu_label, save_path=None, custom_font_config=None, figsize=(9, 6)
):
    """Type 3: For a single GPU, x=natoms, y=speedup, legend=systems."""
    font_config = custom_font_config or CURRENT_FONT_CONFIG
    fig, ax = plt.subplots(figsize=figsize)

    gpu_data = df[df["GPU"] == gpu_label].copy()
    for system in gpu_data["System"].unique():
        system_df = gpu_data[gpu_data["System"] == system].sort_values("Natoms").dropna(subset=["Speedup"])
        ax.plot(
            system_df["Natoms"], system_df["Speedup"],
            marker="o",
            label=get_short_system_name(system),
            color=system_colors.get(system, "gray"),
        )

    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Speedup", fontsize=font_config["label"])
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.legend(loc="best", frameon=True, shadow=False, fontsize=font_config["legend"])
    ax.tick_params(axis="both", labelsize=font_config["tick"])
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    # Global style
    setup_plot_style(base_font_size=28)
    figsize = (10, 7)
    show = True
    show =False 

    system_name_list = ["CNT_6_0", "MgO_1x1x2", "Si_diamond_2x2x1"]
    supercell_list = list(range(1, 23))
    method = "MP_scheme1_BF164precond"

    # Use canonical labels for consistency in figures.
    # (Directories are resolved automatically via GPU_META.dir_candidates)
    gpu_keys = ["A100", "RTX A6000", "L40S", "RTX 4090"]

    combined_df = create_combined_dataframe(
        gpu_keys, system_name_list, supercell_list, method, base_dir="."
    )
    print(combined_df)

    # Output directories may need to exist
    os.makedirs("./Figures_performance_vs_size", exist_ok=True)

    # Generate figures per system
    for system in system_name_list:
        # Type 1: Speedup vs size (legend by GPU)
        filename = f"./Figures_performance_vs_size/type1_speedup_{system}.svg"
        # plot_type1_speedup_by_gpu(combined_df, system, filename, legend_path=None, figsize=figsize)
        filename_legend = "./Figures_performance_vs_size/legend_box1.svg"
        plot_type1_speedup_by_gpu(combined_df, system, filename, legend_path=filename_legend, figsize=figsize, show=show)
        print(f"Save plot to {filename}")

        # Type 2: Time comparison (FP64 vs MP1*) per GPU
        filename = f"./Figures_performance_vs_size/type2_time_{system}.svg"
        filename_legend = "./Figures_performance_vs_size/legend_box2.svg"
        plot_type2_time_comparison(combined_df, system, filename, legend_path=filename_legend, figsize=figsize, show=show)
        print(f"Save legend box to {filename_legend}")
        print(f"Save plot to {filename}")

    # Example: Type 3 for a specific GPU
    # plot_type3_speedup_by_system(combined_df, 'RTX 4090', 'type3_rtx4090_speedup.svg', figsize=figsize)

