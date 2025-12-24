"""
Peak memory usage analysis.

x-axis: Number of atoms
y-axis: Peak memory (GiB)
legend: FP64, FP32, MP1*
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

# For system colors
system_colors = {
    "CNT_6_0": "#e377c2",      # pink
    "MgO_1x1x2": "#17becf",    # cyan
    "Si_diamond_2x2x1": "#bcbd22",  # yellow-green
}

# Mapping from opt_name in CSV to display labels
OPT_NAME_MAPPING = {
    "DP": "FP64",
    "SP": "FP32", 
    "MP_scheme1_BF164precond": "MP1*",
}

# Colors for different methods
METHOD_COLORS = {
    "FP64": "#1f77b4",  # blue
    "FP32": "#ff7f0e",  # orange
    "MP1*": "#2ca02c",  # green
}

# Line styles for different methods
METHOD_LINESTYLES = {
    "FP64": "-",
    "FP32": "--",
    "MP1*": "-.",
}

# Markers for different methods
METHOD_MARKERS = {
    "FP64": "o",
    "FP32": "s",
    "MP1*": "^",
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


def get_short_system_name(system_name):
    """Convert system name to short form for display."""
    mapping = {"CNT_6_0": "CNT", "MgO_1x1x2": "MgO", "Si_diamond_2x2x1": "Si"}
    return mapping.get(system_name, system_name)


# --- Data loading ------------------------------------------------------------

def load_peak_alloc_data(csv_path="peak_alloc_results.csv"):
    """
    Load peak allocation data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded data with processed columns
    """
    df = pd.read_csv(csv_path)
    
    # Add Natoms column
    df["Natoms"] = df.apply(lambda row: row["n"] * natoms_dict[row["System"]], axis=1)
    
    # Map opt_name to display method name
    df["Method"] = df["opt_name"].map(OPT_NAME_MAPPING)
    
    # Filter only the methods we want to display
    df = df[df["Method"].notna()].copy()
    
    return df


# --- Plotting functions ------------------------------------------------------

def plot_peak_memory_by_system(
    df, 
    system_name, 
    save_path=None, 
    legend_path=None,
    custom_font_config=None, 
    figsize=(9, 6), 
    ncol=3, 
    show=True,
):
    """
    Plot peak memory usage for a single system.
    
    x-axis: Number of atoms
    y-axis: Peak memory (GiB)
    legend: FP64, FP32, MP1*
    
    Args:
        df: DataFrame with peak allocation data
        system_name: Name of the system to plot
        save_path: Path to save the main figure
        legend_path: Path to save the separate legend box
        custom_font_config: Custom font configuration
        figsize: Figure size
        ncol: Number of columns in legend
        show: Whether to show the plot interactively
    """
    font_config = custom_font_config or CURRENT_FONT_CONFIG
    fig, ax = plt.subplots(figsize=figsize)
    alpha = 0.85
    
    # Filter data for the specified system
    system_data = df[df["System"] == system_name].copy()
    
    handles, labels = [], []
    
    # Plot each method
    for method in ["FP64", "FP32", "MP1*"]:
        method_data = (
            system_data[system_data["Method"] == method]
            .sort_values("Natoms")
            .dropna(subset=["peak_alloc_GiB"])
        )
        
        if not method_data.empty:
            h, = ax.plot(
                method_data["Natoms"], 
                method_data["peak_alloc_GiB"],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                alpha=alpha,
                label=method,
                color=METHOD_COLORS[method],
            )
            handles.append(h)
            labels.append(method)
    
    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Peak memory (GiB)", fontsize=font_config["label"])
    
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.tick_params(axis="both", labelsize=font_config["tick"])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Save a separate legend box if requested
    if legend_path and len(handles) > 0:
        fig_leg = plt.figure(figsize=(4, 1.5))
        fig_leg.legend(
            handles, labels, loc="center", frameon=True, ncol=ncol,
            fontsize=font_config["legend"], handlelength=3.0, handleheight=1.0
        )
        fig_leg.tight_layout()
        fig_leg.savefig(legend_path, bbox_inches="tight")
        print(f"Saved legend to {legend_path}")
        plt.close(fig_leg)


def plot_peak_memory_all_systems(
    df,
    save_path=None,
    custom_font_config=None,
    figsize=(18, 5),
    show=True,
):
    """
    Plot peak memory usage for all systems in a single figure with subplots.
    
    Args:
        df: DataFrame with peak allocation data
        save_path: Path to save the figure
        custom_font_config: Custom font configuration
        figsize: Figure size
        show: Whether to show the plot interactively
    """
    font_config = custom_font_config or CURRENT_FONT_CONFIG
    
    systems = df["System"].unique()
    n_systems = len(systems)
    
    fig, axes = plt.subplots(1, n_systems, figsize=figsize, sharey=True)
    
    if n_systems == 1:
        axes = [axes]
    
    alpha = 0.85
    all_handles, all_labels = [], []
    
    for idx, system_name in enumerate(systems):
        ax = axes[idx]
        system_data = df[df["System"] == system_name].copy()
        
        # Plot each method
        for method in ["FP64", "FP32", "MP1*"]:
            method_data = (
                system_data[system_data["Method"] == method]
                .sort_values("Natoms")
                .dropna(subset=["peak_alloc_GiB"])
            )
            
            if not method_data.empty:
                h, = ax.plot(
                    method_data["Natoms"], 
                    method_data["peak_alloc_GiB"],
                    marker=METHOD_MARKERS[method],
                    linestyle=METHOD_LINESTYLES[method],
                    alpha=alpha,
                    label=method,
                    color=METHOD_COLORS[method],
                )
                if idx == 0:  # Only collect handles from first subplot
                    all_handles.append(h)
                    all_labels.append(method)
        
        ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
        ax.set_title(get_short_system_name(system_name), fontsize=font_config["title"])
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
        ax.tick_params(axis="both", labelsize=font_config["tick"])
        ax.set_ylim(bottom=0)
    
    # Set y-label only on the leftmost subplot
    axes[0].set_ylabel("Peak memory (GiB)", fontsize=font_config["label"])
    
    # Add a single legend for all subplots
    fig.legend(
        all_handles, all_labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(all_labels),
        frameon=True,
        fontsize=font_config["legend"]
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved combined plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    """Main function to generate peak memory plots."""
    
    # Setup plot style (same as original)
    setup_plot_style(base_font_size=28)
    figsize = (10, 7)
    show = False  # Set to True to display plots interactively
    
    # Load data
    df = load_peak_alloc_data("peak_alloc_results.csv")
    print("Loaded peak allocation data:")
    print(df.head())
    print(f"\nSystems: {df['System'].unique()}")
    print(f"Methods: {df['Method'].unique()}")
    
    # Create output directory
    os.makedirs("./Figures_peak_memory", exist_ok=True)
    
    # Generate individual plots for each system
    for system in df["System"].unique():
        filename = f"./Figures_peak_memory/peak_memory_{system}.svg"
        legend_filename = "./Figures_peak_memory/legend_peak_memory.svg"
        
        plot_peak_memory_by_system(
            df, 
            system, 
            save_path=filename,
            legend_path=legend_filename,
            figsize=figsize,
            show=show
        )
    
    # Generate combined plot with all systems
    combined_filename = "./Figures_peak_memory/peak_memory_all_systems.svg"
    plot_peak_memory_all_systems(
        df,
        save_path=combined_filename,
        figsize=(18, 5),
        show=show
    )
    
    print("\n✓ All peak memory plots generated successfully!")


if __name__ == "__main__":
    main()
