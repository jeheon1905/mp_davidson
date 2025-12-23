"""
Plot residual history with enhanced styling options.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style(base_font_size=14):
    """Sets the style for matplotlib plots, including font sizes and other aesthetic parameters.

    Args:
        base_font_size (int): Base font size for the plot. Other text elements will be scaled relative to this size.
    """
    # scale factors for different text elements
    scale_factors = {
        "title": 1.3,  # 130% of base font size
        "label": 1.15,  # 115% of base font size
        "tick": 1.0,  # 100% of base font size
        "legend": 0.85,  # 85% of base font size
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
            # additional style settings
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": True,
        }
    )


def remove_converged_residuals(resnorm, tol=1e-3):
    """Make residual norms corresponding to converged vectors zero."""
    resnorm = np.array([r.numpy() for r in resnorm])
    unlock = np.full(resnorm.shape[1], True)
    for i in range(len(resnorm)):
        res = resnorm[i][unlock]
        resnorm[i][~unlock] = 0.0  # make resnorm of converged vectors zero
        is_convg = res < tol
        unlock[unlock] = ~is_convg
    return resnorm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        help="input history file (.pt)",
        required=True,
    )
    parser.add_argument(
        "--save",
        type=str,
        help="save filename (e.g., 'fig.png', 'fig.svg', etc.)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--num_eig",
        type=int,
        help="number of occupied eigenvalues (default to None)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--num_virt",
        type=int,
        help="number of virtual eigenvalues to plot in addition to occupied states (default to None)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--convg_tol",
        type=float,
        help="convergence tolerance (default to 1e-3)",
        required=False,
        default=1e-7,
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="plot eigval or residue history",
        choices=["eigval", "residual"],
        required=False,
        default="residue",
    )
    parser.add_argument(
        "--ref_filepath",
        type=str,
        help="reference history file (.pt)",
        default=None,
    )
    parser.add_argument(
        "--title",
        type=str,
        help="title for the plot",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--font_size",
        type=int,
        help="base font size for the plot (default to 14)",
        required=False,
        default=14,
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        help="figure size as width height (default to 3.6 4.2)",
        required=False,
        default=[3.6, 4.2],
    )
    parser.add_argument(
        "--diag_iter",
        type=int,
        help="number of diagonalization iterations to plot (default to all available iterations)",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    # set up plot style
    setup_plot_style(base_font_size=args.font_size)

    eigvalHistory, resHistory = torch.load(args.filepath)

    ## Set reference eigenvalues
    if args.ref_filepath is not None:
        eigvalHistory_ref, _ = torch.load(args.ref_filepath)
        ref_eigval = eigvalHistory_ref[-1]
    else:
        ref_eigval = eigvalHistory[-1]
    eigvalHistory = abs(torch.stack(eigvalHistory) - ref_eigval)

    if args.num_eig:
        num_total = args.num_eig + (args.num_virt if args.num_virt else 0)
        eigvalHistory = torch.stack(
            [eigval[:num_total] for eigval in eigvalHistory]
        )
        resHistory = torch.stack([res[:num_total] for res in resHistory])
    elif args.num_virt:
        # if only num_virt is specified, use all available states
        pass

    eigvalHistory = eigvalHistory.to(torch.float64)
    resHistory = resHistory.to(torch.float64)

    # Apply diag_iter to limit iterations if specified
    if args.diag_iter is not None:
        eigvalHistory = eigvalHistory[: args.diag_iter]
        resHistory = resHistory[: args.diag_iter]

    iterationNumber = len(resHistory)
    i_iter = np.arange(1, iterationNumber + 1)
    resHistory = remove_converged_residuals(resHistory, tol=args.convg_tol)
    resHistory = torch.from_numpy(resHistory).T

    if args.plot == "residual":
        result = resHistory
        ylabel = "Residual Norm"
    else:
        result = eigvalHistory.T
        ylabel = "Eigenvalue Error (Hartree)"

    # figure options
    figsize = args.figsize
    tick_length = 6
    tick_width = 1.0
    linewidth = 1.5

    # plot figure
    plt.figure(figsize=figsize)
    if args.title is not None:
        plt.title(args.title)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Determine the split point between occupied and virtual states
    num_occ = args.num_eig if args.num_eig else len(result)
    num_virt = args.num_virt if args.num_virt else 0

    # Plot occupied states (black)
    plot_options_occ = {"color": "k", "linestyle": "-", "linewidth": linewidth}
    for i in range(min(num_occ, len(result))):
        res = result[i]
        alpha = 1.0 / num_occ * (i + 1)
        plt.plot(i_iter, res, **plot_options_occ, alpha=alpha)

    # Plot virtual states (red)
    if num_virt > 0 and len(result) > num_occ:
        plot_options_virt = {"color": "r", "linestyle": "-", "linewidth": linewidth}
        num_virt_actual = min(num_virt, len(result) - num_occ)
        for i in range(num_occ, num_occ + num_virt_actual):
            res = result[i]
            alpha = 1.0 / num_virt_actual * (i - num_occ + 1)
            plt.plot(i_iter, res, **plot_options_virt, alpha=alpha)

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.tick_params(length=tick_length, width=tick_width)
    # plt.xlim(0, 41)
    plt.ylim(bottom=args.convg_tol)
    plt.tight_layout()

    # save figure
    if args.save:
        save_filename = args.save
        plt.savefig(save_filename)
        print(f"{save_filename} is saved.")
    else:
        plt.show()
