"""
Plot residual history.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


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
        help="number of eigenvalues (default to None)",
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
    args = parser.parse_args()

    eigvalHistory, resHistory = torch.load(args.filepath)

    ## TODO: replace eigvalHistory[-1] with the true eigenvalues (from DP calculation)
    if args.ref_filepath is not None:
        eigvalHistory_ref, _ = torch.load(args.ref_filepath)
        ref_eigval = eigvalHistory_ref[-1]
    else:
        ref_eigval = eigvalHistory[-1]
    eigvalHistory = abs(torch.stack(eigvalHistory) - ref_eigval)

    if args.num_eig:
        eigvalHistory = torch.stack(
            [eigval[: args.num_eig] for eigval in eigvalHistory]
        )
        resHistory = torch.stack([res[: args.num_eig] for res in resHistory])

    eigvalHistory = eigvalHistory.to(torch.float64)
    resHistory = resHistory.to(torch.float64)

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
    figsize = [3.6, 4.2]  # Main text 용.
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10
    tick_labelsize = 12
    tick_length = 6
    tick_width = 1.0
    linewidth = 2

    # plot figure
    plt.figure(figsize=figsize)
    if args.title is not None:
        plt.title(args.title, fontsize=title_fontsize)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plot_options = {"color": "k", "linestyle": "-"}
    for i, res in enumerate(result):
        alpha = 1.0 / len(result) * (i + 1)
        plt.plot(i_iter, res, **plot_options, alpha=alpha)

    plt.xlabel("Iteration", fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.yscale("log")
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize)
    # plt.xlim(0, 41)
    plt.ylim(bottom=args.convg_tol)
    plt.tight_layout()

    # save figure
    if args.save:
        save_filename = args.save
        plt.savefig(save_filename, dpi=300)
        print(f"{save_filename} is saved.")
    else:
        plt.show()
