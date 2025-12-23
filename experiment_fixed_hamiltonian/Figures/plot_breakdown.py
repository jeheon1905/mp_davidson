import re
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style(base_font_size=14):
    """전체 플롯 스타일 설정을 한 곳에서 관리합니다.

    Args:
        base_font_size (int): 기본 폰트 크기. 다른 요소들은 이에 비례하여 조정됩니다.
    """
    # 비례 계산을 위한 스케일 팩터
    scale_factors = {
        "title": 1.3,  # 제목은 기본 크기의 130%
        "label": 1.15,  # 축 레이블은 115%
        "tick": 1.0,  # 틱 레이블은 100%
        "legend": 0.85,  # 범례는 85%
    }

    plt.rcParams.update(
        {
            # 기본 폰트 크기
            "font.size": base_font_size,
            # 제목 관련
            "axes.titlesize": base_font_size * scale_factors["title"],
            "figure.titlesize": base_font_size * scale_factors["title"],
            # 축 레이블
            "axes.labelsize": base_font_size * scale_factors["label"],
            # 틱 레이블
            "xtick.labelsize": base_font_size * scale_factors["tick"],
            "ytick.labelsize": base_font_size * scale_factors["tick"],
            # 범례
            "legend.fontsize": base_font_size * scale_factors["legend"],
            "legend.title_fontsize": base_font_size * scale_factors["legend"],
            # 추가 설정 (선택사항)
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": True,
        }
    )


def parse_log_file(filepath):
    """Parse log file and extract timer summary section as a dict."""
    print(f"[+] Parsing log file: {filepath}")
    in_summary = False
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            if (
                "======================== Timer Summary ========================"
                in line
            ):
                in_summary = True
            elif in_summary:
                if re.match(r"-{20,}", line):
                    continue
                elif "Elapsed time" in line or line.strip() == "":
                    break
                else:
                    match = re.match(r"^(.*?)\s+\|\s+([\d\.]+)\s+\|\s+(\d+)", line)
                    if match:
                        label = match.group(1).strip()
                        total = float(match.group(2))
                        data[label] = total
    return data


def aggregate_timing_by_category(breakdown, category_map, phase):
    """Group raw timing data into defined categories.

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

    # Add ETC time
    sum_of_categories = sum(aggregated.values())
    if phase == "fixed":
        etc_time = breakdown["davidson"] - sum_of_categories
    elif phase == "scf":
        etc_time = (
            breakdown["GOSPEL.calculate"]
            + breakdown["GOSPEL.initialize"]
            - sum_of_categories
        )
    elif phase == "preconditioning":
        # Total preconditioning time is measured by the 'Preconditioning' timer
        etc_time = breakdown["Preconditioning"] - sum_of_categories
    else:
        raise ValueError(f"Unknown phase: {phase}")
    aggregated["ETC"] = etc_time
    return aggregated


def plot_accumulated_histogram(
    all_data,
    labels,
    categories,
    colors,
    output_file="accumulated_histogram.svg",
    figsize=(9, 6),
    bar_width=0.8,
    save_legend_separately=True,
    legend_cols=3,
    phase="fixed",
):
    """Draw stacked bar plot (accumulated histogram) with optional separate legend.

    Args:
        all_data: List of dictionaries containing timing data
        labels: X-axis labels
        categories: Category names for stacking
        colors: Color mapping for categories
        output_file: Output filename
        figsize: Figure size tuple (width, height)
        bar_width: Width of bars (0-1)
        save_legend_separately: If True, save legend as a separate file
        legend_cols: Number of columns in legend
    """
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    # 막대 그래프 그리기
    bars = []
    for category in categories:
        values = [data.get(category, 0) for data in all_data]
        bar = ax.bar(
            x,
            values,
            width=bar_width,
            bottom=bottom,
            label=category,
            color=colors.get(category, "gray"),
        )
        bars.append(bar)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if phase == "fixed":
        ax.set_ylabel("Diag. time (sec)")
    elif phase == "scf":
        ax.set_ylabel("SCF time (sec)")
    elif phase == "preconditioning":
        ax.set_ylabel("Preconditioning time (sec)")
    else:
        raise ValueError

    # 그리드 추가 (선택사항)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    if save_legend_separately:
        # 메인 플롯에서 범례 제거
        ax.legend().set_visible(False)

        # 범례를 위한 별도 figure 생성
        fig_legend = plt.figure(figsize=(6, 2))
        handles, labels_legend = ax.get_legend_handles_labels()
        fig_legend.legend(
            handles,
            labels_legend,
            loc="center",
            ncol=legend_cols,
            frameon=True,
            fancybox=True,
            shadow=False,
        )

        # 범례 파일 저장
        legend_filename = (
            output_file.rsplit(".", 1)[0] + "_legend." + output_file.rsplit(".", 1)[1]
        )
        fig_legend.savefig(legend_filename, bbox_inches="tight")
        print(f"[+] Saved legend to {legend_filename}")
        plt.close(fig_legend)
    else:
        # 기존처럼 플롯에 범례 포함
        ax.legend(loc="upper right", ncol=2)

    # 메인 플롯 저장
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    print(f"[+] Saved plot to {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to directory containing log files",
    )
    parser.add_argument(
        "--supercell", type=str, required=True, help="Supercell name like '1_1_8'"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="List of methods, e.g. DP SP MP_scheme1 ...",
        default=["DP", "MP_scheme1", "MP_scheme3", "DP_SP4precond"],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Run labels to appear on the x-axis",
        default=["DP", "MP$^1$", "MP$^3$", "MP$^6$"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="accumulated_histogram.svg",
        help="Output file for plot",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="aggregated_data.json",
        help="Optional JSON output file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="fixed",
        help="calculation type",
        choices=["fixed", "scf", "preconditioning"],
    )
    parser.add_argument(
        "--font_size", type=int, default=18, help="Base font size for all plot elements"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[5.0, 4.0],
        help="Figure size (width height)",
    )
    parser.add_argument("--bar_width", type=float, default=0.8, help="Bar width (0-1)")
    parser.add_argument(
        "--separate_legend", action="store_true", help="Save legend as a separate file"
    )
    parser.add_argument(
        "--legend_cols", type=int, default=3, help="Number of columns in legend"
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        default=".speed.log",
        help="Log filename suffix. Default: .speed.log (e.g., 1_1_8_DP.speed.log)",
    )
    args = parser.parse_args()

    # 플롯 스타일 설정 (폰트 크기 통합 관리)
    setup_plot_style(base_font_size=args.font_size)

    if args.phase == "scf":
        # Timer categories and labels
        category_map = {
            "Diagonalization": ["Davidson.diagonalize"],
            "Initialization": ["GOSPEL.initialize"],
            "Calc. potential and energy": [
                "Hamiltonian.update",
                "Hamiltonian.calc_and_print_energies",
            ],
        }

        category_colors = {
            "Diagonalization": "tab:blue",
            "Initialization": "tab:red",
            "Calc. potential and energy": "tab:pink",
            "ETC": "gray",
        }
    elif args.phase == "preconditioning":
        # Detailed breakdown inside the Preconditioning timer
        category_map = {
            # "ISI. precond": ["ISI. precond"],
            "GAPP": ["ISI. precond"],
            "Hamiltonian op.": ["ISI. (H - e)x"],
            "update search direction": ["ISI. update search direction"],
            "update x0 and r": ["ISI. update x0 and r"],
            "r.T @ z": ["ISI. r.T @ z"],
            "calc alpha": ["ISI. calc alpha"],
            # "ISI. norm": ["ISI. norm"],
        }

        category_colors = {
            # "ISI. precond": "tab:blue",
            "GAPP": "tab:blue",
            "Hamiltonian op.": "tab:red",
            "update search direction": "tab:pink",
            "update x0 and r": "tab:brown",
            "r.T @ z": "tab:green",
            "calc alpha": "tab:purple",
            # "ISI. norm": "tab:orange",
            "ETC": "gray",
        }
    else:
        # Timer categories and labels
        category_map = {
            "Preconditioning": ["Preconditioning"],
            "Hamiltonian op.": [
                "A @ R, B @ R & Redistribution",
                "A @ X, B @ X & Redistribution",
            ],
            "Projection": [
                "Projection (R.H @ AU)",
                "Projection (X.H @ AX)",
                "Projection (Ortho.)",
            ],
            "Rotation": ["Rotation", "Rotation (Ortho.)"],
            "Subspace op.": ["Subspace Diagonalization", "inv & Cholesky (Ortho.)"],
        }

        category_colors = {
            "Preconditioning": "tab:blue",
            "Hamiltonian op.": "tab:red",
            "Projection": "tab:pink",
            "Rotation": "tab:brown",
            "Subspace op.": "tab:green",
            "ETC": "gray",
        }

    # Process each log
    aggregated_results = []
    for method in args.methods:
        if args.phase in ("fixed", "preconditioning"):
            if not args.log_suffix == ".speed.log":
                print("Warning: args.log_suffix={args.log_suffix}")
        else:
            if not args.log_suffix == ".log":
                print("Warning: args.log_suffix={args.log_suffix}")
        log_path = os.path.join(args.log_dir, f"{args.supercell}_{method}{args.log_suffix}")
        raw = parse_log_file(log_path)
        agg = aggregate_timing_by_category(raw, category_map, phase=args.phase)
        aggregated_results.append(agg)

    # Append total_time and acc_fold to the aggregated results
    total_times = [sum(list(result.values())) for result in aggregated_results]
    acc_folds = [total_times[0] / t for t in total_times]
    for i, result in enumerate(aggregated_results):
        result["total_time"] = total_times[i]
        result["acc_fold"] = acc_folds[i]

    if args.phase == "scf":
        scf_steps = []
        for method in args.methods:
            scf_log_path = os.path.join(args.log_dir, f"{args.supercell}_{method}.log")
            with open(scf_log_path, "r") as f:
                for line in f:
                    if "SCF CONVERGED" in line:
                        match = re.search(r"SCF CONVERGED with (\d+) iters", line)
                        if match:
                            scf_steps.append(int(match.group(1)))
                        break

        for i, result in enumerate(aggregated_results):
            result["scf_steps"] = scf_steps[i]

    # Save to JSON
    with open(args.json_out, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"[+] Saved aggregated data to {args.json_out}")

    # Plot
    category_map["ETC"] = None
    plot_accumulated_histogram(
        aggregated_results,
        args.labels,
        list(category_map.keys()),
        category_colors,
        output_file=args.output,
        figsize=tuple(args.figsize),
        bar_width=args.bar_width,
        save_legend_separately=args.separate_legend,
        legend_cols=args.legend_cols,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
