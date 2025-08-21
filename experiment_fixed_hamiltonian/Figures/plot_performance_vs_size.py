"""
Performance enhancement of MP1*.

x-axis: the number of atoms
y-axis: speedups
legend: different GPUs
subfigure: systems
"""

import matplotlib.pyplot as plt
import pandas as pd


# natoms info
natoms_dict = {
    "CNT_6_0": 24,
    "MgO_1x1x2": 16,
    "Si_diamond_2x2x1": 32,
}

# 전역 폰트 설정 변수
CURRENT_FONT_CONFIG = {
    "base": 14,
    "title": 18.2,
    "label": 16.1,
    "tick": 14,
    "legend": 11.9,
}

# 색상 팔레트 정의
gpu_colors = {
    "A100": "#1f77b4",  # blue
    "L40S": "#ff7f0e",  # orange
    "A6000": "#2ca02c",  # green
}

system_colors = {
    "CNT_6_0": "#e377c2",  # pink
    "MgO_1x1x2": "#17becf",  # cyan
    "Si_diamond_2x2x1": "#bcbd22",  # yellow-green
}


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
        "subtitle": 1.1,  # 부제목은 110%
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
            # 추가 설정
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": True,
            # 선 두께 설정 (폰트 크기에 비례)
            "lines.linewidth": base_font_size / 10,
            # "lines.markersize": base_font_size * 0.6,
            "lines.markersize": base_font_size * 0.4,
            # 그리드 설정
            "grid.linewidth": base_font_size / 20,
            "grid.alpha": 0.3,
        }
    )

    # 전역 변수로 현재 폰트 설정 저장 (그래프 함수에서 사용)
    global CURRENT_FONT_CONFIG
    CURRENT_FONT_CONFIG = {
        "base": base_font_size,
        "title": base_font_size * scale_factors["title"],
        "label": base_font_size * scale_factors["label"],
        "tick": base_font_size * scale_factors["tick"],
        "legend": base_font_size * scale_factors["legend"],
    }


def get_results_dict(
    gpu,
    system_name,
    method,
    supercell_list=range(1, 11),
    target="speedup",  # options=["speedup", "baseline_time", "time"]
):
    results_dict = {}
    for supercell in supercell_list:
        filename = f"csv_results_{gpu}/{system_name}.1_1_{supercell}.csv"

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
                raise ValueError(
                    "Invalid target specified. Use 'speedup', 'baseline_time', or 'time'."
                )

            value = data[label].iloc[0]
            natoms = supercell * natoms_dict[system_name]
            results_dict[natoms] = value

        except FileNotFoundError:
            print(f"Error: File {filename} not found")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results_dict


def create_combined_dataframe(gpu_list, system_name_list, supercell_list, method):
    """모든 결과를 하나의 DataFrame으로 결합"""
    all_results = []

    for gpu in gpu_list:
        for system_name in system_name_list:
            # 각 target에 대한 결과 가져오기
            speedup_dict = get_results_dict(
                gpu, system_name, method, supercell_list, target="speedup"
            )
            baseline_dict = get_results_dict(
                gpu, system_name, method, supercell_list, target="baseline_time"
            )
            time_dict = get_results_dict(
                gpu, system_name, method, supercell_list, target="time"
            )

            # 각 supercell에 대한 결과를 행으로 추가
            for supercell in supercell_list:
                natoms = supercell * natoms_dict[system_name]

                if natoms in speedup_dict:  # 데이터가 있는 경우만 추가
                    row = {
                        "GPU": gpu,
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

    # DataFrame 생성
    df = pd.DataFrame(all_results)

    # 정렬 (GPU -> System -> Supercell 순서로)
    df = df.sort_values(["GPU", "System", "Supercell"]).reset_index(drop=True)

    return df


# System 이름을 짧게 변환하는 함수
def get_short_system_name(system_name):
    mapping = {"CNT_6_0": "CNT", "MgO_1x1x2": "MgO", "Si_diamond_2x2x1": "Si"}
    return mapping.get(system_name, system_name)


def plot_type1_speedup_by_gpu(
    df, system_name, save_path=None, custom_font_config=None, figsize=(9, 6)
):
    """
    Type 1: 하나의 System에 대한 그림
    x-axis: natoms, y-axis: Speedup, legend: GPU 종류

    Args:
        df: DataFrame with performance data
        system_name: Name of the system to plot
        save_path: Path to save the figure
        custom_font_config: Custom font configuration dict (optional)
    """
    # 폰트 설정 사용
    font_config = custom_font_config or CURRENT_FONT_CONFIG

    fig, ax = plt.subplots(figsize=figsize)

    # 선택된 시스템 데이터만 필터링
    system_data = df[df["System"] == system_name].copy()

    # GPU별로 플롯
    for gpu in system_data["GPU"].unique():
        gpu_data = system_data[system_data["GPU"] == gpu].sort_values("Natoms")
        # NaN 값 제거
        gpu_data = gpu_data.dropna(subset=["Speedup"])

        ax.plot(
            gpu_data["Natoms"],
            gpu_data["Speedup"],
            marker="o",
            # markersize=font_config['base']*0.6,
            # linewidth=font_config['base']/5,
            label=gpu,
            color=gpu_colors.get(gpu, "gray"),
        )

    # 그래프 꾸미기
    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Speedup", fontsize=font_config["label"])
    # ax.set_title(f'Speedup vs System Size - {get_short_system_name(system_name)}',
    #              fontsize=font_config['title'], fontweight='bold')

    # y=1 참조선 추가
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.legend(loc="best", frameon=True, shadow=False, fontsize=font_config["legend"])

    # 틱 레이블 크기 설정
    ax.tick_params(axis="both", labelsize=font_config["tick"])

    # y축 범위 설정 (최소값 0.5)
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_type2_time_comparison(
    df, system_name, save_path=None, legend_path=None,
    custom_font_config=None, figsize=(9, 6)
):
    """
    Type 2: 하나의 System에 대한 그림
    x-axis: natoms, y-axis: Time, legend: GPU 및 method (DP, MP1*)

    Args:
        df: DataFrame with performance data
        system_name: Name of the system to plot
        save_path: Path to save the figure
        legend_path: Path to save the legend box
        custom_font_config: Custom font configuration dict (optional)
    """
    font_config = custom_font_config or CURRENT_FONT_CONFIG

    fig, ax = plt.subplots(figsize=figsize)

    system_data = df[df["System"] == system_name].copy()
    line_styles = {"DP": "-", "MP1*": "--"}

    for gpu in system_data["GPU"].unique():
        gpu_data = system_data[system_data["GPU"] == gpu].sort_values("Natoms")

        dp_data = gpu_data.dropna(subset=["Baseline_Time"])
        if not dp_data.empty:
            ax.plot(
                dp_data["Natoms"], dp_data["Baseline_Time"],
                marker="o", linestyle=line_styles["DP"],
                label=f"DP ({gpu})",
                color=gpu_colors.get(gpu, "gray"),
            )

        mp1_data = gpu_data.dropna(subset=["Method_Time"])
        if not mp1_data.empty:
            ax.plot(
                mp1_data["Natoms"], mp1_data["Method_Time"],
                marker="s", linestyle=line_styles["MP1*"],
                label=f"MP1* ({gpu})",
                color=gpu_colors.get(gpu, "gray"),
            )

    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Diag. time (sec)", fontsize=font_config["label"])
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.tick_params(axis="both", labelsize=font_config["tick"])

    # 원래 Figure에서는 legend를 그리지 않음
    handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # 범례만 별도 Figure로 저장
    if legend_path:
        fig_leg = plt.figure(figsize=(4, 2))
        fig_leg.legend(
            handles, labels, loc="center", frameon=True, ncol=2,
            fontsize=font_config["legend"], handlelength=3.0, handleheight=1.0
        )
        fig_leg.tight_layout()
        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)


def plot_type3_speedup_by_system(
    df, gpu_name, save_path=None, custom_font_config=None, figsize=(9, 6)
):
    """
    Type 3: 하나의 GPU에 대한 그림
    x-axis: natoms, y-axis: Speedup, legend: system 종류

    Args:
        df: DataFrame with performance data
        gpu_name: Name of the GPU to plot
        save_path: Path to save the figure
        custom_font_config: Custom font configuration dict (optional)
    """
    # 폰트 설정 사용
    font_config = custom_font_config or CURRENT_FONT_CONFIG

    fig, ax = plt.subplots(figsize=figsize)

    # 선택된 GPU 데이터만 필터링
    gpu_data = df[df["GPU"] == gpu_name].copy()

    # System별로 플롯
    for system in gpu_data["System"].unique():
        system_df = gpu_data[gpu_data["System"] == system].sort_values("Natoms")
        # NaN 값 제거
        system_df = system_df.dropna(subset=["Speedup"])

        ax.plot(
            system_df["Natoms"],
            system_df["Speedup"],
            marker="o",
            markersize=font_config["base"] * 0.6,
            linewidth=font_config["base"] / 5,
            label=get_short_system_name(system),
            color=system_colors.get(system, "gray"),
        )

    # 그래프 꾸미기
    ax.set_xlabel("Number of atoms", fontsize=font_config["label"])
    ax.set_ylabel("Speedup", fontsize=font_config["label"])
    # ax.set_title(f'Speedup vs System Size - {gpu_name}',
    #              fontsize=font_config['title'], fontweight='bold')

    # y=1 참조선 추가
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=font_config["base"] / 20)
    ax.legend(loc="best", frameon=True, shadow=False, fontsize=font_config["legend"])

    # 틱 레이블 크기 설정
    ax.tick_params(axis="both", labelsize=font_config["tick"])

    # y축 범위 설정 (최소값 0.5)
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 기본 스타일 설정 적용
    setup_plot_style(base_font_size=28)
    figsize = (10, 7)

    system_name_list = ["CNT_6_0", "MgO_1x1x2", "Si_diamond_2x2x1"]
    supercell_list = list(range(1, 20))
    method = "MP_scheme1_BF164precond"
    gpu_list = ["A100", "A6000", "L40S"]

    # 전체 결과를 하나의 DataFrame으로 생성
    combined_df = create_combined_dataframe(
        gpu_list, system_name_list, supercell_list, method
    )
    print(combined_df)

    # Plot figure
    # 개별 플롯 생성 예시
    for system in ["CNT_6_0", "MgO_1x1x2", "Si_diamond_2x2x1"]:
        # Type 1: Comparions of the speedups relative to DP calculation
        filename = f"./Figures_performance_vs_size/type1_speedup_{system}.svg"
        plot_type1_speedup_by_gpu(combined_df, system, filename, figsize=figsize)
        print(f"Save plot to {filename}")

        # Type 2: Diag. time comparison as a function of the number of atoms
        filename = f"./Figures_performance_vs_size/type2_time_{system}.svg"
        filename_legend = "./Figures_performance_vs_size/legend_box.svg"
        # plot_type2_time_comparison(combined_df, system, filename, figsize=figsize)
        plot_type2_time_comparison(combined_df, system, filename, legend_path=filename_legend, figsize=figsize)
        print(f"Save legend box to {filename_legend}")
        print(f"Save plot to {filename}")

    # # Type 3: A100 GPU의 시스템별 Speedup
    # plot_type3_speedup_by_system(combined_df, 'A100', 'type3_a100_speedup.svg', figsize=figsize)
