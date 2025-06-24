import re
import glob
from collections import defaultdict


def parse_log_file(filepath):
    in_summary = False
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '======================== Timer Summary ========================' in line:
                in_summary = True
            elif in_summary:
                if re.match(r'-{20,}', line):
                    continue
                elif 'Elapsed time' in line or line.strip() == '':
                    break
                else:
                    match = re.match(r'^(.*?)\s+\|\s+([\d\.]+)\s+\|\s+(\d+)', line)
                    if match:
                        label = match.group(1).strip()
                        total = float(match.group(2))
                        data[label] = total
    return data


def print_results(results, baseline_data):
    # 정렬 기준: baseline 시간의 내림차순
    sorted_labels = sorted(baseline_data.items(), key=lambda x: -x[1])

    for method, timings in results.items():
        print(f"\n--- Method: {method} ---")
        for label, base_time in sorted_labels:
            if label in timings:
                current_time = timings[label]
                speedup = base_time / current_time
                print(f"{label:<50} : {base_time:>8.4f} sec -> {current_time:>8.4f} sec ({speedup:.2f}x)")


def compute_speedups(reference_log, probed_logs):
    baseline_data = parse_log_file(reference_log)
    results = defaultdict(dict)

    for filepath in probed_logs:
        method = filepath.split('/')[-1].replace(".log", "")  # 파일명에서 method 추출
        current_data = parse_log_file(filepath)

        for label in baseline_data:
            base_time = baseline_data.get(label)
            current_time = current_data.get(label)
            if base_time > 0 and current_time is not None:
                results[method][label] = current_time

    return results, baseline_data


if __name__ == "__main__":
    """
    Example usage:
    python analyze_acc.py --ref_log ./DP.log --prb_log ./SP.log ./MP1.log
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compare performance logs against a reference log")
    parser.add_argument('--ref_log', type=str, required=True, help="Reference log file (e.g., DP.log)")
    parser.add_argument('--prb_log', type=str, nargs='+', required=True, help="Log files to compare against reference")
    args = parser.parse_args()

    results, baseline_data = compute_speedups(args.ref_log, args.prb_log)
    print_results(results, baseline_data)


# def compute_speedups(log_dir, supercell):
#     prefix = "_".join(map(str, supercell))
# 
#     baseline_path = f"{log_dir}/{prefix}_DP.speed.log"
#     baseline_data = parse_log_file(baseline_path)
# 
#     results = defaultdict(dict)
# 
#     for filepath in glob.glob(f"{log_dir}/{prefix}_*.speed.log"):
#         if "recalc_convg_history" in filepath:
#             continue
# 
#         method = filepath.split(f"{prefix}_")[1].replace(".speed.log", "")
#         if method == "DP":
#             continue
# 
#         current_data = parse_log_file(filepath)
# 
#         for label in baseline_data:
#             base_time = baseline_data.get(label)
#             current_time = current_data.get(label)
#             if base_time > 0 and current_time is not None:
#                 results[method][label] = current_time  # 시간 자체 저장
# 
#     return results, baseline_data
# 
# 
# if __name__ == "__main__":
#     """
#     e.g.,
#     python analyze_acc.py ../expt.CNT_6_0.A100.dense --supercell 1 1 10
#     """
#     import argparse
# 
#     # make log_dir as an argument
#     parser = argparse.ArgumentParser(description='Compute speedups for different methods')
#     parser.add_argument('log_dir', type=str, help='Directory containing log files')
#     parser.add_argument(
#         "--supercell", type=int, nargs="+", default=[1, 1, 1], help="supercell"
#     )
#     args = parser.parse_args()
# 
#     results, baseline_data = compute_speedups(args.log_dir, supercell=args.supercell)
#     print_results(results, baseline_data)

