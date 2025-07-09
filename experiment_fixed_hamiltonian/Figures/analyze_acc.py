import re
import csv
from collections import defaultdict


def parse_log_file(filepath):
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


def print_results(results, baseline_data, all_labels):
    # 정렬 기준: baseline 시간의 내림차순 (없으면 0으로 처리)
    sorted_labels = sorted(all_labels, key=lambda x: -baseline_data.get(x, 0))

    for method, timings in results.items():
        print(f"\n--- Method: {method} ---")
        for label in sorted_labels:
            if label in timings:
                current_time = timings[label]
                base_time = baseline_data.get(label, None)
                if base_time:
                    speedup = base_time / current_time
                    print(
                        f"{label:<50} : {base_time:>8.4f} sec -> {current_time:>8.4f} sec ({speedup:.2f}x)"
                    )
                else:
                    print(
                        f"{label:<50} : No baseline -> {current_time:>8.4f} sec (N/A)"
                    )


def save_to_csv(results, baseline_data, all_labels, output_file):
    """결과를 CSV 파일로 저장"""
    # 정렬 기준: baseline 시간의 내림차순 (없으면 0으로 처리)
    sorted_labels = sorted(all_labels, key=lambda x: -baseline_data.get(x, 0))

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        # 헤더 생성
        fieldnames = ["Label", "Baseline_Time"]
        for method in sorted(results.keys()):
            fieldnames.extend([f"{method}_Time", f"{method}_Speedup"])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 데이터 행 작성
        for label in sorted_labels:
            base_time = baseline_data.get(label, None)
            row = {"Label": label, "Baseline_Time": f"{base_time:.4f}" if base_time else "N/A"}

            for method in sorted(results.keys()):
                if label in results[method]:
                    current_time = results[method][label]
                    row[f"{method}_Time"] = f"{current_time:.4f}"
                    
                    if base_time and base_time > 0:
                        speedup = base_time / current_time
                        row[f"{method}_Speedup"] = f"{speedup:.2f}"
                    else:
                        row[f"{method}_Speedup"] = "N/A"
                else:
                    row[f"{method}_Time"] = "N/A"
                    row[f"{method}_Speedup"] = "N/A"

            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def save_detailed_csv(results, baseline_data, all_labels, output_file):
    """더 상세한 형식의 CSV 파일로 저장 (각 method별 행)"""
    sorted_labels = sorted(all_labels, key=lambda x: -baseline_data.get(x, 0))

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Method",
            "Label",
            "Baseline_Time",
            "Current_Time",
            "Speedup",
            "Time_Reduction_%",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for method in sorted(results.keys()):
            for label in sorted_labels:
                if label in results[method]:
                    current_time = results[method][label]
                    base_time = baseline_data.get(label, None)
                    
                    if base_time and base_time > 0:
                        speedup = base_time / current_time
                        time_reduction = ((base_time - current_time) / base_time) * 100
                        baseline_str = f"{base_time:.4f}"
                        speedup_str = f"{speedup:.2f}"
                        reduction_str = f"{time_reduction:.1f}"
                    else:
                        baseline_str = "N/A"
                        speedup_str = "N/A"
                        reduction_str = "N/A"

                    writer.writerow(
                        {
                            "Method": method,
                            "Label": label,
                            "Baseline_Time": baseline_str,
                            "Current_Time": f"{current_time:.4f}",
                            "Speedup": speedup_str,
                            "Time_Reduction_%": reduction_str,
                        }
                    )


def compute_speedups(reference_log, probed_logs):
    baseline_data = parse_log_file(reference_log)
    results = defaultdict(dict)
    all_labels = set(baseline_data.keys())  # 모든 label 수집

    for filepath in probed_logs:
        method = filepath.split("/")[-1].replace(".log", "")  # 파일명에서 method 추출
        current_data = parse_log_file(filepath)
        
        # prb_log의 모든 label도 추가
        all_labels.update(current_data.keys())
        
        # 모든 current_data의 값들을 저장 (baseline 존재 여부와 관계없이)
        for label, current_time in current_data.items():
            results[method][label] = current_time

    return results, baseline_data, all_labels


if __name__ == "__main__":
    """
    Example usage:
    python analyze_acc.py --ref_log ./DP.log --prb_log ./SP.log ./MP1.log
    python analyze_acc.py --ref_log ./DP.log --prb_log ./SP.log ./MP1.log --csv results.csv
    python analyze_acc.py --ref_log ./DP.log --prb_log ./SP.log ./MP1.log --csv results.csv --detailed
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare performance logs against a reference log"
    )
    parser.add_argument(
        "--ref_log", type=str, required=True, help="Reference log file (e.g., DP.log)"
    )
    parser.add_argument(
        "--prb_log",
        type=str,
        nargs="+",
        required=True,
        help="Log files to compare against reference",
    )
    parser.add_argument("--csv", type=str, help="Output CSV file name (optional)")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Save detailed CSV format with one row per method-label pair",
    )
    args = parser.parse_args()

    results, baseline_data, all_labels = compute_speedups(args.ref_log, args.prb_log)
    print_results(results, baseline_data, all_labels)

    # CSV 저장
    if args.csv:
        if args.detailed:
            # 상세 형식 저장
            detailed_filename = args.csv.replace(".csv", "_detailed.csv")
            save_detailed_csv(results, baseline_data, all_labels, detailed_filename)
            print("Saved detailed results to:", detailed_filename)
        else:
            # 기본 형식 저장
            save_to_csv(results, baseline_data, all_labels, args.csv)
            print("Saved results to:", args.csv)
