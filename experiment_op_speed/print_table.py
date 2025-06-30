import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any


def parse_timer_summaries(log_content: str) -> pd.DataFrame:
    """
    BaTiO3.log 파일에서 Timer Summary 정보를 파싱하여 DataFrame으로 반환
    """

    # Timer Summary 블록들을 찾기 위한 패턴
    timer_pattern = r"======================== Timer Summary ========================(.*?)(?=========================|$)"
    timer_blocks = re.findall(timer_pattern, log_content, re.DOTALL)

    results = []

    for block in timer_blocks:
        # 각 블록에서 정보 추출
        block_data = {}

        # Operation 정보 추출 (예: "Operation: projection (DP)")
        operation_pattern = r"Operation: (\w+) \((\w+)\)\s*\|\s*([\d.]+)\s*\|\s*(\d+)"
        operations = re.findall(operation_pattern, block)

        # 메타데이터 추출
        supercell_match = re.search(r"supercell:\s*\[([\d,\s]+)\]", block)
        dtype_match = re.search(r"dtype:\s*(\w+)", block)
        dense_proj_match = re.search(r"use_dense_proj:\s*(\w+)", block)
        dense_kinetic_match = re.search(r"use_dense_kinetic:\s*(\w+)", block)

        # supercell에서 마지막 숫자만 추출 (예: [1, 1, 5] -> 5)
        supercell_size = None
        if supercell_match:
            supercell_numbers = [
                int(x.strip()) for x in supercell_match.group(1).split(",")
            ]
            supercell_size = supercell_numbers[-1]  # 마지막 값 사용

        # 각 operation에 대해 레코드 생성
        for op_name, op_dtype, time, count in operations:
            record = {
                "operation": op_name,
                "operation_dtype": op_dtype,
                "time_seconds": float(time),
                "count": int(count),
                "supercell_size": supercell_size,
                "dtype": dtype_match.group(1) if dtype_match else None,
                "use_dense_proj": (
                    dense_proj_match.group(1).lower() == "true"
                    if dense_proj_match
                    else None
                ),
                "use_dense_kinetic": (
                    dense_kinetic_match.group(1).lower() == "true"
                    if dense_kinetic_match
                    else None
                ),
            }
            results.append(record)

    return pd.DataFrame(results)


def create_performance_table(df: pd.DataFrame, use_dense_proj=False):
    """
    원하는 형식의 성능 테이블 생성
    """
    # use_dense_proj 값으로 필터링
    df_filtered = df[df["use_dense_proj"] == use_dense_proj].copy()

    # DP 데이터 분리
    df_dp = df_filtered[df_filtered["dtype"] == "DP"].copy()
    df_others = df_filtered[df_filtered["dtype"] != "DP"].copy()

    # 피벗 테이블 생성 - DP용
    dp_pivot = df_dp.pivot_table(
        values="time_seconds",
        index="operation",
        columns="supercell_size",
        aggfunc="sum",
    )

    # 피벗 테이블 생성 - 다른 dtype용
    other_pivot = df_others.pivot_table(
        values="time_seconds",
        index=["operation", "dtype"],
        columns="supercell_size",
        aggfunc="sum",
    )

    # 결과를 저장할 리스트
    result_rows = []
    result_index = []

    # 각 operation별로 데이터 구성
    for op_name in ["projection", "kinetic", "local", "nonlocal", "tensordot"]:
        if op_name not in dp_pivot.index:
            continue

        dp_times = dp_pivot.loc[op_name]

        # 각 dtype의 가속 비율 추가
        for dtype in ["SP", "TF32", "BF16"]:
            if (op_name, dtype) in other_pivot.index:
                other_times = other_pivot.loc[(op_name, dtype)]
                # DP 시간 / 다른 dtype 시간 = 가속 비율
                acceleration_ratio = dp_times / other_times
                result_rows.append(acceleration_ratio)
                result_index.append((op_name, dtype))

        # 해당 operation의 DP time 추가
        result_rows.append(dp_times)
        result_index.append(("DP time (sec)", ""))

    # DataFrame 생성
    formatted_table = pd.DataFrame(
        result_rows, index=pd.MultiIndex.from_tuples(result_index)
    )

    # 열 이름을 정수로 정렬
    if not formatted_table.empty:
        formatted_table = formatted_table.reindex(
            sorted(formatted_table.columns), axis=1
        )

    return formatted_table, dp_pivot, other_pivot


def print_performance_table(table: pd.DataFrame, title: str = "Performance Comparison"):
    """
    성능 테이블을 보기 좋게 출력
    """
    print(f"\n{'='*120}")
    print(f"{title:^120}")
    print(f"{'='*120}")
    print(f"{'':20} {'Supercell Size':^100}")
    print(f"{'Operation':<12} {'Precision':<12}", end="")

    # 열 헤더 출력
    for col in table.columns:
        print(f"{col:>10}", end="")
    print()
    print("-" * 120)

    # 데이터 출력
    current_op = None
    for (op, dtype), row in table.iterrows():
        if op == "DP time (sec)":
            print("-" * 120)
            print(f"{op:<24}", end=" ")
            for val in row:
                if pd.notna(val):
                    print(f"{val:>10.6f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()
            print("-" * 120)
        else:
            if current_op != op:
                current_op = op
                print(f"{op:<12} {dtype:<12}", end="")
            else:
                print(f"{'':12} {dtype:<12}", end="")

            for val in row:
                if pd.notna(val):
                    print(f"{val:>10.2f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()

    print("(Acc. ratio with respect to DP)")
    print("=" * 120)


def analyze_performance_trends(df: pd.DataFrame):
    """
    성능 트렌드 분석
    """
    print("\n=== 성능 트렌드 분석 ===")

    # use_dense_proj별로 분석
    for use_dense in [False, True]:
        df_filtered = df[df["use_dense_proj"] == use_dense]
        if df_filtered.empty:
            continue

        print(f"\n--- use_dense_proj = {use_dense} ---")

        # 각 dtype별 평균 성능
        for dtype in ["SP", "TF32", "BF16"]:
            dtype_data = df_filtered[df_filtered["dtype"] == dtype]
            if not dtype_data.empty:
                avg_time = dtype_data.groupby("operation")["time_seconds"].mean()
                print(f"\n{dtype} 평균 실행 시간:")
                for op, time in avg_time.items():
                    print(f"  {op}: {time:.6f}s")


# 사용 예시
if __name__ == "__main__":
    # 파일 읽기
    import sys

    filename = sys.argv[-1]

    # with open('../BaTiO3.log', 'r') as f:
    with open(filename, "r") as f:
        log_content = f.read()

    # 파싱
    df = parse_timer_summaries(log_content)

    # 데이터 요약
    print("=== 파싱된 데이터 요약 ===")
    print(f"총 레코드 수: {len(df)}")
    print(f"고유 operation: {df['operation'].unique()}")
    print(f"고유 dtype: {df['dtype'].unique()}")
    print(f"고유 supercell 크기: {sorted(df['supercell_size'].unique())}")

    # use_dense_proj=False인 경우의 테이블 생성
    try:
        table_false, dp_pivot_false, other_pivot_false = create_performance_table(
            df, use_dense_proj=False
        )
        if not table_false.empty:
            print_performance_table(
                table_false, "Performance Comparison (use_dense_proj=False)"
            )
            filename = "performance_dense_proj_false.csv"
            table_false.to_csv(filename)
            print(f"Save {filename}")

            # 평균 가속 비율 계산
            print("\n=== 평균 가속 비율 (use_dense_proj=False) ===")
            for dtype in ["SP", "TF32", "BF16"]:
                dtype_values = []
                for idx, row in table_false.iterrows():
                    if len(idx) > 1 and idx[1] == dtype:
                        dtype_values.extend([v for v in row if pd.notna(v)])
                if dtype_values:
                    print(f"{dtype}: 평균 {np.mean(dtype_values):.2f}x 가속")
    except Exception as e:
        print(f"use_dense_proj=False 테이블 생성 중 오류: {e}")

    # use_dense_proj=True인 경우의 테이블 생성
    try:
        table_true, dp_pivot_true, other_pivot_true = create_performance_table(
            df, use_dense_proj=True
        )
        if not table_true.empty:
            print_performance_table(
                table_true, "Performance Comparison (use_dense_proj=True)"
            )
            filename = "performance_dense_proj_true.csv"
            table_true.to_csv(filename)
            print(f"Save {filename}")

            # 평균 가속 비율 계산
            print("\n=== 평균 가속 비율 (use_dense_proj=True) ===")
            for dtype in ["SP", "TF32", "BF16"]:
                dtype_values = []
                for idx, row in table_true.iterrows():
                    if len(idx) > 1 and idx[1] == dtype:
                        dtype_values.extend([v for v in row if pd.notna(v)])
                if dtype_values:
                    print(f"{dtype}: 평균 {np.mean(dtype_values):.2f}x 가속")
    except Exception as e:
        print(f"use_dense_proj=True 테이블 생성 중 오류: {e}")

    print("\n결과가 CSV 파일로 저장되었습니다.")

    # 추가 분석
    analyze_performance_trends(df)
