#!/usr/bin/env python3
import re
import glob
import pandas as pd
from pathlib import Path

def extract_peak_alloc(file_path):
    """
    파일에서 'Fixed davidson diagonalization' 라인의 peak_alloc 값을 추출
    
    Args:
        file_path: 로그 파일 경로
    
    Returns:
        float: peak_alloc 값 (GiB), 없으면 None
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Fixed davidson diagonalization' in line:
                    # peak_alloc=XX.XXGiB 패턴 매칭
                    match = re.search(r'peak_alloc=([0-9.]+)GiB', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def parse_filename(filename):
    """
    파일명에서 n과 opt_name 추출
    패턴: 1_1_{n}_{opt_name}.speed.log
    
    Args:
        filename: 파일명
    
    Returns:
        tuple: (n, opt_name) 또는 (None, None)
    """
    match = re.match(r'1_1_(\d+)_(.+)\.speed\.log', filename)
    if match:
        n = int(match.group(1))
        opt_name = match.group(2)
        return n, opt_name
    return None, None

def collect_peak_alloc_data(base_dir='..'):
    """
    모든 시스템의 peak_alloc 데이터 수집
    
    Args:
        base_dir: 기본 디렉토리 (expt.* 폴더들이 있는 위치)
    
    Returns:
        pandas.DataFrame: 수집된 데이터
    """
    systems = ["CNT_6_0", "MgO_1x1x2", "Si_diamond_2x2x1"]
    
    data = []
    
    for system in systems:
        # 패턴에 맞는 모든 파일 찾기
        pattern = f"{base_dir}/expt.{system}/1_1_*_*.speed.log"
        files = glob.glob(pattern)
        
        for file_path in files:
            filename = Path(file_path).name
            n, opt_name = parse_filename(filename)
            
            if n is not None and opt_name is not None:
                peak_alloc = extract_peak_alloc(file_path)
                
                if peak_alloc is not None:
                    data.append({
                        'System': system,
                        'n': n,
                        'opt_name': opt_name,
                        'peak_alloc_GiB': peak_alloc
                    })
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 정렬: System, n, opt_name 순서로
    if not df.empty:
        df = df.sort_values(['System', 'n', 'opt_name']).reset_index(drop=True)
    
    return df

def main():
    """
    메인 함수: 데이터 수집, 출력, 저장
    """
    print("Collecting peak_alloc data from log files...")
    
    # 데이터 수집
    df = collect_peak_alloc_data()
    
    if df.empty:
        print("No data found!")
        return
    
    # 테이블 형식으로 출력
    print("\n" + "="*80)
    print("Peak Allocation Data")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 통계 정보 출력
    print(f"\nTotal records: {len(df)}")
    print(f"Systems: {df['System'].nunique()}")
    print(f"Unique n values: {sorted(df['n'].unique())}")
    print(f"Unique opt_names: {sorted(df['opt_name'].unique())}")
    
    # CSV 파일로 저장
    output_file = 'peak_alloc_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    df = main()
