#!/usr/bin/env bash
# set -euo pipefail

cd ../..  # Go to the 최상위 directory

# ---------------- User config ----------------
# 대상 시스템
SYSTEMS=("CNT_6_0" "MgO_1x1x2" "Si_diamond_2x2x1")

# 시스템별 nocc (필수: 키를 모든 SYSTEMS에 대해 지정)
declare -A NOCC_MAP=(
  ["CNT_6_0"]=48
  ["MgO_1x1x2"]=64
  ["Si_diamond_2x2x1"]=64
)

# 시스템별 supercell_sizes (공백으로 구분된 문자열로 저장)
# 예: "1 5 10" 또는 "1 4 7"
declare -A SUPERCELL_MAP=(
  ["CNT_6_0"]="1 5 10"
  ["MgO_1x1x2"]="1 5 10"
  ["Si_diamond_2x2x1"]="1 4 7"
)

# 공통 저장 디렉토리
save_root=./experiment_fixed_hamiltonian
save_fig_dir=${save_root}/Figures/Figures_convg_history_SI
mkdir -p "${save_fig_dir}"

option_names=(
  "DP.recalc_convg_history" "SP.recalc_convg_history" "MP_scheme1.recalc_convg_history" "MP_scheme2.recalc_convg_history" "MP_scheme3.recalc_convg_history" "MP_scheme4.recalc_convg_history" "MP_scheme5.recalc_convg_history" "DP_SP4precond.recalc_convg_history"
  "TF32.recalc_convg_history" "MP_scheme1_TF32.recalc_convg_history" "MP_scheme2_TF32.recalc_convg_history" "MP_scheme3_TF32.recalc_convg_history" "MP_scheme4_TF32.recalc_convg_history" "MP_scheme5_TF32.recalc_convg_history" "DP_TF324precond.recalc_convg_history"
  "MP_scheme1_BF164precond.recalc_convg_history" "MP_scheme1_BF16.recalc_convg_history" "MP_scheme2_BF16.recalc_convg_history" "MP_scheme3_BF16.recalc_convg_history" "MP_scheme4_BF16.recalc_convg_history" "MP_scheme5_BF16.recalc_convg_history" "DP_BF164precond.recalc_convg_history"
)

title_names=(
  "FP64" "FP32" "MP1(FP32)" "MP2(FP32)" "MP3(FP32)" "MP4(FP32)" "MP5(FP32)" "MP6(FP32)"
         "TF32" "MP1(TF32)" "MP2(TF32)" "MP3(TF32)" "MP4(TF32)" "MP5(TF32)" "MP6(TF32)"
  "MP1*"        "MP1(BF16)" "MP2(BF16)" "MP3(BF16)" "MP4(BF16)" "MP5(BF16)" "MP6(BF16)"
)
# ---------------------------------------------

# (선택) 배열 길이 sanity check
if [[ ${#option_names[@]} -ne ${#title_names[@]} ]]; then
  echo "[ERROR] option_names (${#option_names[@]}) != title_names (${#title_names[@]})" >&2
  exit 1
fi

# Plotting (system loop)
for system_name in "${SYSTEMS[@]}"; do
  echo "================ System: ${system_name} ================"

  # nocc 조회 (set -u 안전하게)
  if [[ -z ${NOCC_MAP[$system_name]+_} ]]; then
    echo "[ERROR] NOCC_MAP missing key '${system_name}'" >&2
    exit 1
  fi
  nocc=${NOCC_MAP[$system_name]}

  # supercell_sizes 조회
  if [[ -z ${SUPERCELL_MAP[$system_name]+_} ]]; then
    echo "[ERROR] SUPERCELL_MAP missing key '${system_name}'" >&2
    exit 1
  fi
  read -r -a supercell_sizes <<< "${SUPERCELL_MAP[$system_name]}"

  save_dir=${save_root}/expt.${system_name}

  for n in "${supercell_sizes[@]}"; do
    cell_name="1_1_${n}"
    num_eig=$(( nocc * n ))
    echo "Processing supercell size: ${cell_name} (num_eig = ${num_eig})"

    for idx in "${!option_names[@]}"; do
      opt_name="${option_names[$idx]}"
      title_name="${title_names[$idx]}"

      prefix="${save_dir}/${cell_name}_${opt_name}"
      retHistory_file="${prefix}.pt"
      refHistory_file="${save_dir}/${cell_name}_DP_ref.pt"

      if [[ ! -f "${retHistory_file}" ]]; then
        echo "  [WARN] Missing history file: ${retHistory_file} (skip ${opt_name})"
        continue
      fi

      echo "  Option: ${opt_name}"
      echo "    Loading file: ${retHistory_file}"

      # residual
      echo "    Plotting residual convergence history..."
      python plot_convg_history.py --convg_tol 1e-7 --filepath "${retHistory_file}" \
        --num_eig "${num_eig}" --plot residual \
        --title "${title_name}" \
        --save "${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.residual.svg"
      echo "    -> Saved residual plot to ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.residual.svg"

      # eigval
      if [[ -f "${refHistory_file}" ]]; then
        ref_flag=(--ref_filepath "${refHistory_file}")
      else
        echo "    [WARN] Missing ref file for eigval: ${refHistory_file} (use without ref)"
        ref_flag=()
      fi

      echo "    Plotting eigenvalue convergence history..."
      python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
        "${ref_flag[@]}" \
        --num_eig "${num_eig}" --plot eigval \
        --title "${title_name}" \
        --save "${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.eigval.svg"
      echo "    -> Saved eigenvalue plot to ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.eigval.svg"
    done

    echo "Finished processing ${system_name} / ${cell_name}."
  done
done

echo "All plotting complete."
cd -  # Back to original directory
