#!/usr/bin/env bash
# set -euo pipefail   # 필요시 해제하세요 (엄격 모드)

# ===================== User config =====================
SYSTEMS=("CNT_6_0" "MgO_1x1x2" "Si_diamond_2x2x1")

# 시스템별 supercell sizes (공백으로 구분된 문자열)
declare -A SUPERCELL_MAP=(
  ["CNT_6_0"]="1 5 10"
  ["MgO_1x1x2"]="1 5 10"
  ["Si_diamond_2x2x1"]="1 4 7"
)

# Figures 디렉토리 (모든 시스템 공통 경로)
FIG_DIR="./Figures_convg_history_SI"

# 병합 결과 저장 경로
OUT_DIR="./Figures_convg_history_SI"
mkdir -p "${OUT_DIR}"

# merge_svg.py 실행 파일 경로
MERGER="merge_svg.py"

# 그리드/레이아웃 설정
ROWS=3
COLS=8
EMPTY_CELLS=("1,0" "2,1")
CELL_W=150
CELL_H=150
GUTTER_X=-40
GUTTER_Y=-25
MARGIN_L=0
MARGIN_T=0
MARGIN_R=0
MARGIN_B=0

# PNG도 같이 뽑을지 여부
DO_PNG=false
PNG_DPI=300
# ======================================================

# === 중요: 파일 순서를 option_names 순서대로 유지 ===
option_names=(
  "DP.recalc_convg_history" "SP.recalc_convg_history" "MP_scheme1.recalc_convg_history" "MP_scheme2.recalc_convg_history" "MP_scheme3.recalc_convg_history" "MP_scheme4.recalc_convg_history" "MP_scheme5.recalc_convg_history" "DP_SP4precond.recalc_convg_history"
  "TF32.recalc_convg_history" "MP_scheme1_TF32.recalc_convg_history" "MP_scheme2_TF32.recalc_convg_history" "MP_scheme3_TF32.recalc_convg_history" "MP_scheme4_TF32.recalc_convg_history" "MP_scheme5_TF32.recalc_convg_history" "DP_TF324precond.recalc_convg_history"
  "MP_scheme1_BF164precond.recalc_convg_history" "MP_scheme1_BF16.recalc_convg_history" "MP_scheme2_BF16.recalc_convg_history" "MP_scheme3_BF16.recalc_convg_history" "MP_scheme4_BF16.recalc_convg_history" "MP_scheme5_BF16.recalc_convg_history" "DP_BF164precond.recalc_convg_history"
)

# kinds: eigval / residual
KINDS=("eigval" "residual")

# ---- helper: 파일 리스트를 option_names 순서대로 생성 ----
# 새 규칙 반영: 파일명 = <FIG_DIR>/<system>.<cell>_<opt>.<kind>.svg
# 예: CNT_6_0.1_1_1_MP_scheme1.recalc_convg_history.eigval.svg
build_file_list() {
  local system="$1"   # 예: CNT_6_0
  local cell="$2"     # 예: 1_1_1
  local kind="$3"     # eigval | residual
  local files=()

  for opt in "${option_names[@]}"; do
    local f="${FIG_DIR}/${system}.${cell}_${opt}.${kind}.svg"
    if [[ -f "$f" ]]; then
      files+=("$f")
    else
      echo "[WARN] missing: $f" >&2
    fi
  done

  printf '%s\n' "${files[@]}"
}

# ---- helper: merge 실행 ----
run_merge() {
  local system="$1"
  local n="$2"
  local kind="$3"

  local cell="1_1_${n}"

  # 파일 리스트 생성 (option_names 순서 보장)
  mapfile -t FILES < <(build_file_list "$system" "$cell" "$kind")

  local count="${#FILES[@]}"
  if (( count == 0 )); then
    echo "[SKIP] No input files for system=${system}, cell=${cell}, kind=${kind}"
    return
  fi

  # 출력 파일 경로
  local out_svg="${OUT_DIR}/merged.${system}.${cell}.${kind}.svg"
  # PNG 출력 경로 (옵션)
  local png_args=()
  if [[ "${DO_PNG}" == true ]]; then
    local out_png="${OUT_DIR}/merged.${system}.${cell}.${kind}.png"
    png_args=(--png "${out_png}" --dpi "${PNG_DPI}")
  fi

  echo "[INFO] Merging (${system}, ${cell}, ${kind}) : ${count} files"
  # merge_svg.py 호출 (기본: cover + clip)
  python "${MERGER}" \
    "${FILES[@]}" \
    --rows "${ROWS}" --cols "${COLS}" \
    $(for e in "${EMPTY_CELLS[@]}"; do printf -- "--empty %s " "$e"; done) \
    --cell-size "${CELL_W}" "${CELL_H}" \
    --gutter "${GUTTER_X}" "${GUTTER_Y}" \
    --margin "${MARGIN_L}" "${MARGIN_T}" "${MARGIN_R}" "${MARGIN_B}" \
    --output "${out_svg}" \
    "${png_args[@]}"
}

# ===================== main loop =====================
for sys in "${SYSTEMS[@]}"; do
  echo "===== System: ${sys} ====="

  # 시스템별 supercell sizes 가져오기 (문자열 → 배열)
  if [[ -z ${SUPERCELL_MAP[$sys]+_} ]]; then
    echo "[ERROR] SUPERCELL_MAP missing key '${sys}'" >&2
    exit 1
  fi
  read -r -a SUPERCELL_SIZES <<< "${SUPERCELL_MAP[$sys]}"

  for n in "${SUPERCELL_SIZES[@]}"; do
    for kind in "${KINDS[@]}"; do
      run_merge "${sys}" "${n}" "${kind}"
    done
  done
done

echo "[DONE] All merges complete. Output dir: ${OUT_DIR}"
