#!/usr/bin/env bash
# set -euo pipefail   # 필요시 해제하세요 (엄격 모드)

# ===================== User config =====================
SYSTEMS=("CNT_6_0" "MgO_1x1x2" "Si_diamond_2x2x1")

# Figures 디렉토리 (모든 시스템 공통 경로)
FIG_DIR="./Figures_convg_history_add_virtual"

# 병합 결과 저장 경로
OUT_DIR="./Figures_convg_history_add_virtual"
mkdir -p "${OUT_DIR}"

# merge_svg.py 실행 파일 경로
MERGER="merge_svg.py"

# 그리드/레이아웃 설정 (1행 5열)
ROWS=1
COLS=5
EMPTY_CELLS=()  # 빈 셀 없음
CELL_W=200
CELL_H=200
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
  "DP.recalc_convg_history"
  "SP.recalc_convg_history"
  "MP_scheme1.recalc_convg_history"
  "DP_SP4precond.recalc_convg_history"
  "MP_scheme1_BF164precond.recalc_convg_history"
)

# kinds: eigval / residual
KINDS=("eigval" "residual")

# ---- helper: 파일 리스트를 option_names 순서대로 생성 ----
# 파일명 = <FIG_DIR>/<system>.<cell>_<opt>.<kind>.svg
# 예: CNT_6_0.1_1_1_DP.recalc_convg_history.eigval.svg
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
  local kind="$2"

  local cell="1_1_1"  # add_virtual은 1_1_1만 사용

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
    --cell-size "${CELL_W}" "${CELL_H}" \
    --gutter "${GUTTER_X}" "${GUTTER_Y}" \
    --margin "${MARGIN_L}" "${MARGIN_T}" "${MARGIN_R}" "${MARGIN_B}" \
    --output "${out_svg}" \
    "${png_args[@]}"
}

# ===================== main loop =====================
for sys in "${SYSTEMS[@]}"; do
  echo "===== System: ${sys} ====="

  for kind in "${KINDS[@]}"; do
    run_merge "${sys}" "${kind}"
  done
done

echo "[DONE] All merges complete. Output dir: ${OUT_DIR}"
