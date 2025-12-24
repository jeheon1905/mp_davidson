#!/bin/bash

cd ../..  # Go to the 최상위 directory

# Load the configuration file passed by the user
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash plot.sh configs/config.SYSTEM_NAME.sh"
  cd -  # Go back to the original directory
  return 1 2>/dev/null || exit 1
fi

echo "Loading configuration from $CONFIG..."
source "$CONFIG"
echo "Configuration loaded successfully."

# ===== Title name mapping =====
declare -A TITLE_MAP=(
  ["DP.recalc_convg_history"]="FP64"
  ["SP.recalc_convg_history"]="FP32"
  ["MP_scheme1.recalc_convg_history"]="MP1"
  ["DP_SP4precond.recalc_convg_history"]="MP6"
  ["MP_scheme1_BF164precond.recalc_convg_history"]="MP1*"
)

# Validate all option names before processing
echo "===== Validating option names ====="
for opt_name in "${option_names[@]}"; do
  if [[ -z ${TITLE_MAP[$opt_name]+_} ]]; then
    echo "[ERROR] Option '${opt_name}' not found in TITLE_MAP" >&2
    echo "[ERROR] Valid options are: ${!TITLE_MAP[@]}" >&2
    cd -  # Go back to the original directory
    return 1 2>/dev/null || exit 1
  fi
done
echo "[OK] All option names are valid"
echo ""

# Define figure save directory
save_fig_dir="./experiment_fixed_hamiltonian/Figures/Figures_convg_history_add_virtual"
mkdir -p ${save_fig_dir}

# Plotting
for n in "${supercell_sizes[@]}"; do
  cell_name="1_1_${n}"
  num_eig=$((${nocc} * n))
  diag_iter=60

  # Calculate total nbands and num_virt
  nbands=$(python -c "print(round($nocc * $n * $unocc_ratio))")
  num_virt=$((nbands - num_eig))

  echo "Processing supercell size: $cell_name"
  echo "  num_eig (occupied) = ${num_eig}"
  echo "  nbands (total)     = ${nbands}"
  echo "  num_virt (virtual) = ${num_virt}"

  for opt_name in "${option_names[@]}"; do
    # Get title from mapping
    title="${TITLE_MAP[$opt_name]}"
    
    prefix="${save_dir}/${cell_name}_${opt_name}"
    retHistory_file="${prefix}.pt"
    refHistory_file="${save_dir}/${cell_name}_DP_ref.pt"
    echo "  Option: $opt_name (title: ${title})"
    echo "    Loading file: ${retHistory_file}"

    echo "    Plotting residual convergence history..."
    python plot_convg_history.py --convg_tol 1e-7 --filepath "${retHistory_file}" \
      --num_eig ${num_eig} --num_virt ${num_virt} --plot residual \
      --diag_iter ${diag_iter} \
      --title "${title}" \
      --save ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.residual.svg
    echo "    -> Saved residual plot to ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.residual.svg"

    echo "    Plotting eigenvalue convergence history..."
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --ref_filepath ${refHistory_file} \
      --num_eig ${num_eig} --num_virt ${num_virt} --plot eigval \
      --diag_iter ${diag_iter} \
      --title "${title}" \
      --save ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.eigval.svg
    echo "    -> Saved eigval plot to ${save_fig_dir}/${system_name}.${cell_name}_${opt_name}.eigval.svg"
  done

  echo "Finished processing $cell_name."
done

echo "All plotting complete."

cd -  # Go back to the original directory
