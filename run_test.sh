#!/bin/bash

# Usage example)
# bash run_test.sh configs/config.SYSTEM_NAME.sh

set -e  # for debugging.

# Load the configuration file passed by the user
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_test.sh configs/config.SYSTEM_NAME.sh"
  exit 1
fi

source "$CONFIG"

calc_options="--phase fixed --pp_type $pp_type --fixed_convg_tol $fixed_convg_tol \
      --no_shift_thr $no_shift_thr --spacing $spacing \
      --precond_iter $precond_iter --diag_iter $diag_iter --use_cuda $use_cuda"

for n in "${supercell_sizes[@]}"; do
  cell="1 1 ${n}"
  cell_name="1_1_${n}"
  nbands=$(python -c "print(round($nocc * $n * $unocc_ratio))")

  for idx in "${!options[@]}"; do
    opt="${options[$idx]}"
    opt_name="${option_names[$idx]}"

    prefix="${save_dir}/${cell_name}_${opt_name}"
    mkdir -p "$(dirname $prefix)"

    cmd="python test.py \
      $system_options \
      $calc_options \
      $additional_options \
      --nbands $nbands \
      --supercell $cell \
      --retHistory ${prefix}.pt \
      $opt > ${prefix}.log 2>&1"

    echo "Running: $prefix"
    eval $cmd
  done
done

# Plotting
for n in "${supercell_sizes[@]}"; do
  cell_name="1_1_${n}"
  num_eig=$((${nocc} * n))

  for opt_name in "${option_names[@]}"; do
    prefix="${save_dir}/${cell_name}_${opt_name}"
    retHistory_file="${prefix}.pt"

    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --num_eig ${num_eig} --plot residual \
      --save ${prefix}.residual.png

    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --num_eig ${num_eig} --plot eigval \
      --save ${prefix}.eigval.png
  done
done
