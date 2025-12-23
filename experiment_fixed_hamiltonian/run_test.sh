#!/bin/bash

# Usage example)
# bash run_test.sh configs/config.SYSTEM_NAME.sh

# set -e  # for debugging.
cd ..  # Go to the parent directory

# Load the configuration file passed by the user
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_test.sh configs/config.SYSTEM_NAME.sh"
  exit 1
fi

source "$CONFIG"

# Define base calc_options once
base_calc_options="--phase fixed --pp_type $pp_type --fixed_convg_tol $fixed_convg_tol \
      --no_shift_thr $no_shift_thr --spacing $spacing \
      --precond_iter $precond_iter --use_cuda $use_cuda"

for n in "${supercell_sizes[@]}"; do
  cell="1 1 ${n}"
  cell_name="1_1_${n}"
  nbands=$(python -c "print(round($nocc * $n * $unocc_ratio))")
  calc_options="$base_calc_options --diag_iter $diag_iter"

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
      --warmup 0 \
      $opt > ${prefix}.log 2>&1"

    echo "Running: $prefix"
    echo "$cmd"
    eval "$cmd"
  done

  # Reference calculation (DP with extended iterations)
  calc_options_ref="$base_calc_options --diag_iter $((diag_iter + 20))"
  opt="--fp DP"
  opt_name="DP_ref"

  prefix="${save_dir}/${cell_name}_${opt_name}"
  mkdir -p "$(dirname $prefix)"

  cmd="python test.py \
    $system_options \
    $calc_options_ref \
    $additional_options \
    --nbands $nbands \
    --supercell $cell \
    --retHistory ${prefix}.pt \
    --warmup 0 \
    $opt > ${prefix}.log 2>&1"

  echo "Running: $prefix"
  echo "$cmd"
  eval "$cmd"
done

cd -  # Go back to the original directory
