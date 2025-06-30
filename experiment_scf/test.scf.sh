#!/bin/bash

# Usage example)
# bash run_test.sh configs/config.SYSTEM_NAME.sh

# set -e # for debugging.
cd ..  # Go to the parent directory

# Load the configuration file passed by the user
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_test.sh configs/config.SYSTEM_NAME.sh"
  exit 1
fi

source "$CONFIG"

# Common system options
diag_iter=2
fixed_convg_tol=1e-3
precond_iter=5
scf_energy_tol=1e-5
use_cuda=1

# diag_iter=3  # TEST: to enhance scf convergence
# save_dir=./experiment_scf/expt.${system_name}.diag_iter${diag_iter}

phase=scf
calc_options="--phase $phase --scf_energy_tol $scf_energy_tol \
      --pp_type $pp_type --fixed_convg_tol $fixed_convg_tol \
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
      $opt > ${prefix}.log 2>&1"

    echo "Running: $prefix"
    eval $cmd
  done
done

cd -  # Go back to the original directory
