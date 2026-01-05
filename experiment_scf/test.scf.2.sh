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
use_cuda=1
diag_iter=2
precond_iter=5
fixed_convg_tol=1e-3
scf_energy_tol=1e-7
scf_density_tol=1e-5

phase=scf

# Seed array
seeds=(42 43 44 45 46)

# Loop over seeds
for seed in "${seeds[@]}"; do
  save_dir=./experiment_scf/expt.${system_name}.etol${scf_energy_tol}.dtol${scf_density_tol}.seed${seed}

  # Define base calc_options (without diag_iter, fixed_convg_tol, scf_energy_tol)
  base_calc_options="--phase $phase \
        --pp_type $pp_type \
        --no_shift_thr $no_shift_thr --spacing $spacing \
        --precond_iter $precond_iter --use_cuda $use_cuda \
        --seed $seed"

  for n in "${supercell_sizes[@]}"; do
    cell="1 1 ${n}"
    cell_name="1_1_${n}"
    nbands=$(python -c "print(round($nocc * $n * $unocc_ratio))")

    # Standard calculation options
    calc_options="$base_calc_options \
            --diag_iter $diag_iter \
            --fixed_convg_tol $fixed_convg_tol \
            --scf_energy_tol $scf_energy_tol \
            --scf_density_tol $scf_density_tol"

    # Run standard calculations
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

    # Reference calculation (DP with tighter thresholds) - only for seed 42
    if [ "$seed" -eq 42 ]; then
      calc_options_ref="$base_calc_options \
            --warmup 0 \
            --diag_iter 11 \
            --fixed_convg_tol $(awk "BEGIN {print $fixed_convg_tol * 0.01}") \
            --scf_energy_tol $(awk "BEGIN {print $scf_energy_tol * 0.01}") \
            --scf_density_tol $(awk "BEGIN {print $scf_density_tol * 0.01}") \
            --seed $seed"
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
        $opt > ${prefix}.log 2>&1"

      echo "Running: $prefix"
      eval $cmd
    fi
  done
done

cd -  # Go back to the original directory
