#!/bin/bash
# Define the supercell array (each element is a string in the format "a b c")
supercells=(
  "1 1 1"
  "1 1 2"
  "1 1 3"
  "1 1 4"
  "1 1 5"
  "1 1 6"
)

# Define the array of options
options=(
  "--fp DP"
  "--fp SP --dynamic --precond_fp SP"
  "--fp DP --dynamic --precond_fp SP"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 1"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 2"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 3"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 4"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 5"

  "--fp DP --recalc_convg_history"
  "--fp SP --dynamic --precond_fp SP --recalc_convg_history"
  "--fp DP --dynamic --precond_fp SP --recalc_convg_history"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 1 --recalc_convg_history"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 2 --recalc_convg_history"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 3 --recalc_convg_history"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 4 --recalc_convg_history"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 5 --recalc_convg_history"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP"
  "SP"
  "DP_SP4precond"
  "MP_scheme1"
  "MP_scheme2"
  "MP_scheme3"
  "MP_scheme4"
  "MP_scheme5"

  "DP.recalc_convg_history"
  "SP.recalc_convg_history"
  "DP_SP4precond.recalc_convg_history"
  "MP_scheme1.recalc_convg_history"
  "MP_scheme2.recalc_convg_history"
  "MP_scheme3.recalc_convg_history"
  "MP_scheme4.recalc_convg_history"
  "MP_scheme5.recalc_convg_history"
)

precond_iter=5
use_cuda=1
# use_cuda=0

# Set a low convg_tol and unify diag_iter for easier comparison through visualization
diag_iter=40
fixed_convg_tol=1e-10

# Execute the loop for each supercell and each option
for cell in "${supercells[@]}"; do
  # Replace spaces with underscores to create the cell name
  cell_name=${cell// /_}

  # Iterate over each index of the options array
  for idx in "${!options[@]}"; do
    opt="${options[$idx]}"
    opt_name="${option_names[$idx]}"

    # Store the command to be executed in a single string variable
    cmd="python test.py --supercell $cell --pbc 0 0 1 \
      --filepath ../system_cif/CNT_6.0.cif \
      --phase fixed --pp_type SG15 --fixed_convg_tol $fixed_convg_tol \
      --precond_iter $precond_iter --diag_iter $diag_iter \
      --use_dense_proj --use_cuda $use_cuda \
      --retHistory History_${cell_name}_${opt_name}.pt \
      $opt > ${cell_name}_${opt_name}.log 2>&1"

    # Print the command
    echo "Running command: $cmd"
    # Execute the command
    eval $cmd
  done
done

# Generate plots from retHistory files for each supercell and each option
for cell in "${supercells[@]}"; do
  # Replace spaces with underscores for the cell name used in filenames
  cell_name=${cell// /_}

  # Determine num_eig based on the value of the supercell (direct comparison)
  if [ "$cell" == "1 1 1" ]; then
    num_eig=48
  elif [ "$cell" == "1 1 2" ]; then
    num_eig=96
  elif [ "$cell" == "1 1 3" ]; then
    num_eig=144
  elif [ "$cell" == "1 1 4" ]; then
    num_eig=192
  elif [ "$cell" == "1 1 5" ]; then
    num_eig=240
  elif [ "$cell" == "1 1 6" ]; then
    num_eig=288
  else
    echo "Unknown supercell: $cell"
    continue
  fi

  # Generate plots for each option
  for opt_name in "${option_names[@]}"; do
    retHistory_file="History_${cell_name}_${opt_name}.pt"

    echo "Generating residual plot from ${retHistory_file} with num_eig=${num_eig}"
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" --num_eig ${num_eig} --plot residual --save History.residual.${cell_name}_${opt_name}.png

    echo "Generating eigval plot from ${retHistory_file} with num_eig=${num_eig}"
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" --num_eig ${num_eig} --plot eigval --save History.eigval.${cell_name}_${opt_name}.png
  done
done
