#!/bin/bash

# Activate your environment if needed
# source activate your_env_name

# Define supercells, convergence flag, and plot types
supercells=("1_1_1" "1_1_2" "1_1_3" "1_1_4" "1_1_5" "1_1_6")
plots=("eigval" "residual")
recalc_flags=(true false)

# Iterate over each combination
for supercell in "${supercells[@]}"; do
  for plot in "${plots[@]}"; do
    for recalc in "${recalc_flags[@]}"; do
      
      # Start message
      echo "Running merge_plot.py with supercell=${supercell}, plot=${plot}, recalc_convg_history=${recalc}"

      # Build command
      cmd="python merge_plot.py --plot ${plot} --supercell ${supercell}"

      # If recalc_convg_history is true, add the flag
      if [ "$recalc" = true ]; then
        cmd="${cmd} --recalc_convg_history"
      fi

      # Run the command
      echo "$cmd"
      eval $cmd

      # Optional sleep if you want to delay between runs
      # sleep 1

    done
  done
done

echo "All tasks finished!"
