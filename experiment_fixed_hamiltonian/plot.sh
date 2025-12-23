#!/bin/bash

cd ..  # Go to the parent directory

# Load the configuration file passed by the user
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash plot.sh configs/config.SYSTEM_NAME.sh"
  exit 1
fi

echo "Loading configuration from $CONFIG..."
source "$CONFIG"
echo "Configuration loaded successfully."

# Plotting
for n in "${supercell_sizes[@]}"; do
  cell_name="1_1_${n}"
  num_eig=$((${nocc} * n))
  echo "Processing supercell size: $cell_name (num_eig = $num_eig)"

  for opt_name in "${option_names[@]}"; do
    prefix="${save_dir}/${cell_name}_${opt_name}"
    retHistory_file="${prefix}.pt"
    refHistory_file="${save_dir}/${cell_name}_DP_ref.pt"
    echo "  Option: $opt_name"
    echo "    Loading file: ${retHistory_file}"

    echo "    Plotting residual convergence history..."
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --num_eig ${num_eig} --plot residual \
      --save ${prefix}.residual.png
    echo "    -> Saved residual plot to ${prefix}.residual.png"

    echo "    Plotting eigenvalue convergence history..."
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --ref_filepath ${refHistory_file} \
      --num_eig ${num_eig} --plot eigval \
      --save ${prefix}.eigval.png
    echo "    -> Saved eigval plot to ${prefix}.eigval.png"
  done

  echo "Finished processing $cell_name."
done

echo "All plotting complete."

cd -  # Go back to the original directory
