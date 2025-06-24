#!/bin/bash

cd ../..  # Go to the 최상위 directory

supercell_sizes=(5)
system_name=CNT_6_0
nocc=48

# experiment name and save directory
save_dir=./experiment_fixed_hamiltonian/expt.${system_name}
save_fig_dir=./experiment_fixed_hamiltonian/Figures/Figures_convg_history

option_names=(
  # recalc=False (Supplementary Figure)
  "DP" "SP" "MP_scheme1_BF164precond"
  "MP_scheme1" "MP_scheme2" "MP_scheme3"
  "MP_scheme4" "MP_scheme5" "DP_SP4precond"

  # recalc=True (Main text Figure)
  "DP.recalc_convg_history" "SP.recalc_convg_history" "MP_scheme1_BF164precond.recalc_convg_history"
  "MP_scheme1.recalc_convg_history" "MP_scheme2.recalc_convg_history" "MP_scheme3.recalc_convg_history"
  "MP_scheme4.recalc_convg_history" "MP_scheme5.recalc_convg_history" "DP_SP4precond.recalc_convg_history"
)

title_names=(
  "DP" "SP" "MP1*"
  "MP1" "MP2" "MP3"
  "MP4" "MP5" "MP6"

  "DP" "SP" "MP1*"
  "MP1" "MP2" "MP3"
  "MP4" "MP5" "MP6"
)


# Plotting
for n in "${supercell_sizes[@]}"; do
  cell_name="1_1_${n}"
  num_eig=$((${nocc} * n))
  echo "Processing supercell size: $cell_name (num_eig = $num_eig)"

  for idx in "${!option_names[@]}"; do
    opt_name="${option_names[$idx]}"
    title_name="${title_names[$idx]}"

    prefix="${save_dir}/${cell_name}_${opt_name}"
    retHistory_file="${prefix}.pt"
    refHistory_file="${save_dir}/${cell_name}_DP.pt"
    echo "  Option: $opt_name"
    echo "    Loading file: ${retHistory_file}"

    echo "    Plotting residual convergence history..."
    python plot_convg_history.py --convg_tol 1e-7 --filepath "${retHistory_file}" \
      --num_eig ${num_eig} --plot residual \
      --title "${title_name}" \
      --save ${save_fig_dir}/${cell_name}_${opt_name}.residual.svg
    echo "    -> Saved residual plot to ${save_fig_dir}/${cell_name}_${opt_name}.residual.svg"

    echo "    Plotting eigenvalue convergence history..."
    python plot_convg_history.py --convg_tol 1e-14 --filepath "${retHistory_file}" \
      --ref_filepath ${refHistory_file} \
      --num_eig ${num_eig} --plot eigval \
      --title "${title_name}" \
      --save ${save_fig_dir}/${cell_name}_${opt_name}.eigval.svg
    echo "    -> Saved eigenvalue plot to ${save_fig_dir}/${cell_name}_${opt_name}.eigval.svg"
  done

  echo "Finished processing $cell_name."
done

echo "All plotting complete."

cd -  # Go back to the original directory
