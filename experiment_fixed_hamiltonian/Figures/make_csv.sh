#!/bin/bash

supercell_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
option_names=("MP_scheme1_BF164precond")
system_list=("CNT_6_0" "MgO_1x1x2" "Si_diamond_2x2x1")
gpu_info="A100"


# Create results directory
results_dir="csv_results_${gpu_info}"
mkdir -p "${results_dir}"

# Main processing loop
for system in "${system_list[@]}"; do
  for n in "${supercell_sizes[@]}"; do

    cell="1 1 ${n}"
    cell_name="1_1_${n}"

    for idx in "${!option_names[@]}"; do
      opt_name="${option_names[$idx]}"

      cmd="python analyze_acc.py \
        --ref_log ../expt.${system}/1_1_${n}_DP.speed.log \
        --prb_log ../expt.${system}/1_1_${n}_${opt_name}.speed.log \
        --csv csv_results_${gpu_info}/${system}.1_1_${n}.csv
      "
      echo $cmd
      eval $cmd
    done
  done
done

echo "Analysis complete! Results saved in: ${results_dir}"
