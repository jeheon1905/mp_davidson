#!/bin/bash

supercell_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
supercell_sizes=(1)  # NOTE: DEBUG
supercell_sizes=(5)  # NOTE: DEBUG
supercell_sizes=(1 3 5 7 9 11)

system_name=CNT_6_0
nocc=48
system_options="--filepath ../data/systems/${system_name}.cif --pbc 0 0 1"

use_cuda=1
spacing=0.15
pp_type=SG15
unocc_ratio=1.05

calc_options="--pp_type $pp_type --spacing $spacing --use_cuda $use_cuda"
options=(
    # "--fp DP"
    # "--fp SP"
    # "--fp SP --allow_tf32"
    # "--fp HP"
    # "--fp BF16"

    "--fp DP --use_dense_proj"
    "--fp SP --use_dense_proj"
    "--fp SP --allow_tf32 --use_dense_proj"
    "--fp HP --use_dense_proj"
    "--fp BF16 --use_dense_proj"
)
operation="projection kinetic local nonlocal tensordot"

for n in "${supercell_sizes[@]}"; do
  cell="1 1 ${n}"
  cell_name="1_1_${n}"
  nbands=$(python -c "print(round($nocc * $n * $unocc_ratio))")

  for idx in "${!options[@]}"; do
    opt="${options[$idx]}"

    cmd="python test_op_time.py $system_options --nbands $nbands --supercell $cell $opt \
        --operation $operation"

    echo $cmd
    eval $cmd
  done
done