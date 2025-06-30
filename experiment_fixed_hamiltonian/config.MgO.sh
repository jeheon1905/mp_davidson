# configuration for MgO
supercell_sizes=(1 2 3 4 5 6 7 8 9 10)

system_name=MgO_1x1x2
nocc=64
system_options="--filepath ./data/systems/${system_name}.cif --pbc 1 1 1"

# experiment name and save directory
save_dir=./experiment_fixed_hamiltonian/expt.${system_name}
additional_options="--use_dense_proj"

# Define the calculation options
diag_iter=40
fixed_convg_tol=1e-10
precond_iter=5
use_cuda=1
no_shift_thr=10.0
spacing=0.15
pp_type=SG15
unocc_ratio=1.05

# Define the array of options
options=(
  "--fp DP"
  "--fp SP --multi_dtype DP SP --precond_fp SP"
  "--fp DP --multi_dtype DP SP --precond_fp SP"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 2"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 3"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 4"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 5"

  "--fp DP --recalc_convg_history"
  "--fp SP --multi_dtype DP SP --precond_fp SP --recalc_convg_history"
  "--fp DP --multi_dtype DP SP --precond_fp SP --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 2 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 3 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 4 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 5 --recalc_convg_history"

  # TF32
  "--fp SP --multi_dtype DP SP --precond_fp SP --allow_tf32"
  "--fp DP --multi_dtype DP SP --precond_fp SP --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1 --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 2 --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 3 --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 4 --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 5 --allow_tf32"

  "--fp SP --multi_dtype DP SP --precond_fp SP --recalc_convg_history --allow_tf32"
  "--fp DP --multi_dtype DP SP --precond_fp SP --recalc_convg_history --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1 --recalc_convg_history --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 2 --recalc_convg_history --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 3 --recalc_convg_history --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 4 --recalc_convg_history --allow_tf32"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 5 --recalc_convg_history --allow_tf32"

  # BF16
  "--fp DP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_dtype BF16"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype BF16"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 2 --MP_dtype BF16"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 3 --MP_dtype BF16"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 4 --MP_dtype BF16"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 5 --MP_dtype BF16"

  "--fp DP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_dtype BF16 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype BF16 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 2 --MP_dtype BF16 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 3 --MP_dtype BF16 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 4 --MP_dtype BF16 --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 5 --MP_dtype BF16 --recalc_convg_history"

  # MP using SP + precond using BF16
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype SP"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype SP --recalc_convg_history"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP" "SP" "DP_SP4precond" "MP_scheme1" "MP_scheme2" "MP_scheme3" "MP_scheme4" "MP_scheme5"
  "DP.recalc_convg_history" "SP.recalc_convg_history" "DP_SP4precond.recalc_convg_history" "MP_scheme1.recalc_convg_history" "MP_scheme2.recalc_convg_history" "MP_scheme3.recalc_convg_history" "MP_scheme4.recalc_convg_history" "MP_scheme5.recalc_convg_history"
  "TF32" "DP_TF324precond" "MP_scheme1_TF32" "MP_scheme2_TF32" "MP_scheme3_TF32" "MP_scheme4_TF32" "MP_scheme5_TF32"
  "TF32.recalc_convg_history" "DP_TF324precond.recalc_convg_history" "MP_scheme1_TF32.recalc_convg_history" "MP_scheme2_TF32.recalc_convg_history" "MP_scheme3_TF32.recalc_convg_history" "MP_scheme4_TF32.recalc_convg_history" "MP_scheme5_TF32.recalc_convg_history"
  "DP_BF164precond" "MP_scheme1_BF16" "MP_scheme2_BF16" "MP_scheme3_BF16" "MP_scheme4_BF16" "MP_scheme5_BF16"
  "DP_BF164precond.recalc_convg_history" "MP_scheme1_BF16.recalc_convg_history" "MP_scheme2_BF16.recalc_convg_history" "MP_scheme3_BF16.recalc_convg_history" "MP_scheme4_BF16.recalc_convg_history" "MP_scheme5_BF16.recalc_convg_history"
  "MP_scheme1_BF164precond" "MP_scheme1_BF164precond.recalc_convg_history"
)
