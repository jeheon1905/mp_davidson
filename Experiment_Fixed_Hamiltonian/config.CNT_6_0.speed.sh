# configuration for CNT (6, 0)
supercell_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)

system_name=CNT_6_0
nocc=48
system_options="--filepath ./data/systems/${system_name}.cif --pbc 0 0 1"

# experiment name and save directory
save_dir=./Experiment_Fixed_Hamiltonian/expt.${system_name}
additional_options="--use_dense_proj"

# Define the calculation options
diag_iter=20  # 'diag_iter' is set to 20 for speed test.
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

  # TF32
  "--fp SP --multi_dtype DP SP --precond_fp SP --allow_tf32"
  "--fp DP --multi_dtype DP SP --precond_fp SP --allow_tf32"
  # "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1 --allow_tf32"
  # "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 2 --allow_tf32"
  # "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 3 --allow_tf32"
  # "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 4 --allow_tf32"
  # "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 5 --allow_tf32"

  # BF16
  "--fp DP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_dtype BF16"
  # "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype BF16"
  # "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 2 --MP_dtype BF16"
  # "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 3 --MP_dtype BF16"
  # "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 4 --MP_dtype BF16"
  # "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 5 --MP_dtype BF16"

  # MP using SP + precond using BF16
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype SP"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP" "SP" "DP_SP4precond" "MP_scheme1" "MP_scheme2" "MP_scheme3" "MP_scheme4" "MP_scheme5"
  "TF32" "DP_TF324precond"
  "DP_BF164precond"
  "MP_scheme1_BF164precond"
)
