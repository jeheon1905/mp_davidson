# configuration for Si_diamond
supercell_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

system_name=Si_diamond_2x2x1
nocc=64
system_options="--filepath ./data/systems/${system_name}.cif --pbc 1 1 1"

# experiment name and save directory
save_dir=./experiment_fixed_hamiltonian/expt.${system_name}
additional_options="--use_dense_proj"

# Define the calculation options
diag_iter=20  # 'diag_iter' is set to 20 for speed test.
fixed_convg_tol=1e-10
precond_iter=5
use_cuda=1
no_shift_thr=10.0
spacing=0.15
# pp_type=SG15
pp_type=NNLP # for testing NNLP
unocc_ratio=1.05

# Define the array of options
options=(
  "--fp DP"
  "--fp SP --multi_dtype DP SP --precond_fp SP"
  "--fp DP --multi_dtype DP SP --precond_fp SP"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1"
  # MP using SP + precond using BF16
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype SP"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP" "SP" "DP_SP4precond" "MP_scheme1"
  "MP_scheme1_BF164precond"
)
