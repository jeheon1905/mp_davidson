# configuration for Si_diamond
supercell_sizes=(1 2 3 4 5 6 7 8 9 10)

system_name=Si_diamond_2x2x1
nocc=64
system_options="--filepath ./data/systems/${system_name}.cif --pbc 1 1 1"

# experiment name and save directory
save_dir=expt.${system_name}.dense
additional_options="--use_dense_kinetic --use_dense_proj"

# Define the calculation options
diag_iter=40
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
  "--fp SP --dynamic --precond_fp SP"
  "--fp DP --dynamic --precond_fp SP"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 1"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 2"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 3"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 4"
  "--fp MP --dynamic --precond_fp SP --MP_scheme 5"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP" "SP" "DP_SP4precond" "MP_scheme1" "MP_scheme2" "MP_scheme3" "MP_scheme4" "MP_scheme5"
)
