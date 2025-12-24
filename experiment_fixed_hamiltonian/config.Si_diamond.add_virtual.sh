# configuration for Si_diamond
supercell_sizes=(1)

system_name=Si_diamond_2x2x1
nocc=64
system_options="--filepath ./data/systems/${system_name}.cif --pbc 1 1 1"

# experiment name and save directory
save_dir=./experiment_fixed_hamiltonian/expt.${system_name}.add_virtual
additional_options="--use_dense_proj"

# Define the calculation options
diag_iter=100  # more 'diag_iter'
fixed_convg_tol=1e-10
precond_iter=5
use_cuda=1
no_shift_thr=10.0
spacing=0.15
pp_type=SG15
unocc_ratio=1.10

# Define the array of options
options=(
  "--fp DP --recalc_convg_history"
  "--fp SP --multi_dtype DP SP --precond_fp SP --recalc_convg_history"
  "--fp MP --multi_dtype DP SP --precond_fp SP --MP_scheme 1 --recalc_convg_history"
  "--fp DP --multi_dtype DP SP --precond_fp SP --recalc_convg_history"
  "--fp MP --multi_dtype DP SP BF16 --precond_fp BF16 --MP_scheme 1 --MP_dtype SP --recalc_convg_history"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP.recalc_convg_history"
  "SP.recalc_convg_history"
  "MP_scheme1.recalc_convg_history"
  "DP_SP4precond.recalc_convg_history"
  "MP_scheme1_BF164precond.recalc_convg_history"
)
