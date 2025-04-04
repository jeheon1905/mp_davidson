# configuration for Cu_fcc
supercell_sizes=(1 2 3 4 5 6 7 8 9 10)

system_name=Cu_fcc_3x3x1
nocc=342
system_options="--filepath ./data/systems/${system_name}.cif --pbc 1 1 1"

# experiment name and save directory
save_dir=expt.${system_name}.A100.dense
additional_options="--use_dense_kinetic --use_dense_proj"

# save_dir=expt.${system_name}.A100.sparse
# additional_options=""

# save_dir=expt.${system_name}.A100.tf32
# additional_options="--use_dense_kinetic --use_dense_proj --allow_tf32"

# Define the calculation options
diag_iter=40
fixed_convg_tol=1e-10
precond_iter=5
use_cuda=1
no_shift_thr=10.0
spacing=0.2
pp_type=SG15
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

  # "--fp DP --recalc_convg_history"
  # "--fp SP --dynamic --precond_fp SP --recalc_convg_history"
  # "--fp DP --dynamic --precond_fp SP --recalc_convg_history"
  # "--fp MP --dynamic --precond_fp SP --MP_scheme 1 --recalc_convg_history"
  # "--fp MP --dynamic --precond_fp SP --MP_scheme 2 --recalc_convg_history"
  # "--fp MP --dynamic --precond_fp SP --MP_scheme 3 --recalc_convg_history"
  # "--fp MP --dynamic --precond_fp SP --MP_scheme 4 --recalc_convg_history"
  # "--fp MP --dynamic --precond_fp SP --MP_scheme 5 --recalc_convg_history"
)

# Define the array of option names corresponding to the above options
option_names=(
  "DP" "SP" "DP_SP4precond" "MP_scheme1" "MP_scheme2" "MP_scheme3" "MP_scheme4" "MP_scheme5"
  # "DP.recalc_convg_history" "SP.recalc_convg_history" "DP_SP4precond.recalc_convg_history" "MP_scheme1.recalc_convg_history" "MP_scheme2.recalc_convg_history" "MP_scheme3.recalc_convg_history" "MP_scheme4.recalc_convg_history" "MP_scheme5.recalc_convg_history"
)