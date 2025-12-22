# Figure 4: Time breakdown (SCF)


```bash
mkdir Figures_time_breakdown

# CNT (6, 0)
python ../../experiment_fixed_hamiltonian/Figures/plot_breakdown.py \
  --log_dir ../expt.CNT_6_0 \
  --supercell 1_1_7 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.CNT.1_1_7.svg \
  --json_out Figures_time_breakdown/time_breakdown.CNT.1_1_7.json \
  --phase scf \
  --separate_legend \
  --log_suffix .log \
  --legend_cols 4

# MgO
python ../../experiment_fixed_hamiltonian/Figures/plot_breakdown.py \
  --log_dir ../expt.MgO_1x1x2 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.MgO_1x1x2.1_1_5.svg \
  --json_out Figures_time_breakdown/time_breakdown.MgO_1x1x2.1_1_5.json \
  --phase scf \
  --separate_legend \
  --log_suffix .log \
  --legend_cols 4

# Si diamond
python ../../experiment_fixed_hamiltonian/Figures/plot_breakdown.py \
  --log_dir ../expt.Si_diamond_2x2x1 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.1_1_5.svg \
  --json_out Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.1_1_5.json \
  --phase scf \
  --separate_legend \
  --log_suffix .log \
  --legend_cols 4
```
