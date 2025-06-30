
# Figure 1 and Figure S1
```bash
# Save figures into ./Figures_convg_history
source plot.convg_history.sh
```

# Figure S2
Not implemented yet.


# Figure 2: Time breakdown

- CNT (6, 0)

```bash
python plot_breakdown.py \
  --log_dir ../expt.CNT_6_0 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'DP' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.CNT_1_1_5.svg \
  --json_out Figures_time_breakdown/time_breakdown.CNT_1_1_5.json \
  --phase fixed \
  --separate_legend
```

- BaTiO3

```bash
python plot_breakdown.py \
  --log_dir ../expt.BaTiO3_2x2x1 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'DP' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.BaTiO3_2x2x1.svg \
  --json_out Figures_time_breakdown/time_breakdown.BaTiO3_2x2x1.json \
  --phase fixed \
  --separate_legend
```

- Si diamond

```bash
python plot_breakdown.py \
  --log_dir ../expt.Si_diamond_2x2x1 \
  --supercell 1_1_4 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'DP' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.svg \
  --json_out Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.json \
  --phase fixed \
  --separate_legend

- MgO

```bash
python plot_breakdown.py \
  --log_dir ../expt.MgO_1x1x2 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'DP' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.MgO_1x1x2.svg \
  --json_out Figures_time_breakdown/time_breakdown.MgO_1x1x2.json \
  --phase fixed \
  --separate_legend
```
```

# Figure 3:



