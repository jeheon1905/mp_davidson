# Figures

After peforming speed tests at `expeirment_fixed_hamiltnoian/`, plot figures.

## Figure 1 and Figure S1

Plot convergence history with x-axis of iterations and y-axis of eigenvalue errors or residual norm.
Compared methods: DP, SP, MP1, MP2, MP3, MP4, MP5, MP6, MP1*

```bash
# Save figures into ./Figures_convg_history
source plot.convg_history.sh
```

## Figure S2

Plot convergence history with x-axis of iterations and y-axis of eigenvalue errors or residual norm.
Compared methods: all

```bash
# Save figures into ./Figures_convg_histry_SI
source plot.convg_history.SI.sh
source merge.convg_history.SI.sh  # merge the history figures to the single figure
```


## Figure 2: Time breakdown

```bash
mkdir Figures_time_breakdown

# CNT (6, 0)
python plot_breakdown.py \
  --log_dir ../expt.CNT_6_0 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.CNT.1_1_5.svg \
  --json_out Figures_time_breakdown/time_breakdown.CNT.1_1_5.json \
  --phase fixed \
  --separate_legend

# MgO
python plot_breakdown.py \
  --log_dir ../expt.MgO_1x1x2 \
  --supercell 1_1_7 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.MgO_1x1x2.1_1_7.svg \
  --json_out Figures_time_breakdown/time_breakdown.MgO_1x1x2.1_1_7.json \
  --phase fixed \
  --separate_legend

# Si diamond
python plot_breakdown.py \
  --log_dir ../expt.Si_diamond_2x2x1 \
  --supercell 1_1_3 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.1_1_3.svg \
  --json_out Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.1_1_3.json \
  --phase fixed \
  --separate_legend
```


## Figure 3: performance enhancements as a function of system sizes

Plot performance enhancements of three different systems with x-axis of the number of atoms and y-axis of performance enhancements.

```bash
source make_csv.sh
python plot_performance_vs_size.py
```


## Figure: Time breakdown of preconditioning
```bash
# CNT (6, 0)
python plot_breakdown.py \
  --log_dir ../expt.CNT_6_0 \
  --supercell 1_1_5 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown_precond.CNT.1_1_5.svg \
  --json_out Figures_time_breakdown/time_breakdown_precond.CNT.1_1_5.json \
  --phase preconditioning \
  --separate_legend

# MgO
python plot_breakdown.py \
  --log_dir ../expt.MgO_1x1x2 \
  --supercell 1_1_7 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown_precond.MgO_1x1x2.1_1_7.svg \
  --json_out Figures_time_breakdown/time_breakdown_precond.MgO_1x1x2.1_1_7.json \
  --phase preconditioning \
  --separate_legend

# CNT (6, 0)
python plot_breakdown.py \
  --log_dir ../expt.Si_diamond_2x2x1 \
  --supercell 1_1_3 \
  --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
  --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
  --output Figures_time_breakdown/time_breakdown_precond.Si_diamond_2x2x1.1_1_3.svg \
  --json_out Figures_time_breakdown/time_breakdown_precond.Si_diamond_2x2x1.1_1_3.json \
  --phase preconditioning \
  --separate_legend
```

