# Figure 4: Time breakdown (SCF)

- Table: SCF results (statistics of energy error, convg. iter., and time breakdowns (see log files)

```bash
mkdir Figures_time_breakdown

# Statistics of SCF results with different seed numbers
python parse_gospel_logs.py ./ \
    -p "../expt.CNT_6_0.etol1e-6.dtol1e-4.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-6.dtol1e-4.json \
    &> Figures_time_breakdown/output.CNT_6_0.etol1e-6.dtol1e-4.log
python parse_gospel_logs.py ./ \
    -p "../expt.MgO_1x1x2.etol1e-6.dtol1e-4.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-6.dtol1e-4.json \
    &> Figures_time_breakdown/output.MgO_1x1x2.etol1e-6.dtol1e-4.log
python parse_gospel_logs.py ./ \
    -p "../expt.Si_diamond_2x2x1.etol1e-6.dtol1e-4.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-6.dtol1e-4.json \
    &> Figures_time_breakdown/output.Si_diamond_2x2x1.etol1e-6.dtol1e-4.log

python parse_gospel_logs.py ./ \
    -p "../expt.CNT_6_0.etol1e-7.dtol1e-5.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-7.dtol1e-5.json \
    &> Figures_time_breakdown/output.CNT_6_0.etol1e-7.dtol1e-5.log
python parse_gospel_logs.py ./ \
    -p "../expt.MgO_1x1x2.etol1e-7.dtol1e-5.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-7.dtol1e-5.json \
    &> Figures_time_breakdown/output.MgO_1x1x2.etol1e-7.dtol1e-5.log
python parse_gospel_logs.py ./ \
    -p "../expt.Si_diamond_2x2x1.etol1e-7.dtol1e-5.seed4*/*.log" \
    --show-timers \
    --timer-json Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-7.dtol1e-5.json \
    &> Figures_time_breakdown/output.Si_diamond_2x2x1.etol1e-7.dtol1e-5.log

# Plot time breakdown figures from json files
python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-6.dtol1e-4.json \
    -o Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-6.dtol1e-4.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend
python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-6.dtol1e-4.json \
    -o Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-6.dtol1e-4.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend
python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-6.dtol1e-4.json \
    -o Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-6.dtol1e-4.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend

python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-7.dtol1e-5.json \
    -o Figures_time_breakdown/time_breakdown.CNT_6_0.etol1e-7.dtol1e-5.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend
python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-7.dtol1e-5.json \
    -o Figures_time_breakdown/time_breakdown.MgO_1x1x2.etol1e-7.dtol1e-5.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend
python plot_from_json.py \
    Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-7.dtol1e-5.json \
    -o Figures_time_breakdown/time_breakdown.Si_diamond_2x2x1.etol1e-7.dtol1e-5.svg \
    --methods DP MP_scheme1 DP_SP4precond MP_scheme1_BF164precond \
    --labels 'FP64' 'MP1' 'MP6' 'MP1*' \
    --separate-legend
```
