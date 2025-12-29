


```bash
DEVICE="A100"
# DEVICE="RTX4090"
python plot_performance.py ../results/MgO_1x1x2.${DEVICE}.log \
    --xscale linear \
    --font-size-title 20 \
    --font-size-label 16 \
    --show \
    --font-size-tick 14 \
    --material MgO_1x1x2 \
    --no-dp-time \
    --dp-comparison \
    --use-dense-proj \
    --xscale linear \
    --yscale-dp-time log
```

