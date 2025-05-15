# Diagonalization of fixed Hamiltonian

## Config files
- "./config.CNT_6_0.sh": config file for CNT (6, 0)
- "./config.Si_diamond.sh": config file for Si diamond
- "./config.BaTiO3.sh": config file for BaTiO3


```bash
# 1. Convergence test
source run_test.sh Experiment_Fixed_Hamiltonian/config.CNT_6_0.sh
source run_test.sh Experiment_Fixed_Hamiltonian/config.Si_diamond.sh
source run_test.sh Experiment_Fixed_Hamiltonian/config.BaTiO3.sh

# Plot convergence history
source plot.sh Experiment_Fixed_Hamiltonian/config.CNT_6_0.sh
source plot.sh Experiment_Fixed_Hamiltonian/config.Si_diamond.sh
source plot.sh Experiment_Fixed_Hamiltonian/config.BaTiO3.sh

# 2. Speed test
source run_test.speed.sh Experiment_Fixed_Hamiltonian/config.CNT_6_0.speed.sh
source run_test.speed.sh Experiment_Fixed_Hamiltonian/config.Si_diamond.speed.sh
source run_test.speed.sh Experiment_Fixed_Hamiltonian/config.BaTiO3.speed.sh
```
