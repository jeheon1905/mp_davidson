# Diagonalization of Fixed Hamiltonian

This directory contains experiments related to the **diagonalization of fixed Hamiltonians**.

Two types of tests are conducted:
1. **Convergence Tests**: Analyze the convergence history of eigenvalues and residual norms across iterations.
2. **Speed Tests**: Benchmark the execution time under different computational settings.

---

## Config Files

This directory includes configuration files for running **diagonalization tests** on different material systems. Each system has two types of config files:

1. **Convergence test config** (`config.XXX.sh`)
2. **Speed test config** (`config.XXX.speed.sh`)

### 🧪 Available Systems
- CNT (6, 0)
- Si (diamond)
- BaTiO₃ (perovskite)

---

## Usage Example

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

