# Experiment: Operation Speed Benchmark

This directory contains scripts to benchmark the **execution time** of individual Hamiltonian operations and to evaluate the **speed and accuracy** of various floating point (FP) precisions for common numerical computations.

## 1. Hamiltonian Operation Time Benchmark (`test_op_time.py`)

The script `test_op_time.py` measures the **execution time of each operation** involved in Hamiltonian construction. This includes tasks such as projection, kinetic energy evaluation, and pseudopotential applications.

### Usage example

```bash
python test_op_time.py --filepath ../data/systems/BaTiO3_2x2x1.cif --pbc 1 1 1 \
       --nbands 100 --supercell 1 1 5 \
       --fp SP --use_dense_proj \
       --operation projection kinetic local nonlocal tensordot
```

🛠️ Notes
- Supercell and number of bands can be tuned to simulate different workloads.
- Supports multiple floating point types and backend options.



## 2. Automated Benchmark Runner (test.operation.sh)

The shell script `test.operation.sh` automates batch execution of `test_op_time.py` across:
- Multiple supercell sizes
- Floating point modes (DP, SP, TF32, HP, BF16)

▶️ To Run:
```bash
source ./test.operation.sh
```


## 3. Floating Point Speed & Accuracy Benchmark (compare_fp_speed_and_accuracy.py)

This script benchmarks the speed and accuracy of different floating point precisions (FP64, FP32, FP16, and TF32) on common operations such as:

- Dense matrix multiplication
- Sparse-dense multiplication
- Tensordot operations 


### Usage example

```bash
# dense-dense matrix multiplication
python compare_fp_speed_and_accuracy.py --allow_tf32

# Sparse-dense matrix multiplication
python compare_fp_speed_and_accuracy.py --allow_tf32 --sparse

# tensordot operation
python compare_fp_speed_and_accuracy.py --allow_tf32 --operation tensordot --N 100 --M 100 --K 100 --batch 100
```

---

## 📁 File Summary

| File name                      | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `test_op_time.py`             | Measures timing for individual Hamiltonian operations   |
| `test.operation.sh`           | Automates batch benchmarking over precision/supercells  |
| `compare_fp_speed_and_accuracy.py` | Benchmarks precision for common numerical operations |


