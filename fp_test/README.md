# Speed and Accuracy Test of FP Arithmetic

This script benchmarks the speed and accuracy of different floating point precisions (FP64, FP32, FP16, and TF32) on common operations such as:

- Dense matrix multiplication
- Sparse-dense multiplication
- Tensordot operations 


## Usage Examples

```bash
# dense-dense matrix multiplication
python compare_fp_speed_and_accuracy.py --allow_tf32

# Sparse-dense matrix multiplication
python compare_fp_speed_and_accuracy.py --allow_tf32 --sparse

# tensordot operation
python compare_fp_speed_and_accuracy.py --allow_tf32 --operation tensordot --N 100 --M 100 --K 100 --batch 100
```