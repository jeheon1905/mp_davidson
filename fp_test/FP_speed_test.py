import argparse
import os
import time
import torch
import numpy as np
import scipy.sparse as sparse

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark dense-dense and sparse-dense matrix multiplications using NumPy, SciPy, and PyTorch."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=5000,
        help="Matrix size: N x N for sparse, N x M for dense (default to 5000)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=5000,
        help="Matrix size: N x N for sparse, N x M for dense (default to 5000)",
    )
    parser.add_argument(
        "--num_iter", type=int, default=10, help="Number of iterations for benchmarking"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type for the matrices",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Include sparse-dense matrix multiplication tests",
    )
    parser.add_argument(
        "--omp",
        type=int,
        default=1,
        help="Number of OpenMP threads to use for NumPy, SciPy, and PyTorch.",
    )
    return parser.parse_args()


def init_dense_matrix(m, n, dtype):
    """Initialize a random dense matrix of shape (m x n)."""
    return np.random.randn(m, n).astype(dtype)


def init_sparse_matrix(n, dtype):
    """
    Initialize a random sparse symmetric matrix of shape (n x n).
    Example uses diagonals with random values.
    """
    B = np.zeros((n, n), dtype=dtype)
    for i in range(5):
        # Add random diagonals
        diag_vals = np.random.randn(n - i).astype(dtype)
        B[np.arange(n - i), np.arange(i, n)] += diag_vals  # upper diag
    # Make symmetric
    B = (B + B.T) / 2
    return sparse.csr_matrix(B)


def scipy_to_torch_sparse(A_scipy):
    """
    Convert a scipy.sparse matrix to a torch.sparse Tensor.
    Supports csr_matrix and coo_matrix.
    """
    if isinstance(A_scipy, sparse.csr_matrix):
        A_torch = torch.sparse_csr_tensor(
            A_scipy.indptr,
            A_scipy.indices,
            A_scipy.data,
            A_scipy.shape,
        )
    elif isinstance(A_scipy, sparse.coo_matrix):
        A_torch = torch.sparse_coo_tensor(
            np.vstack((A_scipy.row, A_scipy.col)),
            A_scipy.data,
            A_scipy.shape,
        )
    else:
        raise NotImplementedError(f"Unsupported type(A_scipy) = {type(A_scipy)}")
    return A_torch


def time_function(func, num_iter=10, warmup=3, device="cpu"):
    """
    A helper function to benchmark execution time of a given 'func'.
    If device == 'cuda', use CUDA events for measuring time. Otherwise use time.time().
    warmup: number of warm-up calls before actual timing.
    """
    # Warmup
    for _ in range(warmup):
        func()
        if device == "cuda":
            torch.cuda.synchronize()

    # Actual timing
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iter):
            func()
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> seconds
    else:
        start = time.time()
        for _ in range(num_iter):
            func()
        elapsed = time.time() - start

    return elapsed


def benchmark_numpy_dense(M, num_iter):
    """
    Benchmark dense-dense multiplication using NumPy.
    """
    # print("\n[NumPy Dense-Dense on CPU]")
    M_T = M.T.copy()

    # Define the multiplication function
    def matmul():
        _ = M_T @ M

    elapsed = time_function(matmul, num_iter=num_iter, warmup=3, device="cpu")
    print(f"[NumPy Dense-Dense on CPU]: {elapsed:.4f} sec")


def benchmark_torch_dense(M, num_iter, device="cpu"):
    """
    Benchmark dense-dense multiplication using PyTorch (CPU or CUDA).
    """
    T = torch.from_numpy(M).to(device)
    T_T = T.T.clone()

    # Define the multiplication function
    def matmul():
        _ = T_T @ T

    elapsed = time_function(matmul, num_iter=num_iter, warmup=3, device=device)
    # print(f"PyTorch time: {elapsed:.4f} sec")
    print(f"[PyTorch Dense-Dense on {device.upper()}]: {elapsed:.4f} sec")


def benchmark_scipy_sparse_dense(A_dense, A_sparse, num_iter):
    """
    Benchmark sparse-dense multiplication using SciPy.
    """
    # Define the multiplication function
    def matmul():
        _ = A_sparse @ A_dense

    elapsed = time_function(matmul, num_iter=num_iter, warmup=3, device="cpu")
    print(f"[SciPy Sparse-Dense]: {elapsed:.4f} sec")


def benchmark_torch_sparse_dense(A_dense, A_sparse, num_iter, device="cpu"):
    """
    Benchmark sparse-dense multiplication using PyTorch (CPU or CUDA).
    """
    # Convert
    A_torch = torch.from_numpy(A_dense).to(device)
    A_sparse_torch = scipy_to_torch_sparse(A_sparse).to(device)

    # Define the multiplication function
    def matmul():
        _ = A_sparse_torch @ A_torch

    elapsed = time_function(matmul, num_iter=num_iter, warmup=3, device=device)
    print(f"[PyTorch Sparse-Dense on {device.upper()}]: {elapsed:.4f} sec")


def main():
    args = parse_arguments()
    print(f"args={args}")

    if torch.cuda.is_available():
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set OpenMP threads for NumPy, SciPy, and PyTorch
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    os.environ["MKL_NUM_THREADS"] = str(1)
    os.environ["NUMEXPR_NUM_THREADS"] = str(1)
    torch.set_num_threads(args.omp)
    print(f"Using OMP_NUM_THREADS={args.omp}")

    print(f"Dense matrix size: {args.N} x {args.M}")
    print(f"Sparse matrix size: {args.N} x {args.N}")
    print(f"Data type: {args.dtype}")
    print(f"Number of iterations: {args.num_iter}")

    # --- Create Dense Matrix ---
    A_dense = init_dense_matrix(args.N, args.M, args.dtype)

    # --- Benchmark: Dense-Dense (NumPy and Torch) ---
    benchmark_numpy_dense(A_dense, args.num_iter)
    benchmark_torch_dense(A_dense, args.num_iter, device="cpu")
    if torch.cuda.is_available():
        benchmark_torch_dense(A_dense, args.num_iter, device="cuda")

    # --- Optionally Benchmark: Sparse-Dense ---
    if args.sparse:
        A_sparse = init_sparse_matrix(args.N, args.dtype)
        nnz_ratio = A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1]) * 100
        print(f"Nonzero ratio: {nnz_ratio:.2f}%")

        benchmark_scipy_sparse_dense(A_dense, A_sparse, args.num_iter)
        benchmark_torch_sparse_dense(A_dense, A_sparse, args.num_iter, device="cpu")

        if torch.cuda.is_available():
            benchmark_torch_sparse_dense(
                A_dense, A_sparse, args.num_iter, device="cuda"
            )


if __name__ == "__main__":
    main()
