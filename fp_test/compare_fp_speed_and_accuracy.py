import argparse
import os
import time
import torch
import numpy as np
import scipy.sparse as sparse  # sparse 관련 연산을 위해 추가


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare speed and accuracy across FP64, FP32, FP16, and TF32 for matrix multiplication."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=5000,
        help="Matrix size: (N, M) x (M, K) (default: 5000)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=5000,
        help="Matrix size: (N, M) x (M, K) (default: 5000)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5000,
        help="Matrix size: (N, M) x (M, K) (default: 5000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=300,
        help="number of batch for my_tensordot test (default: 300)",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=10,
        help="Number of iterations for benchmarking (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the comparison on (default: cuda).",
    )
    parser.add_argument(
        "--omp",
        type=int,
        default=1,
        help="Number of OpenMP threads to use for NumPy and PyTorch on CPU.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable toggling TF32 (only relevant if device=cuda).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed number (default: 42)."
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor (default: 1.0)."
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse-dense multiplication instead of dense-dense.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.9,
        help="Sparsity of sparse matrix (default: 0.9).",
    )
    parser.add_argument(
        "--operation",
        type=str,
        default="matmul",
        choices=["matmul", "tensordot"],
        help="Operation type for dense multiplication: 'matmul' or 'tensordot' (default: matmul)",
    )
    return parser.parse_args()


def time_function(func, num_iter=10, warmup=3, device="cpu"):
    """
    A helper function to benchmark execution time of a given 'func'.
    If device == 'cuda', use CUDA events for measuring time. Otherwise use time.time().
    warmup: number of warm-up calls before actual timing.
    """
    # 워밍업
    for _ in range(warmup):
        func()
        if device == "cuda":
            torch.cuda.synchronize()

    # 실제 측정
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iter):
            func()
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0  # 밀리초 -> 초
    else:
        start = time.time()
        for _ in range(num_iter):
            func()
        elapsed = time.time() - start

    return elapsed


def init_sparse_matrix(n, m, sparsity, dtype=np.float64):
    """
    Initialize a random sparse matrix of shape (n, m).

    Parameters:
        n (int): 행렬의 행 개수.
        m (int): 행렬의 열 개수.
        sparsity (float): 0과 1 사이의 값으로, 0은 밀집, 1은 완전 희소.
        dtype (data-type): 행렬의 데이터 타입.

    Returns:
        scipy.sparse.csr_matrix: 생성된 sparse 행렬.
    """
    # 밀도는 non-zero 비율로, 1 - sparsity를 사용합니다.
    density = 1 - sparsity

    # random sparse matrix 생성 (data_rvs: non-zero 값 생성 함수)
    matrix = sparse.random(
        n,
        m,
        density=density,
        data_rvs=lambda s: np.random.randn(s).astype(dtype),
        format="csr",
        dtype=dtype,
    )
    return matrix


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


def compare_accuracy_and_speed(
    N,
    M,
    K,
    batch,
    num_iter,
    device="cuda",
    allow_tf32_flag=False,
    scale=1.0,
    sparse_mode=False,
    sparsity=0.9,
    operation="matmul",
):
    """
    FP64, FP32, FP16, TF32 연산의 속도와 정확도를 비교합니다.

    dense_mode일 때:
      1. FP64 dense 행렬 A_t64를 생성한 후,
      2. ref_result = A_t64.T @ A_t64 (dense‑dense 곱셈)을 구합니다.
      3. 각 낮은 정밀도에 대해 연산 수행 시간과 FP64 대비 상대 오차를 계산합니다.

    sparse_mode일 때:
      1. dense 행렬 A_dense (shape: N x M)와 sparse 행렬 A_sparse (shape: N x N)를 각각 생성한 후,
      2. ref_result = A_sparse_torch @ A_t64_dense (sparse‑dense 곱셈)을 구합니다.
      3. 각 정밀도별로 두 행렬을 해당 dtype으로 변환하여 연산 후 상대 오차를 계산합니다.
    """
    if not sparse_mode:
        if operation == "matmul":
            benchmark_dense_matmul(N, M, K, num_iter, device, allow_tf32_flag, scale)
        elif operation == "tensordot":
            benchmark_dense_tensordot(
                N, M, K, batch, num_iter, device, allow_tf32_flag, scale
            )
        else:
            raise ValueError(f"Unknown operation mode: {operation}")
    else:
        benchmark_sparse_matmul(
            N, M, K, num_iter, device, allow_tf32_flag, scale, sparsity
        )


def benchmark_sparse_matmul(
    N, M, K, num_iter, device, allow_tf32_flag, scale, sparsity
):
    # ----- Sparse 모드 -----
    print("Running sparse-dense multiplication test.")
    print(f"Sparse matrix shape (sparsity={sparsity}): {N} x {M}")
    print(f"Dense matrix shape: {M} x {K}")

    # 1) sparse 행렬 (FP64) 생성 (A_sparse: shape N x N)
    A_sparse_np_64 = init_sparse_matrix(N, M, sparsity, np.float64)
    A_sparse_torch_64 = scipy_to_torch_sparse(A_sparse_np_64).to(
        device=device, dtype=torch.float64
    )

    # 2) dense 행렬 (FP64) 생성 (A_dense: shape N x M)
    A_dense_np_64 = np.random.randn(M, K).astype(np.float64)
    A_t64_dense = torch.from_numpy(A_dense_np_64).to(device=device, dtype=torch.float64)

    # 3) FP64 기준 sparse-dense matmul: A_sparse_torch_64 @ A_t64_dense
    with torch.no_grad():
        ref_start = time.time()
        ref_result = A_sparse_torch_64 @ A_t64_dense
        if device == "cuda":
            torch.cuda.synchronize()
        ref_time = time.time() - ref_start

    print(
        "Reference (FP64) sparse-dense matmul done. (Not part of main benchmark timing)"
    )

    # TF32 기본값 저장
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_flag
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_dtype(dtype_name):
        """
        sparse 모드에서 주어진 dtype에 대해:
            - dense 행렬과 sparse 행렬을 해당 dtype으로 변환,
            - scale 적용 후 matmul 수행,
            - 실행시간과 FP64 기준 결과(ref_result)와의 상대 오차 계산.
        """
        if dtype_name.lower() == "fp64":
            A_dense_local = A_t64_dense
            A_sparse_local = A_sparse_torch_64
        elif dtype_name.lower() == "fp32":
            A_dense_local = A_t64_dense.float()
            A_sparse_local = A_sparse_torch_64.to(dtype=torch.float32)
        elif dtype_name.lower() == "fp16":
            A_dense_local = A_t64_dense.half()
            A_sparse_local = A_sparse_torch_64.to(dtype=torch.float16)
        elif dtype_name.lower() == "tf32":
            A_dense_local = A_t64_dense.float()
            A_sparse_local = A_sparse_torch_64.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown dtype_name: {dtype_name}")

        if scale != 1.0:
            A_dense_local = A_dense_local * scale
            A_sparse_local = A_sparse_local * scale

        # TF32 토글
        old_tf32_setting = torch.backends.cuda.matmul.allow_tf32
        if device == "cuda":
            if dtype_name.lower() == "tf32":
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = False

        def matmul_op():
            if scale != 1.0:
                return (A_sparse_local @ A_dense_local) / (scale**2)
            else:
                return A_sparse_local @ A_dense_local

        elapsed = time_function(matmul_op, num_iter=num_iter, warmup=3, device=device)
        with torch.no_grad():
            out = matmul_op()
        rel_error = (out.double() - ref_result).norm(p="fro") / ref_result.norm(p="fro")

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_setting

        return elapsed, rel_error.item()

    results = {}
    dtypes = ["fp64", "fp32"]
    if device == "cuda":
        dtypes.append("fp16")
        if allow_tf32_flag:
            dtypes.append("tf32")

    for dt in dtypes_to_test:
        t, err = test_dtype(dt)
        results[dt] = (t, err)

    print("\n==== Summary (Sparse-Dense matmul benchmark) ====")
    for dt in dtypes_to_test:
        t, err = results[dt]
        print(f"{dt.upper():>5} | Time: {t:.4f} s | Rel. Error vs FP64: {err:.3e}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


def benchmark_dense_matmul(N, M, K, num_iter, device, allow_tf32_flag, scale):
    print(f"Using dense-dense multiplication: ({N}, {M}) x ({M}, {K})")
    print("Running dense-dense matmul test.")
    A_t64 = torch.randn(N, M, dtype=torch.float64).to(device)
    B_t64 = torch.randn(M, K, dtype=torch.float64).to(device)

    with torch.no_grad():
        ref_result = A_t64 @ B_t64
        if device == "cuda":
            torch.cuda.synchronize()
    print("Reference (FP64) matmul computed.")

    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (
        allow_tf32_flag if device == "cuda" else False
    )

    def test_dtype(dtype_name):
        if dtype_name.lower() == "fp64":
            A_local, B_local = A_t64, B_t64
        elif dtype_name.lower() in ["fp32", "tf32"]:
            A_local, B_local = A_t64.float(), B_t64.float()
        elif dtype_name.lower() == "fp16":
            A_local, B_local = A_t64.half(), B_t64.half()
        else:
            raise ValueError(f"Unknown dtype: {dtype_name}")

        if scale != 1.0:
            A_local = A_local * scale
            B_local = B_local * scale

        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = (
                True if dtype_name.lower() == "tf32" else False
            )

        def op():
            result = A_local @ B_local
            if scale != 1.0:
                result = result / (scale**2)
            return result

        elapsed = time_function(op, num_iter=num_iter, warmup=3, device=device)
        with torch.no_grad():
            out = op()
        rel_error = (out.double() - ref_result).norm(p="fro") / ref_result.norm(p="fro")

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

        return elapsed, rel_error.item()

    results = {}
    dtypes = ["fp64", "fp32"]
    if device == "cuda":
        dtypes.append("fp16")
        if allow_tf32_flag:
            dtypes.append("tf32")

    for dt in dtypes:
        t, err = test_dtype(dt)
        results[dt] = (t, err)

    print("\n==== Summary (Dense-Dense matmul benchmark) ====")
    for dt in dtypes:
        t, err = results[dt]
        print(f"{dt.upper():>5} | Time: {t:.4f} s | Rel. Error vs FP64: {err:.3e}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


def benchmark_dense_tensordot(N, M, K, batch, num_iter, device, allow_tf32_flag, scale):
    print("Running custom tensordot (my_tensordot) test.")
    print(
        f"tensordot shape: ({N}, {M}, {K}, {batch}) x ({N}, {N}) x ({M}, {M}) x ({K}, {K})"
    )
    A_t64 = torch.randn(N, M, K, batch, dtype=torch.float64).to(device)
    Fx_t64 = torch.randn(N, N, dtype=torch.float64).to(device)
    Fy_t64 = torch.randn(M, M, dtype=torch.float64).to(device)
    Fz_t64 = torch.randn(K, K, dtype=torch.float64).to(device)

    def my_tensordot(A, Fx, Fy, Fz):
        tmp = torch.tensordot(A, Fx, ([0], [0]))
        tmp = torch.tensordot(tmp, Fy, ([0], [0]))
        tmp = torch.tensordot(tmp, Fz, ([0], [0]))
        return tmp

    with torch.no_grad():
        ref_result = my_tensordot(A_t64, Fx_t64, Fy_t64, Fz_t64)
        if device == "cuda":
            torch.cuda.synchronize()
    print("Reference (FP64) my_tensordot computed.")

    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (
        allow_tf32_flag if device == "cuda" else False
    )

    def test_dtype(dtype_name):
        if dtype_name.lower() == "fp64":
            A_local, Fx_local, Fy_local, Fz_local = A_t64, Fx_t64, Fy_t64, Fz_t64
        elif dtype_name.lower() in ["fp32", "tf32"]:
            A_local = A_t64.float()
            Fx_local = Fx_t64.float()
            Fy_local = Fy_t64.float()
            Fz_local = Fz_t64.float()
        elif dtype_name.lower() == "fp16":
            A_local = A_t64.half()
            Fx_local = Fx_t64.half()
            Fy_local = Fy_t64.half()
            Fz_local = Fz_t64.half()
        else:
            raise ValueError(f"Unknown dtype: {dtype_name}")

        if scale != 1.0:
            A_local = A_local * scale
            Fx_local = Fx_local * scale
            Fy_local = Fy_local * scale
            Fz_local = Fz_local * scale

        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = (
                True if dtype_name.lower() == "tf32" else False
            )

        def op():
            result = my_tensordot(A_local, Fx_local, Fy_local, Fz_local)
            if scale != 1.0:
                result = result / (scale**4)
            return result

        elapsed = time_function(op, num_iter=num_iter, warmup=3, device=device)
        with torch.no_grad():
            out = op()
        rel_error = (out.double() - ref_result).norm(p="fro") / ref_result.norm(p="fro")

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

        return elapsed, rel_error.item()

    results = {}
    dtypes = ["fp64", "fp32"]
    if device == "cuda":
        dtypes.append("fp16")
        if allow_tf32_flag:
            dtypes.append("tf32")

    for dt in dtypes:
        t, err = test_dtype(dt)
        results[dt] = (t, err)

    print("\n==== Summary (tensordot benchmark) ====")
    for dt in dtypes:
        t, err = results[dt]
        print(f"{dt.upper()}: Time: {t:.4f} s, Relative Error: {err:.3e}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


def main_accuracy_test():
    """
    메인 함수: 설정값 출력 후 정확도 및 속도 비교 실행.
    """
    args = parse_arguments()
    print(f"args={args}")

    if torch.cuda.is_available():
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # CPU 사용 시 OpenMP 스레드 수 설정
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    torch.set_num_threads(args.omp)

    # 재현성을 위해 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Number of iterations: {args.num_iter}")
    print(f"Device: {args.device}")
    print(f"allow_tf32_flag={args.allow_tf32}")
    print(f"Using OMP_NUM_THREADS={args.omp}")

    compare_accuracy_and_speed(
        N=args.N,
        M=args.M,
        K=args.K,
        batch=args.batch,
        num_iter=args.num_iter,
        device=args.device,
        allow_tf32_flag=args.allow_tf32,
        scale=args.scale,
        sparse_mode=args.sparse,
        sparsity=args.sparsity,
        operation=args.operation,
    )


if __name__ == "__main__":
    main_accuracy_test()
