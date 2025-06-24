# Test the time of different operations in GOSPEL
import argparse
import datetime
import os
import torch

from gospel import GOSPEL
from gospel.Hamiltonian import Hamiltonian
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.util import Timer, set_global_seed
from gospel.Eigensolver.precondition import create_preconditioner

from mp_davidson.utils import make_atoms, get_git_commit, block_all_print


def init_gospel(args):
    # NOTE: Creating the system
    atoms = make_atoms(args.filepath, args.supercell, args.pbc)
    atoms.center()

    if args.upf_files is None:
        assert args.pp_type in ["SG15", "ONCV", "TM", "NNLP"]
        pp_path = f"../data/pseudopotentials/{args.pp_type}/"
        if args.pp_type == "SG15":
            pp_prefix = "_ONCV_PBE-1.2.upf"
        elif args.pp_type == "ONCV":
            pp_prefix = ".upf"
        elif args.pp_type == "TM":
            pp_prefix = ".pbe-n-nc.UPF"
        elif args.pp_type == "NNLP":
            pp_prefix = ".nnlp.UPF"
        else:
            raise NotImplementedError
        symbols = set(atoms.get_chemical_symbols())
        upf_files = [pp_path + symbol + pp_prefix for symbol in symbols]
    else:
        upf_files = args.upf_files

    # Make GOSPEL calculator
    calc = GOSPEL(
        use_cuda=args.use_cuda,
        use_dense_kinetic=args.use_dense_kinetic,
        precond_type=None,
        eigensolver=None,
        grid={"spacing": args.spacing},
        pp={
            "upf": upf_files,
            "filtering": True,
            "use_dense_proj": args.use_dense_proj,
        },
        nbands=args.nbands,
        hamiltonian={"use_dense_kinetic": args.use_dense_kinetic},
    )
    atoms.calc = calc
    calc.initialize(atoms)

    calc.hamiltonian = Hamiltonian(
        calc.nspins,
        calc.nbands,
        calc.grid,
        calc.kpoint,
        calc.pp,
        calc.poisson_solver,
        calc.xc_functional,
        calc.eigensolver,
        use_dense_kinetic=calc.parameters.hamiltonian.get("use_dense_kinetic"),
        multi_dtype=calc.parameters.hamiltonian.get("multi_dtype"),
        device=PH.get_device(),
    )
    return atoms, calc


def main(args: argparse.Namespace) -> None:
    atoms, calc = init_gospel(args)

    dtype = {
        "DP": torch.float64,
        "SP": torch.float32,
        "HP": torch.float16,
        "BF16": torch.bfloat16,
    }[args.fp]
    device = PH.get_device()
    nbands = calc.nbands
    ngpts = calc.grid.ngpts

    X = torch.randn(ngpts, nbands).to(device, dtype)

    for op in args.operation:
        if op == "projection":
            matvec = lambda x: x.T @ x
        elif op == "kinetic":
            i_k = 0
            T_s = calc.hamiltonian.get_kinetic_operator(i_k, dtype)
            matvec = lambda x: T_s @ x
        elif op == "local":
            V_loc = torch.randn(ngpts).reshape(-1, 1).to(device, dtype)
            matvec = lambda x: V_loc * x
        elif op == "nonlocal":
            i_k = 0
            V_nl = calc.hamiltonian.pp.get_nonlocal_op(i_k, dtype)
            # matvec = lambda x: x + V_nl(x)
            matvec = lambda x: V_nl(x)
        elif op == "tensordot":
            precond = create_preconditioner(
                "gapp", calc.grid, args.use_cuda, options={"fp": args.fp}
            )
            kernel = precond.solver.batch_compute_potential2
            matvec = lambda x: kernel(x)
        else:
            raise NotImplementedError

        dtype_flag = "TF32" if args.fp == "SP" and args.allow_tf32 else args.fp
        with Timer.track(f"Operation: {op} ({dtype_flag})", True, False):
            _ = matvec(X)
    return


if __name__ == "__main__":
    # NOTE: Parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        type=str,
        nargs="+",
        help="operation type(s)",
        required=True,
        choices=["projection", "kinetic", "nonlocal", "local", "tensordot"],
    )
    parser.add_argument(
        "--filepath", type=str, help="file path (cif or xyz)", required=True
    )
    parser.add_argument(
        "--supercell", type=int, nargs="+", default=[1, 1, 1], help="supercell"
    )
    parser.add_argument(
        "--pbc",
        type=int,
        nargs="+",
        help="periodic boundary condition of each axis. e.g., --pbc 0 0 1",
        required=True,
    )
    parser.add_argument("--pp_type", type=str, default="SG15")
    parser.add_argument("--upf_files", type=str, nargs="+", default=None)
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.15,
        help="grid spacing (ang), default to 0.15",
    )
    parser.add_argument("--nbands", type=int, default=None, help="nbands")
    parser.add_argument(
        "--use_dense_kinetic",
        action="store_true",
        help="whether to construct kinetic op. with dense matmul",
    )
    parser.add_argument(
        "--use_dense_proj",
        action="store_true",
        help="whether to construct nonlocal projections with dense matmul",
    )
    parser.add_argument(
        "--fp",
        type=str,
        choices=["DP", "SP", "HP", "BF16"],
        default="DP",
        help="floating point type",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="whether to allow tensorfloat32 datatype",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--use_cuda",
        type=int,
        default=1,
        help="whether to use CUDA",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="whether to warm up the GPU (0=False, 1=True)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for starting vectors"
    )
    args = parser.parse_args()
    print(f"datetime: {datetime.datetime.now()}")
    print(f"GOSPEL git commit: {get_git_commit('gospel')}")
    print(f"mp_davidson git commit: {get_git_commit('mp_davidson')}")
    print("args=", args)

    # ParallelHelper initialization
    PH.init_from_env(args.use_cuda)
    set_global_seed(args.seed + PH.rank)

    torch.set_num_threads(os.cpu_count() if args.threads is None else args.threads)
    if torch.cuda.is_available():
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        if not args.use_dense_kinetic:
            print("Warning: 'allow_tf32=True' but 'use_dense_kinetic' is set to False.")
        if not args.use_dense_proj:
            print("Warning: 'allow_tf32=True' but 'use_dense_proj' is set to False.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    # Warm up
    if args.warmup:
        with block_all_print():
            main(args)
        print("Warm-up finished")

    Timer.reset()
    main(args)

    # Print each time
    # Build a list of (label, total, count)
    items = []
    for label, data in Timer._records.items():
        if "Operation: " not in label:
            continue
        total_time = data["total"]
        c = data["count"]
        items.append((label, total_time, c))

    print("\n======================== Timer Summary ========================")
    print(f"{'Label':40s} | {'Total(s)':>12} | {'Count':>5}")
    print("-" * 64)
    for label, total_time, c in items:
        print(f"{label:40s} | {total_time:12.6f} | {c:5d}")
    print("-" * 64 + "\n")
    dtype_flag = "TF32" if args.fp == "SP" and args.allow_tf32 else args.fp
    print(f"supercell: {args.supercell}")
    print(f"dtype: {dtype_flag}")
    print(f"use_dense_proj: {args.use_dense_proj}")
    print(f"use_dense_kinetic: {args.use_dense_kinetic}")
