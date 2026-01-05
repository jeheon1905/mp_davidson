import argparse
import datetime
import os
import gc
import sys
import torch
import numpy as np

from gospel import GOSPEL
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.Eigensolver.precondition import create_preconditioner
from gospel.util import Timer, set_global_seed
from mp_davidson.utils import make_atoms, get_git_commit, block_all_print


def _cuda_sync():
    """
    Synchronize the current CUDA device.

    This forces all queued asynchronous CUDA operations to complete,
    ensuring that subsequent GPU memory statistics (allocated / peak)
    reflect the actual state after computation.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _mem_reset(args: argparse.Namespace) -> None:
    """
    Reset CUDA peak memory statistics if memory measurement is enabled.

    This should be called immediately before a code section for which
    peak GPU memory usage needs to be measured. The reset is only
    performed when:
      - --measure-mem is enabled,
      - CUDA is requested and available.
    """
    if getattr(args, "measure_mem", False) and args.use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _mem_report(tag: str, args: argparse.Namespace) -> None:
    """
    Print current and peak GPU memory usage for a labeled code section.

    Reports:
      - current allocated memory,
      - current reserved memory (CUDA caching allocator),
      - peak allocated memory since the last reset,
      - peak reserved memory since the last reset.

    The report is printed only when:
      - --measure-mem is enabled,
      - CUDA is requested and initialized.

    Parameters
    ----------
    tag : str
        Label identifying the computation stage (e.g. 'after initialize',
        'davidson diagonalization').
    args : argparse.Namespace
        Parsed command-line arguments containing the 'measure_mem' flag.
    """
    if not (getattr(args, "measure_mem", False) and args.use_cuda):
        return
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return
    _cuda_sync()
    cur_alloc = torch.cuda.memory_allocated()
    cur_resv = torch.cuda.memory_reserved()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()
    print(
        f"[GPU MEM] {tag} | "
        f"cur_alloc={cur_alloc/1024**3:.2f}GiB, cur_resv={cur_resv/1024**3:.2f}GiB | "
        f"peak_alloc={peak_alloc/1024**3:.2f}GiB, peak_resv={peak_resv/1024**3:.2f}GiB"
    )


def main(args: argparse.Namespace) -> None:
    # NOTE: Creating the system
    atoms = make_atoms(args.filepath, args.supercell, args.pbc)
    atoms.center()
    print(atoms)

    # NOTE: Setting GOSPEL calculator
    # Set pseudopotential options
    if args.upf_files is None:
        assert args.pp_type in ["SG15", "ONCV", "TM", "NNLP"]
        pp_path = f"./data/pseudopotentials/{args.pp_type}/"
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

    # Make eigensolver option
    eigensolver = {
        "type": "parallel_davidson",
        "maxiter": args.diag_iter,
        "locking": False,
        "fill_block": False,
        "verbosity": args.verbosity,
        # "dynamic": args.dynamic,
        "use_MP": (args.fp == "MP"),
        "MP_dtype": args.MP_dtype,
        "MP_scheme": args.MP_scheme,
    }

    # Make GOSPEL calculator
    calc = GOSPEL(
        mixing={"what": args.scf_mixing},
        print_energies=True,
        use_cuda=args.use_cuda,
        use_dense_kinetic=args.use_dense_kinetic,
        precond_type=None,
        eigensolver=eigensolver,
        grid={"spacing": args.spacing},
        pp={
            "upf": upf_files,
            "filtering": True,
            "use_dense_proj": args.use_dense_proj,
        },
        xc={"type": "gga_x_pbe + gga_c_pbe"},
        convergence={
            "scf_maxiter": 100,
            "density_tol": args.scf_density_tol,
            "orbital_energy_tol": np.inf,
            "energy_tol": args.scf_energy_tol,  # only check this for SCF convergence
        },
        occupation={
            "smearing": "Fermi-Dirac",
            "temperature": args.temperature,
        },
        nbands=args.nbands,
        hamiltonian={
            "use_dense_kinetic": args.use_dense_kinetic,
            "multi_dtype": args.multi_dtype,
        },
        random_seed=args.seed,
    )
    atoms.calc = calc
    calc.initialize(atoms)

    _mem_report('after calc.initialize', args)
    _mem_reset(args)

    # NOTE: Make preconditioner
    precond_options = {
        "precond_type": "shift-and-invert",
        "grid": calc.grid,
        "use_cuda": args.use_cuda,
        "options": {
            "inner_precond": "gapp",
            "max_iter": args.precond_iter,
            "fp": args.precond_fp,
            "verbosityLevel": args.verbosity,
            "locking": False,
            "no_shift_thr": args.no_shift_thr,
        },
    }
    calc.eigensolver.preconditioner = create_preconditioner(**precond_options)

    # NOTE: SCF calculation
    if args.phase == "scf":
        _mem_reset(args)
        energy = atoms.get_potential_energy()
        _mem_report("SCF atoms.get_potential_energy", args)

        # Save the converged density
        if args.density_filename is not None:
            torch.save(
                calc.get_density(spin=slice(0, sys.maxsize)), args.density_filename
            )
            print(f"charge density file '{args.density_filename}' is saved.")
    elif args.phase == "fixed":
        # NOTE: Fixed Hamiltonian diagonalization
        from gospel.Hamiltonian import Hamiltonian
        from gospel.Eigensolver.ParallelDavidson import davidson

        # Initialize the electron density
        if args.density_filename is not None:
            density = torch.load(args.density_filename)
            density = density.reshape(1, -1).to(PH.get_device())
        else:
            print("Initializing the density...")
            density = calc.density.init_density()
        calc.density.set_density(density)

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
        calc.hamiltonian.update(calc.density)
        del density, calc.density, calc.kpoint, calc.poisson_solver, calc.xc_functional

        # Initialize eigenpair guess
        if args.guess_filename:
            val, vec = torch.load(
                args.guess_filename
            )  # val.shape=(nbands,), vec.shape=(ngpts, nbands)
            vec = np.array([[vec.T]], dtype=object)
            eigpair = (val, vec)
            calc.eigensolver.set_initial_eigenpair(
                eigpair, device=PH.get_device(), orthonormalize=True
            )
            del eigpair
        else:
            # Initialized with orthonormalized random vectors
            calc.eigensolver._initialize_guess(calc.hamiltonian)

        # NOTE: Set the floating point type for the starting vector
        if args.fp == "SP":
            init_vec_dtype = torch.float32
        elif args.fp == "HP":
            init_vec_dtype = torch.float16
        elif args.fp == "BF16":
            init_vec_dtype = torch.bfloat16
        elif args.fp == "MP":
            if args.MP_scheme in [1, 2, 4]:
                if args.MP_dtype == "SP":
                    init_vec_dtype = torch.float32
                elif args.MP_dtype == "HP":
                    init_vec_dtype = torch.float16
                elif args.MP_dtype == "BF16":
                    init_vec_dtype = torch.bfloat16
            else:
                init_vec_dtype = torch.float64
        else:
            init_vec_dtype = torch.float64

        calc.eigensolver._starting_vector[0, 0] = calc.eigensolver._starting_vector[
            0, 0
        ].to(init_vec_dtype)

        # Reset initialization time records (only measure diagonalization times)
        Timer.reset()
        _mem_reset(args)

        # Diagonalization
        results = davidson(
            A=calc.hamiltonian[0, 0],
            X=calc.eigensolver._starting_vector[0, 0],
            B=None,
            preconditioner=calc.eigensolver.preconditioner,
            tol=args.fixed_convg_tol,
            maxiter=args.diag_iter,
            nblock=args.nblock,
            locking=args.locking,
            fill_block=args.fill_block,
            verbosity=args.verbosity,
            retHistory=(args.retHistory is not None),
            skip_init_ortho=True,  # already orthonormalized at initialization
            timing=True,
            use_MP=(args.fp == "MP"),
            MP_dtype=args.MP_dtype,
            MP_scheme=args.MP_scheme,
            debug_recalc_convg_history=args.recalc_convg_history,
        )
        _mem_report('Fixed davidson diagonalization', args)
        del calc

        # NOTE: Save residual history
        if args.retHistory is not None:
            eigval, eigvec, eigHistory, resHistory = results
            print("Saving convergence history...")
            torch.save((eigHistory, resHistory), args.retHistory)
            print(f"{args.retHistory} is saved.")
        else:
            eigval, eigvec = results

        if args.save_eig:
            print("Saving eigpair...")
            eigval = eigval.cpu()
            eigvec = PH.merge(eigvec).cpu()
            eigpairs = (eigval, eigvec)
            filename = "eigpairs.pt"
            torch.save(eigpairs, filename)
            print(f"{filename} is saved.")
    else:
        raise NotImplementedError("Only support 'scf' or 'fixed'.")


if __name__ == "__main__":
    # NOTE: Parsing input arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--precond_iter",
        type=int,
        default=4,
        help="preconditioner's maxiter, default to 4",
    )
    parser.add_argument(
        "--no_shift_thr",
        type=float,
        default=10.0,
        help="Threshold for not shifting to states with large residues (defaut: 10.0).",
    )
    parser.add_argument(
        "--diag_iter", type=int, default=2, help="eigensolver's maxiter"
    )
    parser.add_argument("--phase", type=str, default="fixed", choices=["scf", "fixed"])
    parser.add_argument(
        "--density_filename",
        type=str,
        default=None,
        help="charge density filename to save (or for initialization)",
    )
    parser.add_argument(
        "--guess_filename", type=str, default=None, help="guess eigpair filename"
    )
    parser.add_argument(
        "--save_eig",
        type=int,
        default=0,
        help="whether to save eigenpairs. 0=False, 1=True",
    )
    parser.add_argument("--nbands", type=int, default=None, help="nbands")
    parser.add_argument(
        "--retHistory", type=str, default=None, help="save residual history?"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for starting vectors"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1160.45,
        help="temperature for smearing (default: 1160.45 K = 0.1 eV)",
    )
    parser.add_argument(
        "--scf_energy_tol",
        type=float,
        default=1e-6,
        help="scf convergence energy tolerance (Hartree / electron)",
    )
    parser.add_argument(
        "--scf_density_tol",
        # type=float,
        type=lambda x: np.inf if x.lower() == 'inf' else float(x),
        default=1e-4,
        help="SCF density tolerance (/electron); use 'inf' to disable",
    )
    parser.add_argument(
        "--scf_mixing",
        type=str,
        default="potential",
        choices=["density", "potential"],
        help="scf mixing type (default: potential)",
    )
    parser.add_argument(
        "--fixed_convg_tol",
        type=float,
        default=0.001,
        help="fixed Hamiltonian convergence residual norm tolerance",
    )
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
        choices=["DP", "SP", "HP", "BF16", "MP"],
        default="DP",
        help="floating point type",
    )
    parser.add_argument(
        "--precond_fp",
        type=str,
        choices=["DP", "SP", "HP", "BF16"],
        default="DP",
        help="floating point precision for preconditioner",
    )
    parser.add_argument(
        "--multi_dtype",
        type=str,
        nargs="+",
        default=None,
        help="multiple dtype for Hamiltonian",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="whether to allow tensorfloat32 datatype",
    )
    parser.add_argument(
        "--recalc_convg_history",
        action="store_true",
        help="whether to recalculate convergence history",
    )
    parser.add_argument(
        "--MP_dtype",
        type=str,
        choices=["SP", "HP", "BF16"],
        default="SP",
        help="mixed precision data type (default: SP)",
    )
    parser.add_argument(
        "--MP_scheme",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="mixed precision scheme",
    )
    parser.add_argument(
        "--nblock",
        type=int,
        default=2,
        help="number of blocks",
    )
    parser.add_argument(
        "--locking",
        action="store_true",
        help="whether to use locking option",
    )
    parser.add_argument(
        "--fill_block",
        action="store_true",
        help="whether to use fill_block option",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="verbosity level",
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
        "--measure-mem",
        action="store_true",
        help="Measure and report GPU peak memory (allocated/reserved) for key stages",
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

    # Warm up: run once to initialize CUDA context, kernels, and libraries
    if args.warmup:
        with block_all_print():
            main(args)
        print("Warm-up finished")

    # Reset CUDA peak memory statistics
    gc.collect()
    _mem_reset(args)

    # Run the actual measurement
    Timer.reset()
    main(args)
    Timer.print_summary()
