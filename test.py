"""
This is the final version of the test code
"""
import argparse
import datetime
import sys
import time
import torch
import numpy as np
from ase import Atoms
from gospel import GOSPEL
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.Eigensolver.precondition import create_preconditioner


def warm_up(use_cuda=True):
    import os

    if use_cuda:
        device = PH.get_device(args.use_cuda)

        ## matmul warm-up
        A = torch.randn(1000, 1000).to(device)
        for i in range(5):
            A @ A
        print(f"Debug: warm-up matmul")

        ## redistribute warm-up
        A = PH.split(A).to(device)
        _A = PH.redistribute(A, dim0=1, dim1=0)
        A = PH.redistribute(_A, dim0=0, dim1=1)
        print(f"Debug: warm-up redistribute")
        del A, _A
    else:
        return


def make_atoms(cif_filename, supercell=[1, 1, 1], pbc=[True, True, True]):
    from ase.build import make_supercell
    from ase.io import read

    atoms = read(cif_filename)
    prim = np.diag(supercell)
    atoms = make_supercell(atoms, prim)
    atoms.set_pbc(pbc)
    vacuum = 3.0
    print(f"vacuum is set to {vacuum}")
    atoms.center(vacuum=vacuum, axis=np.arange(3)[~atoms.get_pbc()])
    return atoms


if __name__ == "__main__":
    # NOTE: 1. Parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="file path (cif or xyz)", required=True)
    parser.add_argument("--supercell", type=int, nargs="+", required=True)
    parser.add_argument(
        "--pbc",
        type=int,
        nargs="+",
        help="periodic boundary condition of each axis. e.g., --pbc True True True",
        required=True,
    )
    parser.add_argument("--pp_type", type=str, default="SG15")
    parser.add_argument("--spacing", type=float, default=0.2, help="grid spacing (ang)")
    # parser.add_argument("--precond_type", type=str, default="shift-and-invert")
    # parser.add_argument("--pcg_precond_type", type=str, default=None, help="precond type for pcg")
    # parser.add_argument("--alpha", type=float, default=0.1, help="precond option alpha")
    # parser.add_argument("--rtol", type=float, default=0.1, help="precond option rtol")
    # parser.add_argument("--no_shift_thr", type=float, default=10.0, help="precond option no_shift_thr")
    parser.add_argument("--precond_iter", type=int, default=4, help="preconditioner's maxiter, default to 4")
    parser.add_argument("--diag_iter", type=int, required=True, help="eigensolver's maxiter")
    parser.add_argument(
        "--use_cuda", type=int, default=1, help="whether using CUDA. 0=False, 1=True"
    )
    parser.add_argument(
        "--phase", type=str, default="scf", help="'scf' or 'fixed'"
    )
    parser.add_argument(
        "--density_filename", type=str, default=None, help="charge density filename to save (or for initialization)"
    )
    parser.add_argument(
        "--guess_filename", type=str, default=None, help="guess eigpair filename"
    )
    parser.add_argument(
        "--save_eig", type=int, default=0, help="whether to save eigenpairs. 0=False, 1=True"
    )
    parser.add_argument(
        "--nbands", type=int, default=None, help="nbands"
    )
    parser.add_argument(
        "--retHistory", type=str, default=None, help="save residual history?"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for starting vectors"
    )
    parser.add_argument(
        "--scf_energy_tol", type=float, default=0.0001, help="scf convergence energy tolerance"
    )
    parser.add_argument(
        "--fixed_convg_tol", type=float, default=0.001, help="fixed Hamiltonian convergence residual norm tolerance"
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
        choices=["DP", "SP", "MP"],
        default="DP",
        help="floating point type",
    )
    parser.add_argument(
        "--precond_fp",
        type=str,
        choices=["DP", "SP"],
        default="DP",
        help="floating point precision for preconditioner",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="whether to use dynamic",
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
    args = parser.parse_args()
    print("args=", args)

    torch.manual_seed(args.seed)
    print(f"datetime: {datetime.datetime.now()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    # set options according to args.fp
    if args.fp == "MP":
        use_MP = torch.float32
    else:
        use_MP = None

    # NOTE: 2. Creating the system
    atoms = make_atoms(args.filepath, args.supercell, args.pbc)
    atoms.center()
    print(atoms)


    # NOTE: 3. Setting GOSPEL calculator

    # Set pseudopotential options
    assert args.pp_type in ["SG15", "ONCV", "TM", "NNLP"]
    pp_path = "../PP_files/"
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


    # Make eigensolver option
    eigensolver = {
        "type": "parallel_davidson",
        "maxiter": args.diag_iter,
        "locking": False,
        "fill_block": False,
        'verbosity': 1,
        'dynamic': args.dynamic,
    }

    # Make GOSPEL calculator
    calc = GOSPEL(
        mixing={"what": "potential"},
        print_energies=True,
        use_cuda=bool(args.use_cuda),
        # use_dense_kinetic=False,
        use_dense_kinetic=args.use_dense_kinetic,
        # precond_type=args.precond_type,
        precond_type=None,
        eigensolver=eigensolver,
        grid={"spacing": args.spacing},
        pp={
            "upf": upf_files,
            "filtering": True,
            # "use_dense_proj": False,
            "use_dense_proj": args.use_dense_proj,
        },
        xc={"type": "gga_x_pbe + gga_c_pbe"},
        convergence={
            "scf_maxiter": 100,
            "density_tol": np.inf,
            "orbital_energy_tol": np.inf,  # mHartree/electron
            "energy_tol": args.scf_energy_tol,
        },
        occupation={
            "smearing": "Fermi-Dirac",
            # "temperature": 1160.45,
            "temperature": 0.0,
        },
        nbands=args.nbands,
    )
    atoms.calc = calc
    calc.initialize(atoms)

    # NOTE: Make preconditioner
    precond_options = {
        "precond_type": "shift-and-invert",
        "grid": calc.grid,
        "use_cuda": args.use_cuda,
        "options": {
            "inner_precond": "gapp",
            "max_iter": args.precond_iter,
            "fp": args.precond_fp,
            "verbosityLevel": 0, # TEST:
            # "verbosityLevel": 1, # TEST:
            "locking": False,
        },
    }
    calc.eigensolver.preconditioner = create_preconditioner(**precond_options)
    print(f"Debug: calc.eigensolver.preconditioner={calc.eigensolver.preconditioner}")


    # NOTE: warm-up
    warm_up(args.use_cuda)


    # NOTE: SCF calculation
    if args.phase == "scf":
        energy = atoms.get_potential_energy()

        # Save the converged density
        if args.density_filename is not None:
            torch.save(calc.get_density(spin=slice(0, sys.maxsize)), args.density_filename)
            print(f"charge density file '{args.density_filename}' is saved.")
    # NOTE: Fixed Hamiltonian diagonalization
    elif args.phase == "fixed":
        from gospel.Hamiltonian import Hamiltonian
        from gospel.Eigensolver.ParallelDavidson import davidson

        # Initialize the electron density
        if args.density_filename is not None:
            device = PH.get_device(calc.parameters["use_cuda"])
            density = torch.load(args.density_filename)
            density = density.reshape(1, -1).to(device)
        else:
            print(f"Debug: initialize density !!!!")
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
            use_dense_kinetic=calc.parameters["use_dense_kinetic"],
            use_cuda=calc.parameters["use_cuda"],
        )
        calc.hamiltonian.update(calc.density)
        del density, calc.density, calc.kpoint, calc.poisson_solver, calc.xc_functional


        # Initialize eigenpair guess
        if args.guess_filename:
            val, vec = torch.load(args.guess_filename)  # val.shape=(nbands,), vec.shape=(ngpts, nbands)
            vec = np.array([[vec.T]], dtype=object)
            eigpair = (val, vec)
            calc.eigensolver.set_initial_eigenpair(eigpair, use_cuda=args.use_cuda, orthonormalize=True)
            del eigpair
        else:
            calc.eigensolver._initialize_guess(calc.hamiltonian)
        print(f"Debug: calc.eigensolver={calc.eigensolver}")

        if args.fp == "SP":
            calc.eigensolver._starting_vector[0, 0] = calc.eigensolver._starting_vector[0, 0].to(torch.float32)

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
            verbosityLevel=1,
            retHistory=(args.retHistory is not None),
            skip_init_ortho=False,
            timing=True,
            use_MP=use_MP,
            MP_scheme=args.MP_scheme,
            debug_recalc_convg_history=args.recalc_convg_history,
        )
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
