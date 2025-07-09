import numpy as np
from ase.build import make_supercell
from ase.io import read, write
from ase.visualize import view


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_cif",
        type=str,
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="Supercell size (default: 1 1 1)",
    )
    parser.add_argument("--save", type=str, help="Save to CIF file")
    parser.add_argument("--view", action="store_true", help="Visualize structure")

    args = parser.parse_args()

    atoms = read(args.input_cif)
    prim = np.diag(args.supercell)
    atoms = make_supercell(atoms, prim)

    print(f"Atoms count: {len(atoms)}")
    print(atoms)

    if args.view:
        view(atoms)

    if args.save:
        write(args.save, atoms)
        print(f"Saved structure to {args.save}")
