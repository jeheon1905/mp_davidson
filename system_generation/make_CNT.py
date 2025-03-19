import numpy as np
from ase.build import nanotube, make_supercell
from ase.visualize import view
from ase.io import write


def main(n, m, vacuum=3.0, supercell=[1, 1, 1], prnt=True):
    cnt = nanotube(n, m)
    prim = np.diag(supercell)
    cnt = make_supercell(cnt, prim)
    cnt.center(vacuum=vacuum, axis=np.arange(3)[~cnt.get_pbc()])
    cnt.center()

    ## print it is metallic?
    if prnt:
        if (2 * n + m) % 3 == 0 or (n - m) % 3 == 0:
            print(f"CNT({n}, {m}) is metallic")
        else:
            print(f"CNT({n}, {m}) is no metallic")
        print(len(cnt), cnt)
    return cnt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nm", type=int, nargs="+", required=True, help="Make CNT (n, m)"
    )
    parser.add_argument(
        "--vacuum", type=float, default=3.0, help="Vacuum size (ang), (default: 3.0)"
    )
    parser.add_argument(
        "--save", type=str, help="Save to CIF file with the given filename"
    )
    parser.add_argument("--supercell", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument(
        "--view",
        action="store_true",
        help="visualize CNT with ASE visualization tool (default: False)",
    )
    args = parser.parse_args()

    n, m = args.nm
    cnt = main(n, m, args.vacuum, args.supercell)

    if args.view:
        view(cnt)

    if args.save:
        # name = f"CNT_{n}.{m}.cif"
        write(args.save, cnt)
        print(f"Save {args.save}")
