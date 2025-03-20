# System Generation

## Files

- `make_nanotube.py`
  Generates a nanotube structure using ASE's `nanotube` function based on user-specified parameters and optionally saves it as a CIF file.

- `make_bulk.py`
  Generates a bulk crystal structure using ASE's `bulk` function based on user-specified parameters (e.g., symbol, crystal structure, lattice constants, supercell) and optionally saves it as a CIF file.

## Usage

### Nanotube Example
```bash
# Generate CNT (6, 0)
python make_nanotube.py --nm 6 0 --save ../data/systems/CNT_6.0.cif

# Generate SiNT (6, 0)
python make_nanotube.py --nm 6 0 --symbol Si --bond 2.24 --save ../data/systems/SiNT_6.0.cif
```

### Bulk Example
```bash
# 1) FCC Cu with a=3.6 Å
python make_bulk.py --symbol Cu --structure fcc --a 3.6 --cubic --save ../data/systems/Cu_fcc.cif

# 2) Diamond-structured Si with a=5.43 Å
python make_bulk.py --symbol Si --structure diamond --a 5.43 --cubic --save ../data/systems/Si_diamond.cif

# 3) FCC Al with a=4.05 Å and 2×2×2 supercell
python make_bulk.py --symbol Al --structure fcc --a 4.05 --cubic --supercell 2 2 2 --save ../data/systems/Al_fcc_2x2x2.cif
```

For more details on building structures, refer to [ASE Build](https://wiki.fysik.dtu.dk/ase/ase/build/build.html).
