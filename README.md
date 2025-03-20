## Abstract of this project
### A Mixed Precision Approach to a Preconditioned Eigensolver for Efficient Density Functional Calculations on GPUs
This project implements a **mixed precision Block Davidson method** to accelerate **real-space Density Functional Theory (DFT)** calculations on GPUs. By combining mixed precision arithmetic with an efficient preconditioning strategy, the eigensolver significantly reduces computational cost while maintaining accuracy. The method is integrated into **GOSPEL**, an open-source real-space DFT package, enabling large-scale electronic structure calculations with improved performance.

## Environment Setup
This project is based on GOSPEL, a real-space DFT code.
Follow the steps below to set up the required environment.
```bash
# Create and activate a conda environment
conda create -n gospel python=3.10 -y
conda activate gospel

# Install PyTorch (CUDA 11.8 version)
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy==1.26.4     # Torch compatibility (numpy version pinned)
pip install ase               # Atomic Simulation Environment
pip install gitpython         # Git interface for Python
pip install "spglib>=1.16.1"  # Symmetry analysis library

# Install GOSPEL (local development mode)
python setup.py develop

# Install pylibxc (for XC functionals)
git clone https://gitlab.com/libxc/libxc.git
git checkout 6.0.0  # Switch to 6.0.0 tag
cd libxc
python setup.py develop

# If pylibxc import fails:
# You may need to add libxc.so* to your library path.
# Example: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libxc
```

## Tests
This section describes how to run the available tests and interpret their results.

### 1. Diagonalization of a Fixed Hamiltonian
This test computes the eigenvalues of a fixed Hamiltonian using an iterative solver. The convergence history can be visualized for further analysis.

```bash
# Diagonalization of a fixed Hamiltonian
python test.py --filepath ./data/systems/CNT_6.0.cif --pbc 0 0 1 \
    --phase fixed --fixed_convg_tol 1e-3 --diag_iter 100 \
    --use_cuda 1 --retHistory History.pt

# Plot the convergence history
python plot_convg_history.py  --filepath History.pt --plot residual \
    --convg_tol 1e-7 --num_eig 48 --save History.residual.png
python plot_convg_history.py  --filepath History.pt --plot eigval \
    --convg_tol 1e-7 --num_eig 48 --save History.eigval.png
```

### 2. SCF Calculation
This test performs a full SCF (Self-Consistent Field) calculation.

```bash
# SCF calculation
python test.py --filepath ./data/systems/CNT_6.0.cif --pbc 0 0 1 \
    --phase scf --use_cuda 1
```

## Generation of Test Systems
You can generate simple test systems such as a carbon nanotube (CNT).
```bash
cd ./system_generation
# example) generation of CNT (6, 0)
python make_nanotube.py --mn 6 0 --save ../data/systems/CNT_6.0.cif
```
For more details on building structures, refer to [ASE Build](https://wiki.fysik.dtu.dk/ase/ase/build/build.html).
