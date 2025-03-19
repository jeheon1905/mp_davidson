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
cd libxc
python setup.py develop

# If pylibxc import fails:
# You may need to add libxc.so* to your library path.
# Example: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libxc
```

## Tests
You can run simple test cases as follows:
### 1. Diagonalization of a Fixed Hamiltonian
This runs the eigenvalue solver on a fixed Hamiltonian.
```bash
# Run the test
python test.py --type fixed

# Plot the convergence history
python plot_residual.py
```

### 2. SCF Calculation
This runs a full SCF calculation.
```bash
# Run the SCF test
python test.py --type scf
```

## Generation of Test Systems
You can generate simple test systems such as Carbon Nanotubes (CNT) or Silicon Nanotubes (SNT).
```bash
# example) generation of CNT (6, 0)
python make_cnt.py ...
```
For more details on building structures, refer to [ASE Build]( https://wiki.fysik.dtu.dk/ase/ase/build/build.html).
