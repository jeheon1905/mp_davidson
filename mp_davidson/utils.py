import importlib.util
import os
import subprocess
import torch

from gospel.ParallelHelper import ParallelHelper as PH


def get_git_commit(package_name: str) -> str:
    """
    Retrieve the current Git commit hash for the package's directory if it belongs to a Git repository.

    Args:
        package_name: The name of the package to check.

    Returns:
        str: A message that includes the package name and the git commit hash,
             or an error message if the package is not found, not under git control,
             or if a git error occurs.
    """
    # Find the package specification using importlib
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.origin:
        return f"{package_name}: package not found"

    # Extract the package installation directory
    package_dir = os.path.dirname(spec.origin)

    # Traverse up the directory tree to search for a .git folder
    current = package_dir
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            try:
                # Retrieve the current git commit hash
                commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=current)
                    .decode()
                    .strip()
                )
                return f"{package_name}: {commit}"
            except subprocess.CalledProcessError:
                return f"{package_name}: no git info (error)"
        next_dir = os.path.dirname(current)
        if next_dir == current:  # Reached the root directory (cross-platform)
            break
        current = next_dir

    return f"{package_name}: not a git repository"


def warm_up(use_cuda=True):
    if use_cuda:
        device = PH.get_device(args.use_cuda)

        ## matmul warm-up
        A = torch.randn(1000, 1000).to(device)
        for i in range(5):
            A @ A
        print("Warm-up matmul")

        ## redistribute warm-up
        A = PH.split(A).to(device)
        _A = PH.redistribute(A, dim0=1, dim1=0)
        A = PH.redistribute(_A, dim0=0, dim1=1)
        print("Warm-up redistribute")
        del A, _A
    else:
        return


if __name__ == "__main__":
    print(get_git_commit("mp_davidson"))
    print(get_git_commit("gospel"))
