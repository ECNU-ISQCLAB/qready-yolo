import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ENV_NAME = "yolov7-csdn"
DEFAULT_PYTHON = "3.10"
DEFAULT_CUDA = "12.1"


ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = Path(__file__).resolve().parent / "requirements-server.txt"
VERIFY_SCRIPT = Path(__file__).resolve().parent / "verify_server_env.py"


def run(command, cwd=ROOT):
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def conda_executable():
    conda = shutil.which("conda")
    if conda:
        return conda
    raise RuntimeError("Could not find `conda` in PATH. Activate Anaconda/Miniconda first.")


def conda_env_exists(conda, env_name):
    result = subprocess.run(
        [conda, "env", "list"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == env_name:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Create the conda environment required by this YOLOv7 PyTorch project."
    )
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--cuda", default=DEFAULT_CUDA, choices=["12.1", "11.8", "cpu"])
    parser.add_argument("--force", action="store_true", help="Remove and recreate the environment.")
    parser.add_argument("--skip-verify", action="store_true", help="Install only; do not run the verification script.")
    args = parser.parse_args()

    conda = conda_executable()

    print("=" * 88)
    print("YOLOv7 environment setup")
    print("=" * 88)
    print(f"project root: {ROOT}")
    print(f"env name: {args.env_name}")
    print(f"python: {args.python}")
    print(f"pytorch cuda runtime: {args.cuda}")
    print(f"requirements: {REQ_FILE}")

    if args.force and conda_env_exists(conda, args.env_name):
        run([conda, "env", "remove", "-n", args.env_name, "-y"])

    if not conda_env_exists(conda, args.env_name):
        run([conda, "create", "-n", args.env_name, f"python={args.python}", "-y"])
    else:
        print(f"\nConda env `{args.env_name}` already exists. Reusing it.")

    run([conda, "install", "-n", args.env_name, "numpy=1.26.4", "-y"])

    if args.cuda == "cpu":
        run([
            conda,
            "install",
            "-n",
            args.env_name,
            "pytorch",
            "torchvision",
            "cpuonly",
            "-c",
            "pytorch",
            "-y",
        ])
    else:
        run([
            conda,
            "install",
            "-n",
            args.env_name,
            "pytorch",
            "torchvision",
            f"pytorch-cuda={args.cuda}",
            "-c",
            "pytorch",
            "-c",
            "nvidia",
            "-y",
        ])

    run([
        conda,
        "run",
        "-n",
        args.env_name,
        "python",
        "-m",
        "pip",
        "install",
        "-r",
        str(REQ_FILE),
    ])

    if not args.skip_verify:
        run([conda, "run", "-n", args.env_name, "python", str(VERIFY_SCRIPT)])

    print("\nDone.")
    print(f"Activate with: conda activate {args.env_name}")
    print("Then train with: python train.py")


if __name__ == "__main__":
    main()
