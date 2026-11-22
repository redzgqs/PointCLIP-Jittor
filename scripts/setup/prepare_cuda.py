"""Expose a local CUDA toolkit and cuDNN 8 through a project-private Jittor prefix."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts import ROOT

PREFIX = ROOT / ".cache/fewshot/jtcuda"
SPLITS = ("ops_infer", "ops_train", "cnn_infer", "cnn_train", "adv_infer", "adv_train")


def cudnn_lib(root):
    for name in ("lib", "lib64"):
        directory = root / name
        if (directory / "libcudnn.so.8").is_file():
            return directory
    raise FileNotFoundError(f"Expected libcudnn.so.8 under {root}/lib or {root}/lib64")


def find_sources(args):
    # A prepared prefix can be reused even after the shell's CUDA_HOME points at it.
    record = PREFIX / "sources.json"
    sources = json.loads(record.read_text()) if record.is_file() else {}
    cuda = args.cuda or sources.get("cuda") or os.environ.get("CUDA_HOME")
    if not cuda:
        compiler = shutil.which("nvcc")
        if not compiler:
            raise ValueError("Pass --cuda /path/to/cuda or set CUDA_HOME to the toolkit root.")
        cuda = Path(compiler).resolve().parents[1]
    cuda = Path(cuda).expanduser().resolve()
    if not (cuda / "bin/nvcc").is_file():
        raise FileNotFoundError(cuda / "bin/nvcc")
    cudnn = args.cudnn or sources.get("cudnn") or os.environ.get("POINTCLIP_CUDNN_ROOT")
    if not cudnn:
        # NVIDIA's pip distribution installs nvidia/cudnn next to the Python packages.
        candidates = [Path(p) / "nvidia/cudnn" for p in sys.path if p] + [cuda]
        cudnn = next((p for p in candidates if (p / "include/cudnn.h").is_file()
                      and any((p / lib / "libcudnn.so.8").is_file() for lib in ("lib", "lib64"))), None)
        if cudnn is None:
            raise ValueError("Pass --cudnn /path/to/cudnn8 or set POINTCLIP_CUDNN_ROOT to its root.")
    cudnn = Path(cudnn).expanduser().resolve()
    if not (cudnn / "include/cudnn.h").is_file():
        raise FileNotFoundError(cudnn / "include/cudnn.h")
    cudnn_lib(cudnn)
    return cuda, cudnn


def make_prefix(stage, cuda, cudnn):
    nvidia = cudnn.parent
    include, lib, bin_dir = (stage / name for name in ("include", "lib64", "bin"))
    for directory in (include, lib, bin_dir):
        directory.mkdir()
    # Retain toolkit headers and link dependencies together, including hidden
    # headers, nvJitLink and split cuDNN components. No library bytes are copied.
    for source in (cuda / "include").iterdir():
        if not source.name.startswith("cudnn"):
            (include / source.name).symlink_to(source)
    for source in (cudnn / "include").glob("cudnn*.h"):
        (include / source.name).symlink_to(source)
    for source in (cuda / "lib64").iterdir():
        if not source.name.startswith("libcudnn"):
            (lib / source.name).symlink_to(source)
    # A lean toolkit may ship nvJitLink only in the matching NVIDIA package.
    for source in sorted((nvidia / "nvjitlink/lib").glob("libnvJitLink.so*")):
        if not (lib / source.name).is_symlink():
            (lib / source.name).symlink_to(source)
    for source in sorted(cudnn_lib(cudnn).glob("libcudnn*.so.8*")):
        (lib / source.name).symlink_to(source)
        alias = lib / (source.name.split(".so.")[0] + ".so")
        if not alias.is_symlink():
            alias.symlink_to(source)

    # A symlinked nvcc can resolve back into the system toolkit. A tiny wrapper
    # keeps Jittor's include/lib discovery here; real nvcc still sees its toolkit.
    for source in (cuda / "bin").iterdir():
        if source.name != "nvcc":
            (bin_dir / source.name).symlink_to(source)
    wrapper = bin_dir / "nvcc"
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(cuda / "bin/nvcc")) + ' "$@"\n')
    wrapper.chmod(0o755)
    for name in ("nvvm", "version.json", "version.txt", "share"):
        if (cuda / name).exists():
            (stage / name).symlink_to(cuda / name)
    target = stage / "targets/x86_64-linux"
    target.mkdir(parents=True)
    (target / "include").symlink_to("../../include")
    (target / "lib").symlink_to("../../lib64")
    (stage / "lib").symlink_to("lib64")


def verify(prefix):
    library_names = ["libcudart.so", "libcublas.so", "libcublasLt.so", "libcurand.so",
                     "libcufft.so", "libcusparse.so", "libcudnn.so"] + [
                         "libcudnn_" + name + ".so.8" for name in SPLITS]
    for path in [prefix / "include/cudnn.h", prefix / "include/cuda_runtime.h"] + [
            prefix / "lib64" / name for name in library_names]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Dynamic loader paths are read at process startup. Check symbol resolution
    # in a child process using the same precedence as the future Jittor process.
    env = os.environ.copy()
    lib = str(prefix / "lib64")
    old = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = lib + (os.pathsep + old if old else "")
    probe = (
        "import ctypes, os, sys; "
        "libs = [ctypes.CDLL(p, mode=os.RTLD_NOW | os.RTLD_GLOBAL) for p in sys.argv[1:]]; "
        "dnn = ctypes.CDLL('libcudnn.so.8'); version = dnn.cudnnGetVersion(); "
        "assert 8000 <= version < 9000, version; print('cuDNN', version, 'loaded')"
    )
    subprocess.run([sys.executable, "-c", probe] +
                   [str(prefix / "lib64" / name) for name in library_names], env=env, check=True)
    subprocess.run([str(prefix / "bin/nvcc"), "--version"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, help="toolkit containing bin/nvcc, include and lib64")
    parser.add_argument("--cudnn", type=Path, help="cuDNN 8 root with include and lib or lib64")
    args = parser.parse_args()
    PREFIX.parent.mkdir(parents=True, exist_ok=True)
    with (PREFIX.parent / "prepare-cuda.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cuda, cudnn = find_sources(args)
        key = dict(cuda=str(cuda), cudnn=str(cudnn))
        if PREFIX.exists():
            if json.loads((PREFIX / "sources.json").read_text()) != key:
                raise ValueError("Existing project prefix uses different toolkit sources: " + str(PREFIX))
            verify(PREFIX)
        else:
            with TemporaryDirectory(prefix=".jtcuda-", dir=str(PREFIX.parent)) as scratch:
                stage = Path(scratch) / "prefix"
                stage.mkdir()
                make_prefix(stage, cuda, cudnn)
                verify(stage)
                (stage / "sources.json").write_text(json.dumps(key, indent=2) + "\n")
                stage.rename(PREFIX)
    print("Ready:", PREFIX)


if __name__ == "__main__":
    main()
