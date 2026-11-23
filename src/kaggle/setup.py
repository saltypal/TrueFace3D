"""Kaggle environment setup: clone pinned upstreams, link private assets, build CUDA extensions.

Usage:  python -m src.kaggle.setup [--skip-build] [--skip-appearance]
Everything is idempotent. Untested on Kaggle (written and reviewed locally only).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import assets, paths

WHEELS = Path("/kaggle/working/wheels")
ARCH = "7.5"  # T4


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=kwargs.pop("check", True), **kwargs)


def git_head(path):
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def clone_pinned(url, sha, dest):
    dest = Path(dest)
    if git_head(dest) == sha:
        return sha
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", dest])
        run(["git", "-C", dest, "remote", "add", "origin", url])
    run(["git", "-C", dest, "fetch", "--depth", "1", "origin", sha])
    run(["git", "-C", dest, "checkout", "--detach", "FETCH_HEAD"])
    if git_head(dest) != sha:
        raise RuntimeError(f"{dest} is not at {sha}")
    return sha


def clone_upstreams(third, include_appearance=True):
    report = {}
    for name, (url, sha, rel) in assets.PINNED.items():
        try:
            report[name] = clone_pinned(url, sha, third / rel)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            report[name] = f"FAILED: {exc}"
    pixel = third / "Pixel3DMM/src/pixel3dmm/preprocessing"
    for src, dst in (("farl.py", "facer/facer/face_parsing/farl.py"), ("facer_transform.py", "facer/facer/transform.py"),
                     ("mica_demo.py", "MICA/demo.py"), ("mica.py", "MICA/micalib/models/mica.py")):
        if (pixel / "replacement_code" / src).is_file() and (pixel / dst).parent.is_dir():
            shutil.copy2(pixel / "replacement_code" / src, pixel / dst)
    if include_appearance:
        for name, (url, rel) in assets.UNPINNED.items():
            dest = third / rel
            try:
                if not dest.is_dir():
                    run(["git", "clone", "--depth", "1", url, dest])
                report[name] = git_head(dest)
            except subprocess.CalledProcessError as exc:
                report[name] = f"FAILED: {exc}"
    return report


def link(src, dst):
    dst = Path(dst)
    if dst.is_symlink():
        if Path(os.readlink(dst)) == Path(src):
            return
        dst.unlink()
    elif dst.exists():
        return  # real file or directory already there
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def link_assets(project=paths.PROJECT):
    """Symlink dataset weights to the project-relative paths each upstream expects."""
    wroot, missing = paths.weights_root(), []
    third = paths.third_party_root()
    if third.resolve() != (project / "third_party").resolve():
        link(third, project / "third_party")
    for rel in assets.REQUIRED_WEIGHTS + assets.OPTIONAL_WEIGHTS:
        src = wroot / rel
        if src.is_file():
            link(src, project / rel)
        elif rel in assets.REQUIRED_WEIGHTS:
            missing.append(rel)
    data = paths.data_root()
    if data.resolve() != (project / "Data").resolve():
        link(data, project / "Data")
    return missing


def pip_install(requirements):
    run([sys.executable, "-m", "pip", "install", "-q", *requirements])


def build_extensions(third, wheels=WHEELS):
    """Build nvdiffrast, flame_solver and PyTorch3D for T4 (sm_75); reuse wheels if present."""
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": ARCH, "MAX_JOBS": "4"}
    wheels.mkdir(parents=True, exist_ok=True)
    attached = list(Path("/kaggle/input").glob("**/wheels")) if Path("/kaggle/input").is_dir() else []
    links = [f"--find-links={p}" for p in [wheels, *attached]]
    results = {}
    for module, dist, path in (("nvdiffrast", "nvdiffrast", third / "nvdiffrast"),
                               ("flame_solver", "flame-solver", third / "RealDenseFace/cuda_extensions/flame_solver"),
                               ("pytorch3d", "pytorch3d", third / "PyTorch3D")):
        if subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True).returncode == 0:
            results[module] = "already importable"
            continue
        reuse = run([sys.executable, "-m", "pip", "install", "-q", "--no-index", *links, dist], check=False, env=env)
        if reuse.returncode == 0:
            results[module] = "installed from wheel"
            continue
        built = run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", wheels, path],
                    check=False, env=env)
        if built.returncode == 0:
            run([sys.executable, "-m", "pip", "install", "-q", "--no-index", f"--find-links={wheels}", dist],
                check=False, env=env)
        results[module] = "built" if built.returncode == 0 else "BUILD FAILED"
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-appearance", action="store_true")
    parser.add_argument("--skip-pip", action="store_true")
    args = parser.parse_args(argv)
    third = paths.third_party_root()
    third.mkdir(parents=True, exist_ok=True)
    report = {"paths": paths.describe()}
    if not args.skip_pip:
        pip_install(["trimesh", "scipy", "scikit-image", "opencv-python-headless", "timm", "roma", "PyYAML",
                     "chumpy", "onnxruntime-gpu", "insightface", "lpips", "loguru", "environs", "mediapy",
                     "tyro", "distinctipy", "pytorch_lightning", "einops", "wandb", "pyvista", "dreifus",
                     "matplotlib", "pandas", "nbformat", "ninja"])
    report["upstreams"] = clone_upstreams(third, not args.skip_appearance)
    report["missing_weights"] = link_assets()
    if not args.skip_build:
        report["builds"] = build_extensions(third)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
