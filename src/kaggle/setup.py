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
        for name, (url, sha, rel) in assets.APPEARANCE_PINNED.items():
            try:
                report[name] = clone_pinned(url, sha, third / rel)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                report[name] = f"FAILED: {exc}"
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
    for rel in assets.REQUIRED_WEIGHTS + assets.APPEARANCE_WEIGHTS + assets.OPTIONAL_WEIGHTS:
        src = wroot / rel
        if src.is_file():
            link(src, project / rel)
        elif rel in assets.REQUIRED_WEIGHTS:
            missing.append(rel)
    data = paths.data_root()
    if data.resolve() != (project / "Data").resolve():
        link(data, project / "Data")
    return missing


def prepare_appearance(project=paths.PROJECT):
    """Unpack OpenDelight/TRUST weights into their checkouts and link TRUST's FLAME/DECA data files."""
    import tarfile
    import zipfile
    third, done = paths.third_party_root(), {}
    od_tar, od_dir = project / "checkpoints/opendelight.tar", third / "OpenDelight/pretrained"
    wck = paths.weights_root() / "checkpoints"
    # Kaggle auto-extracts archives in datasets: the .tar/.zip arrive as directories.
    # Kaggle auto-extracts dataset archives, so find the weight files wherever they landed.
    for marker, dest in (("base_delight_network.pth", od_dir / "opendelight"),
                         ("E_albedo_BalanceAlb.tar", third / "TRUST/data/TRUST_models_BalanceAlb_version")):
        hit = next(iter(sorted(wck.rglob(marker))), None) if wck.is_dir() else None
        done[f"found_{marker}"] = str(hit) if hit else None
        if hit and dest.parent.parent.is_dir() and not (dest / marker).is_file():
            link(hit.parent, dest)
    for name in ("BalanceAlb_model.npz",):
        if (wck / name).is_file() and not (project / "checkpoints" / name).exists():
            link(wck / name, project / "checkpoints" / name)
    if od_tar.is_file() and (third / "OpenDelight").is_dir() and not (od_dir / "opendelight/base_delight_network.pth").is_file():
        od_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(od_tar) as archive:
            archive.extractall(od_dir, filter="data")
    done["opendelight"] = (od_dir / "opendelight/base_delight_network.pth").is_file()
    trust = third / "TRUST"
    zip_path = project / "checkpoints/TRUST_models_BalanceAlb_version.zip"
    if zip_path.is_file() and trust.is_dir() and not (trust / "data/TRUST_models_BalanceAlb_version").is_dir():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(trust / "data")
    for src, name in ((project / "third_party/DECA/data/generic_model.pkl", "generic_model.pkl"),
                      (project / "third_party/DECA/data/deca_model.tar", "deca_model.tar"),
                      (project / "checkpoints/BalanceAlb_model.npz", "BalanceAlb_model.npz")):
        if src.exists() and trust.is_dir():
            link(src.resolve(), trust / "data" / name)
    done["trust"] = all((trust / "data" / n).exists() for n in
                        ("generic_model.pkl", "deca_model.tar", "BalanceAlb_model.npz",
                         "TRUST_models_BalanceAlb_version/E_albedo_BalanceAlb.tar"))
    # OpenDelight's face detector/aligner (ibug) keep weights in git-lfs, so clone with lfs then install editable.
    for name in ("face_detection", "face_alignment"):
        dest = third / f"ibug_{name}"
        try:
            if not dest.is_dir():
                run(["git", "lfs", "install"], check=False)
                run(["git", "clone", "--depth", "1", f"https://github.com/hhj1897/{name}.git", dest])
                run(["git", "-C", dest, "lfs", "pull"], check=False)
            run([sys.executable, "-m", "pip", "install", "-q", "-e", dest], check=False)
            done[f"ibug_{name}"] = True
        except subprocess.CalledProcessError as exc:
            done[f"ibug_{name}"] = f"FAILED: {exc}"
    return done


def install_chumpy():
    """chumpy 0.70's setup.py imports old pip internals and fails on modern pip/Python 3.13.

    The package is pure Python, so copy it into site-packages from the sdist. The numpy/inspect
    shims it needs at import time are applied by RealDenseFace's common/fix_chumpy.py and our workers.
    """
    import site
    import tarfile
    import tempfile
    if subprocess.run([sys.executable, "-c", "import importlib.util as u; exit(u.find_spec('chumpy') is None)"]).returncode == 0:
        return "already installed"
    with tempfile.TemporaryDirectory() as tmp:
        # pip download builds metadata and hits the same setup.py failure, so fetch the sdist directly.
        import urllib.request
        meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/chumpy/0.70/json"))
        url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
        archive = Path(tmp) / url.rsplit("/", 1)[-1]
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        package = next(Path(tmp).glob("chumpy-*/chumpy"))
        target = Path(site.getsitepackages()[0]) / "chumpy"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(package, target)
    return f"copied to {target}"


def pip_install(requirements):
    """Install in one go; if that fails, retry one by one so a single bad package can't block the rest."""
    if run([sys.executable, "-m", "pip", "install", "-q", *requirements], check=False).returncode == 0:
        return {}
    failed = {}
    for package in requirements:
        if run([sys.executable, "-m", "pip", "install", "-q", package], check=False).returncode:
            failed[package] = "pip install failed"
    return failed


def build_extensions(third, wheels=WHEELS):
    """Build nvdiffrast, flame_solver and PyTorch3D for T4 (sm_75); reuse wheels if present."""
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": ARCH, "MAX_JOBS": "4", "FORCE_CUDA": "1"}
    # CUDA >= 13 defaults to -static-global-template-stub=true, which breaks flame_solver's cross-file
    # template kernels (undefined AssembleExpPoseRegJacobianCuda<50/100> at link time).
    if (getattr(__import__("torch").version, "cuda", None) or "0").split(".")[0] >= "13":
        env["NVCC_APPEND_FLAGS"] = (env.get("NVCC_APPEND_FLAGS", "") + " -static-global-template-stub=false").strip()
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
        report["pip_failed"] = pip_install(["trimesh", "scipy", "scikit-image", "opencv-python-headless", "timm", "roma", "PyYAML",
                     "onnxruntime-gpu", "insightface", "lpips", "loguru", "environs", "mediapy",
                     "tyro", "distinctipy", "pytorch_lightning", "einops", "wandb", "pyvista", "dreifus",
                     "matplotlib", "pandas", "nbformat", "ninja",
                     "yacs", "iopath", "fvcore", "kornia", "face_alignment", "pyshtools"])
        try:
            report["chumpy"] = install_chumpy()
        except (subprocess.CalledProcessError, StopIteration, OSError) as exc:
            report["chumpy"] = f"FAILED: {exc}"
    report["upstreams"] = clone_upstreams(third, not args.skip_appearance)
    report["missing_weights"] = link_assets()
    if not args.skip_appearance:
        pip_install(["pyfacer"])
        report["appearance"] = prepare_appearance()
    if not args.skip_build:
        report["builds"] = build_extensions(third)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
