"""Shared building blocks for every generated notebook (Kaggle and local).

Rules enforced here (docs/KAGGLE_LESSONS.md):
- first cell: delete any old checkout and `git clone` the latest commit of the branch; print the commit;
- every long step prints progress (src.kaggle.runner / src.facescape.build.Progress);
- last cell: zip only the results folder (no repo copy, builds or weights) and print its size;
- kernel metadata pins the Docker image ("original") and lists dataset sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/saltypal/TrueFace3D.git"
BRANCH = "facescape-c"
DATASET = "satyapaladugu/facescape-c"          # private, shared with account B
WHEELS = "satyapaladugu/mesha-output"          # cached CUDA wheels (lesson 2)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def clone_cell(branch: str = BRANCH) -> str:
    return f'''# Reset: always start from a fresh clone of the latest commit (never a stale copy).
import os, shutil, subprocess, sys, time, json
REPO_URL, BRANCH = "{REPO_URL}", "{branch}"
REPO = "/kaggle/working/TrueFace3D" if os.path.isdir("/kaggle") else os.path.abspath("_clone_TrueFace3D")
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO], check=True)
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
os.chdir(REPO); sys.path.insert(0, REPO)
T0 = time.time()
print(f"[setup] branch {{BRANCH}} commit {{COMMIT}} at {{time.strftime('%Y-%m-%d %H:%M:%S')}}", flush=True)'''


def run_info_cell(out: str) -> str:
    return f'''# Record what produced these results (lesson 22).
from pathlib import Path
OUT = Path("{out}"); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "run_info.json").write_text(json.dumps({{"commit": COMMIT, "branch": BRANCH,
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local")}}, indent=1))
print("[setup] outputs ->", OUT, flush=True)'''


def zip_cell(out: str, exclude: tuple[str, ...] = ("*.pth", "*.ckpt", "*.whl", "__pycache__")) -> str:
    return f'''# Last cell: one zip of the results only (lessons 18-20).
import fnmatch, zipfile
OUT = Path("{out}"); ZIP = OUT.with_suffix(".zip")
EXCLUDE = {list(exclude)!r}
n = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and not any(fnmatch.fnmatch(part, pat) for part in p.parts for pat in EXCLUDE):
            zf.write(p, p.relative_to(OUT.parent)); n += 1
print(f"[zip] {{n}} files -> {{ZIP}} ({{ZIP.stat().st_size / 1e6:.1f}} MB); total time {{(time.time() - T0) / 60:.1f}} min", flush=True)'''


def save(path: Path, cells: list) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.validate(nb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


def kernel_metadata(user: str, slug: str, code_file: str, datasets: list[str], gpu: bool = True) -> dict:
    return {"id": f"{user}/{slug}", "title": slug, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": datasets, "competition_sources": [], "kernel_sources": [],
            "docker_image_pinning_type": "original"}


def write_kernel_folder(folder: Path, notebook: Path, meta: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / Path(notebook).name).write_bytes(Path(notebook).read_bytes())
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return folder
