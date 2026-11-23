"""Raw photo projection onto the ViT-S fitted mesh (the appearance baseline)."""
from pathlib import Path
import subprocess
import sys

from ...kaggle import paths
from ..projection import export_textured_glb, load_vit_bundle, save_projection
from .base import guarded, not_run


def available():
    weights = paths.PROJECT / "third_party/RealDenseFace/weights/realdenseface/vits.pth"
    return (weights.exists(), "ok" if weights.exists() else f"missing {weights}")


def fit_vits(image_path, fit_dir):
    command = [sys.executable, "-m", "src.cli", "--model", "vits", "reconstruct", "--input", str(image_path),
               "--output", str(fit_dir), "--device", "cuda", "--python", sys.executable, "--precision", "fp32-compat"]
    done = subprocess.run(command, cwd=paths.PROJECT, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("ViT-S fit failed: " + done.stderr.strip()[-300:])


@guarded
def _run(image_path, out_dir, fit_dir=None):
    if fit_dir is None:
        fit_dir = out_dir / "fit"
        if not (fit_dir / "face.npz").is_file():
            fit_vits(image_path, fit_dir)
    mesh = load_vit_bundle(fit_dir)
    result = save_projection(out_dir, image_path, mesh)
    from PIL import Image
    import numpy as np
    glb = export_textured_glb(mesh, np.asarray(Image.open(result["uv_texture"])), out_dir / "textured.glb")
    return {"status": "ok", "fit_dir": str(fit_dir), "glb": glb, **result}


def run(image_path, out_dir, fit_dir=None):
    ok, why = available()
    return _run(image_path, out_dir, fit_dir=fit_dir) if ok else not_run(why)
