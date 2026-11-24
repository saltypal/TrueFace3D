"""OpenDelight: image-space delighting (diffuse albedo) in the original photo frame. UNVERIFIED on Kaggle.

Runs the upstream test.py on a one-image folder. Its pipeline detects the face (ibug RetinaFace, threshold 0.8),
mats it, parses skin with facer, aligns, delights and maps the result back to the input frame (RGBA).
A failed detection on tiny faces is recorded as a failed attempt.
"""
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .base import PYTHON, guarded, not_run, run_cli, third_party, upstream_ready

REPO = "OpenDelight"


def _weights():
    root = third_party(REPO) / "pretrained/opendelight"
    return [root / "base_delight_network.pth", root / "unet_enhancer.pth"]


def available():
    return upstream_ready(REPO, _weights())


@guarded
def _run(image_path, out_dir, **_):
    out_dir = Path(out_dir).resolve()
    inputs, work = out_dir / "od_input", out_dir / "od_work"
    inputs.mkdir(parents=True, exist_ok=True)
    name = "input.png"
    Image.open(image_path).convert("RGB").save(inputs / name)
    # test.py shells out to a bare `python` with bash line continuations for matting, which fails on Windows
    # (and may pick the wrong interpreter). Run the same matting step first with our interpreter.
    mat_root = work / "opendelight/mat"
    mat_root.mkdir(parents=True, exist_ok=True)
    run_cli([PYTHON, "run_matting.py", "--input_root", inputs, "--output_root", mat_root],
            third_party(REPO) / "matting", out_dir / "matting.log")
    run_cli([PYTHON, "test.py", "--data_root", inputs, "--save_root", work, "--device", "0"],
            third_party(REPO), out_dir / "upstream.log")
    rgba = np.asarray(Image.open(work / "opendelight/delight" / name).convert("RGBA"))
    Image.fromarray(rgba[..., :3]).save(out_dir / "render.png")
    Image.fromarray(rgba[..., 3]).save(out_dir / "render_mask.png")
    shutil.copy(inputs / name, out_dir / "reference_photo.png")
    return {"status": "ok", "render": str(out_dir / "render.png"), "render_mask": str(out_dir / "render_mask.png"),
            "reference_photo": str(out_dir / "reference_photo.png"), "space": "image (input frame)"}


def run(image_path, out_dir, **kwargs):
    ok, why = available()
    return _run(image_path, out_dir, **kwargs) if ok else not_run(why)
