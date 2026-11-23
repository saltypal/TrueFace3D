"""DECA: albedo (when the FLAME texture model is present) and DECA's own photo-projected UV texture.

UNVERIFIED on Kaggle: follows DECA's demo_reconstruct flow. DECA needs data/deca_model.tar and
data/generic_model.pkl; FLAME_albedo_from_BFM.npz is optional (without it only the projected texture exists).
"""
from pathlib import Path
import sys

from ...kaggle import paths
from .base import guarded, not_run, third_party, upstream_ready


def _weights():
    root = third_party("DECA") / "data"
    return [root / "deca_model.tar", root / "generic_model.pkl"]


def available():
    return upstream_ready("DECA", _weights())


@guarded
def _run(image_path, out_dir):
    import cv2
    import numpy as np
    import torch
    repo = str(third_party("DECA"))
    sys.path.insert(0, repo)
    from decalib.datasets import datasets
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = (Path(repo) / "data/FLAME_albedo_from_BFM.npz").is_file()
    deca = DECA(config=deca_cfg, device="cuda")
    data = datasets.TestData(str(image_path), iscrop=True, face_detector="fan")[0]
    with torch.no_grad():
        codes = deca.encode(data["image"].to("cuda")[None])
        ops, vis = deca.decode(codes)
    def save(name, tensor):
        image = (tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / name), image[..., ::-1])
        return str(out_dir / name)
    result = {"status": "ok"}
    if "uv_texture_gt" in ops:
        result["uv_texture"] = save("uv_texture.png", ops["uv_texture_gt"])
    if deca_cfg.model.use_tex and "albedo" in ops:
        result["albedo"] = save("albedo.png", ops["albedo"])
    if "shape_images" in vis:
        result["render"] = save("render.png", (vis["shape_images"] ))
    deca.save_obj(str(out_dir / "mesh.obj"), ops)
    result["mesh_obj"] = str(out_dir / "mesh.obj")
    return result


def run(image_path, out_dir):
    ok, why = available()
    return _run(image_path, out_dir) if ok else not_run(why)
