"""DECA: BFM-derived FLAME albedo + SH lighting, plus DECA's photo-projected UV texture.

Needs data/deca_model.tar, data/generic_model.pkl and data/FLAME_albedo_from_BFM.npz. Renders live in DECA's
224 px crop, so the crop is saved as reference_photo.png and clean-vs-photo metrics use it.
UNVERIFIED on Kaggle.
"""
import sys

from .base import guarded, not_run, third_party, upstream_ready


def _weights():
    root = third_party("DECA") / "data"
    return [root / "deca_model.tar", root / "generic_model.pkl", root / "FLAME_albedo_from_BFM.npz"]


def available():
    return upstream_ready("DECA", _weights())


@guarded
def _run(image_path, out_dir, **_):
    import cv2
    import numpy as np
    import torch
    repo = str(third_party("DECA"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from decalib.datasets import datasets
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = True
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg, device="cuda")
    data = datasets.TestData(str(image_path), iscrop=True, face_detector="fan")[0]
    image = data["image"].to("cuda")[None]
    with torch.no_grad():
        codes = deca.encode(image)
        ops, vis = deca.decode(codes)

    def save(name, tensor):
        array = (tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / name), array[..., ::-1] if array.shape[2] == 3 else array[..., 0])
        return str(out_dir / name)

    result = {"status": "ok", "albedo_uv": save("albedo_uv.png", ops["albedo"]),
              "render": save("render.png", ops["rendered_images"]),
              "render_mask": save("render_mask.png", ops["alpha_images"]),
              "reference_photo": save("reference_photo.png", image),
              "clay": save("clay.png", vis["shape_images"])}
    if "uv_texture_gt" in ops:
        result["uv_texture"] = save("uv_texture.png", ops["uv_texture_gt"])
    deca.save_obj(str(out_dir / "mesh.obj"), ops)
    result["mesh_obj"] = str(out_dir / "mesh.obj")
    np.savez(out_dir / "codes.npz", **{k: v.detach().cpu().numpy() for k, v in codes.items() if hasattr(v, "detach")})
    return result


def run(image_path, out_dir, **kwargs):
    ok, why = available()
    return _run(image_path, out_dir, **kwargs) if ok else not_run(why)
