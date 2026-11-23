"""Score saved appearance attempts: mode 1 (clean vs photo) and mode 2 (degraded vs clean)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import metrics


def _load(path, mode="RGB"):
    return None if path is None or not Path(path).is_file() else np.asarray(Image.open(path).convert(mode))


def _attempt(folder):
    folder = Path(folder)
    uv, vis = _load(folder / "uv_texture.png"), _load(folder / "uv_visibility.png", "L")
    albedo = _load(folder / "albedo_uv.png")
    render, mask = _load(folder / "render.png"), _load(folder / "render_mask.png", "L")
    if render is not None and mask is None:
        mask = render.sum(axis=2) > 0
    elif mask is not None:
        mask = mask > 127
    if uv is not None and vis is None:
        vis = uv.sum(axis=2) > 0
    elif vis is not None:
        vis = vis > 127
    return {"uv": uv, "vis": vis, "render": render, "mask": mask, "albedo": albedo}


def _fit_size(image, size_wh, resample=Image.Resampling.BILINEAR):
    if image.shape[1] == size_wh[0] and image.shape[0] == size_wh[1]:
        return image
    return np.asarray(Image.fromarray(image).resize(size_wh, resample))


def score_clean(folder, photo_path):
    a = _attempt(folder)
    if a["render"] is None:
        return {"status": "no_render"}
    h, w = a["render"].shape[:2]
    reference = Path(folder) / "reference_photo.png"
    if reference.is_file():  # methods that render in their own crop frame
        photo_path = reference
    with Image.open(photo_path) as p:
        photo = _fit_size(np.asarray(p.convert("RGB")), (w, h), Image.Resampling.AREA)
    out = metrics.photo_metrics(photo, a["render"], a["mask"])
    if a["vis"] is not None:
        out["uv_coverage"] = metrics.uv_coverage(a["vis"])
    out["status"] = "ok"
    return out


def score_pair(clean_folder, degraded_folder):
    ref, deg = _attempt(clean_folder), _attempt(degraded_folder)
    if ref["render"] is None or deg["render"] is None:
        return {"status": "no_render"}
    h, w = ref["render"].shape[:2]
    deg_render = _fit_size(deg["render"], (w, h))
    deg_mask = _fit_size(deg["mask"].astype(np.uint8) * 255, (w, h), Image.Resampling.NEAREST) > 127
    mask = ref["mask"] & deg_mask
    out = {"status": "ok", "render": metrics.all_metrics(ref["render"], deg_render, mask)}
    if ref["uv"] is not None and deg["uv"] is not None and ref["uv"].shape == deg["uv"].shape:
        out["uv"] = metrics.all_metrics(ref["uv"], deg["uv"], ref["vis"] & deg["vis"])
        out["uv_coverage_ref"] = metrics.uv_coverage(ref["vis"])
        out["uv_coverage_deg"] = metrics.uv_coverage(deg["vis"])
        out["uv_coverage_common"] = metrics.uv_coverage(ref["vis"] & deg["vis"])
    if ref["albedo"] is not None and deg["albedo"] is not None and ref["albedo"].shape == deg["albedo"].shape:
        valid = ref["albedo"].sum(axis=2) > 0
        out["albedo_uv"] = metrics.all_metrics(ref["albedo"], deg["albedo"], valid)
    return out


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")
