"""Appearance metrics inside a mask: PSNR, SSIM, LPIPS, CIEDE2000, ITA, ArcFace cosine, UV coverage.

Mode 1 (clean vs photo): ``photo_metrics(photo, render, face_mask)``.
Mode 2 (degraded vs same method's clean result): ``pair_metrics(uv_clean, vis_clean, uv_deg, vis_deg,
render_clean, render_deg, render_mask)`` on the common visible mask.
Optional dependencies (skimage, lpips, onnxruntime) give None plus a reason instead of crashing.
"""
import os
from pathlib import Path

import numpy as np

UNAVAILABLE = "unavailable"


def _float(image):
    return np.asarray(image, dtype=np.float64)


def masked_psnr(a, b, mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    mse = np.mean((_float(a)[mask] - _float(b)[mask]) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def masked_ssim(a, b, mask):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    mask = np.asarray(mask, bool)
    if not mask.any() or min(a.shape[:2]) < 7:
        return None
    _, ssim_map = structural_similarity(np.asarray(a), np.asarray(b), channel_axis=2, full=True, data_range=255)
    return float(ssim_map[mask].mean())


def masked_lpips(a, b, mask, _cache={}):
    try:
        import lpips
        import torch
    except ImportError:
        return None
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    sl = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_mask = mask[sl][..., None]
    a_c, b_c = np.asarray(a)[sl].copy(), np.asarray(b)[sl].copy()
    b_c = np.where(crop_mask, b_c, a_c)  # outside the mask both images agree, so it contributes nothing
    if "net" not in _cache:
        _cache["net"] = lpips.LPIPS(net="alex", verbose=False).eval()
    to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1  # noqa: E731
    with torch.no_grad():
        return float(_cache["net"](to_t(a_c), to_t(b_c)).item())


def _lab(rgb):
    from skimage.color import rgb2lab
    return rgb2lab(np.clip(_float(rgb) / 255.0, 0, 1))


def masked_ciede2000(a, b, mask):
    try:
        from skimage.color import deltaE_ciede2000
    except ImportError:
        return None
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    return float(deltaE_ciede2000(_lab(a)[mask], _lab(b)[mask]).mean())


def ita_angle(rgb, mask):
    """Individual typology angle (degrees) from the median L*, b* inside the mask."""
    try:
        lab = _lab(rgb)
    except ImportError:
        return None
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    L, b = np.median(lab[mask][:, 0]), np.median(lab[mask][:, 2])
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def ita_difference(a, b, mask):
    x, y = ita_angle(a, mask), ita_angle(b, mask)
    return None if x is None or y is None else float(abs(x - y))


_ARC = {}


def arcface_model():
    """ArcFace (buffalo_l w600k_r50.onnx) through onnxruntime; (session | None, reason)."""
    if "s" in _ARC:
        return _ARC["s"]
    from ..kaggle import paths
    path = Path(os.environ.get("TRUEFACE3D_ARCFACE", paths.PROJECT / "checkpoints/insightface/models/buffalo_l/w600k_r50.onnx"))
    try:
        import onnxruntime as ort
        if not path.is_file():
            raise FileNotFoundError(path)
        _ARC["s"] = (ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]), "ok")
    except Exception as exc:  # noqa: BLE001
        _ARC["s"] = (None, f"ArcFace unavailable: {exc}")
    return _ARC["s"]


def arcface_cosine(a, b, mask):
    """Cosine of ArcFace embeddings of the mask bounding box (unaligned crop; mask outside is gray)."""
    session, _ = arcface_model()
    mask = np.asarray(mask, bool)
    if session is None or not mask.any():
        return None
    from PIL import Image
    ys, xs = np.nonzero(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    def embed(image):
        crop = Image.fromarray(np.where(mask[..., None], np.asarray(image), 128).astype(np.uint8)).crop(box)
        x = (np.asarray(crop.resize((112, 112), Image.Resampling.BILINEAR), np.float32) - 127.5) / 127.5
        v = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
        return v / np.linalg.norm(v)

    return float(embed(a) @ embed(b))


def all_metrics(a, b, mask):
    return {"psnr": masked_psnr(a, b, mask), "ssim": masked_ssim(a, b, mask), "lpips": masked_lpips(a, b, mask),
            "ciede2000": masked_ciede2000(a, b, mask), "ita_diff_deg": ita_difference(a, b, mask),
            "arcface_cos": arcface_cosine(a, b, mask), "mask_pixels": int(np.asarray(mask).sum())}


def uv_coverage(visibility, uv_area=None):
    visibility = np.asarray(visibility, bool)
    area = visibility.size if uv_area is None else (np.asarray(uv_area, bool).sum())
    seen = visibility.sum() if uv_area is None else (visibility & np.asarray(uv_area, bool)).sum()
    return float(seen / max(area, 1))


def photo_metrics(photo, render, face_mask, visibility=None):
    """Mode 1: re-render vs the photo inside the rendered face mask (texel-valid pixels only if given)."""
    mask = np.asarray(face_mask, bool)
    if visibility is not None:
        mask = mask & np.asarray(visibility, bool)
    out = all_metrics(photo, render, mask)
    out["mode"] = "clean_vs_photo"
    return out


def pair_metrics(uv_ref, vis_ref, uv_deg, vis_deg, render_ref, render_deg, render_mask,
                 render_valid_ref=None, render_valid_deg=None):
    """Mode 2: degraded result vs the same method's clean result, on common visible regions."""
    common_uv = np.asarray(vis_ref, bool) & np.asarray(vis_deg, bool)
    uv = all_metrics(uv_ref, uv_deg, common_uv)
    rmask = np.asarray(render_mask, bool)
    for valid in (render_valid_ref, render_valid_deg):
        if valid is not None:
            rmask = rmask & np.asarray(valid, bool)
    rend = all_metrics(render_ref, render_deg, rmask)
    return {"mode": "degraded_vs_clean", "uv": uv, "render": rend,
            "uv_coverage_ref": uv_coverage(vis_ref), "uv_coverage_deg": uv_coverage(vis_deg),
            "uv_coverage_common": uv_coverage(common_uv)}
