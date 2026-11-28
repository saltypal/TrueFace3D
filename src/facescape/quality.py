"""Image-quality measurements for every benchmark input (clean and degraded).

Reference metrics compare each image with its clean photo after resizing it back to the clean crop
size; no-reference metrics are computed on the image as fed to the models.
"""
from __future__ import annotations

import json

import cv2
import numpy as np


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-scale SSIM on grayscale (Gaussian window 11, sigma 1.5)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    va, vb, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def no_reference(img_bgr: np.ndarray) -> dict:
    g = _gray(img_bgr)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    high = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0]),
            "sharpness_lapvar": float(lap.var()),
            "brightness_mean": float(g.mean()),
            "contrast_rms": float(g.std()),
            "noise_sigma_mad": float(1.4826 * np.median(np.abs(high - np.median(high)))),
            "clipped_white_frac": float((g >= 250).mean()),
            "clipped_black_frac": float((g <= 5).mean())}


def full_reference(img_bgr: np.ndarray, clean_bgr: np.ndarray) -> dict:
    up = cv2.resize(img_bgr, clean_bgr.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    mse = float(((up.astype(np.float32) - clean_bgr.astype(np.float32)) ** 2).mean())
    return {"psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-9))),
            "ssim": _ssim(_gray(up), _gray(clean_bgr))}


def measure(root, row: dict, clean_row: dict) -> dict:
    img = cv2.imread(str(root / row["path"]))
    clean = cv2.imread(str(root / clean_row["path"]))
    meta = json.loads(row["meta"]) if row.get("meta") else {}
    out = {"image_id": row["image_id"], "condition": row["condition"], **no_reference(img),
           **full_reference(img, clean)}
    for key in ("face_px", "face_px_after"):
        if key in meta:
            out[key] = meta[key]
    return out
