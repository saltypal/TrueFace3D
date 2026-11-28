"""Undistort a FaceScape photo and cut a square face crop with its exact crop camera."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import CROP_CENTER_UP, CROP_PX, FACE_PX
from .geometry import project, transform


@dataclass
class Crop:
    image: np.ndarray        # (CROP_PX, CROP_PX, 3) uint8 RGB
    k: np.ndarray            # 3x3 intrinsics of the crop (undistorted)
    box: tuple[int, int, int]  # x0, y0, side in the undistorted source image (pixels)
    scale: float             # crop pixels per source pixel
    face_px_source: float    # jaw-to-jaw face width in the source photo (pixels)
    face_px: float           # jaw-to-jaw face width in the crop (FACE_PX unless the photo was too narrow)
    padded: bool             # crop ran past the photo border


def face_frame(vertices_tu: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Crop centre (TU mm) and jaw-to-jaw width (mm) from the 68 TU landmarks."""
    lm = vertices_tu[landmarks]
    width = float(np.linalg.norm(lm[0] - lm[16]))
    centre = lm.mean(0) + np.array([0.0, CROP_CENTER_UP * width, 0.0])
    return centre, width


def crop_photo(photo_rgb: np.ndarray, k: np.ndarray, dist: np.ndarray, r: np.ndarray, t: np.ndarray,
               centre_tu: np.ndarray, width_mm: float) -> Crop:
    k = np.asarray(k, dtype=np.float64)[:3, :3]
    und = cv2.undistort(photo_rgb, k, np.asarray(dist, dtype=np.float64))
    c_cam = transform(centre_tu[None], r, t)[0]
    cx, cy = project(c_cam[None], k)[0]
    face_px_source = k[0, 0] * width_mm / c_cam[2]
    side = int(round(face_px_source * CROP_PX / FACE_PX))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    h, w = und.shape[:2]
    side = min(side, h, w)   # close faces in portrait frames: largest square that fits, face appears larger
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    # slide the window inside the photo (face slightly off-centre) rather than adding black borders
    if side <= w:
        x0 = min(max(x0, 0), w - side)
    if side <= h:
        y0 = min(max(y0, 0), h - side)
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        und = cv2.copyMakeBorder(und, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    patch = und[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    image = cv2.resize(patch, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)
    s = CROP_PX / side
    kc = k.copy()
    kc[0, 0] *= s
    kc[1, 1] *= s
    kc[0, 2] = (k[0, 2] - x0 + 0.5) * s - 0.5
    kc[1, 2] = (k[1, 2] - y0 + 0.5) * s - 0.5
    return Crop(image, kc, (x0, y0, side), s, float(face_px_source), float(face_px_source * s), bool(pad))
