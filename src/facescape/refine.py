"""Refine the toolkit's TU -> multi-view alignment with rigid ICP against the real multi-view scan.

Measured on 340/325/339: the toolkit's Rt/scale leaves a rigid offset of 1.6-2.9 mm and 0.5-2 deg;
after rigid ICP the TU face lies 0.23-0.24 mm (median) from the scan. The refined map is stored as
X_tu_refined = R_ref @ X_tu + t_ref, and composed into each camera's (R, t).
"""
from __future__ import annotations

import io

import numpy as np
from scipy.spatial import cKDTree


def _rigid(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    return r, cb - r @ ca


def scan_in_tu(ply_bytes: bytes, scale: float, rt_tu: np.ndarray) -> np.ndarray:
    import trimesh
    scan = trimesh.load(io.BytesIO(ply_bytes), file_type="ply", process=False)
    return (rt_tu[:3, :3] @ (np.asarray(scan.vertices, dtype=np.float64) * scale).T).T + rt_tu[:3, 3]


def face_points(vertices_tu: np.ndarray, landmarks: np.ndarray, radius_factor: float = 0.6) -> np.ndarray:
    lm = vertices_tu[landmarks]
    centre, radius = lm.mean(0), radius_factor * np.linalg.norm(lm[0] - lm[16])
    return vertices_tu[np.linalg.norm(vertices_tu - centre, axis=1) < radius]


def icp(source: np.ndarray, target: np.ndarray, iters: int = 30, trim: float = 90.0) -> dict:
    tree = cKDTree(target)
    x = source.copy()
    before = tree.query(x)[0]
    r_tot, t_tot = np.eye(3), np.zeros(3)
    for _ in range(iters):
        d, i = tree.query(x)
        keep = d < np.percentile(d, trim)
        r, t = _rigid(x[keep], target[i[keep]])
        x = x @ r.T + t
        r_tot, t_tot = r @ r_tot, r @ t_tot + t
    after = tree.query(x)[0]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r_tot) - 1) / 2, -1, 1))))
    return {"R": r_tot, "t": t_tot, "median_before_mm": float(np.median(before)),
            "median_after_mm": float(np.median(after)), "p95_after_mm": float(np.percentile(after, 95)),
            "rotation_deg": angle, "translation_mm": float(np.linalg.norm(t_tot))}
