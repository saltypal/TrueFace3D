"""Coordinate frames for FaceScape-C.

Frames
------
TU       canonical, millimetres; face looks along +Z, +Y up, +X = subject's left.
MV       multi-view structure-from-motion frame (arbitrary scale), one per subject x expression.
camera   OpenCV camera (x right, y down, z forward) of one photo.

FaceScape gives  X_tu = R_tu (s X_mv) + t_tu  (toolkit Rt_scale_dict) and  X_cam = R_c X_mv + t_c.
Eliminating X_mv and multiplying by s gives a rigid map in mm:
    X_cam_mm = (R_c R_tu^T) X_tu + (s t_c - R_c R_tu^T t_tu)
"""
from __future__ import annotations

import numpy as np

from .constants import VIEW_MAX_PITCH, VIEW_MAX_YAW_ERROR


def tu_to_camera(rt_cam: np.ndarray, scale: float, rt_tu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) taking TU millimetres to this camera's frame in millimetres."""
    rc, tc = rt_cam[:3, :3], rt_cam[:3, 3]
    rtu, ttu = rt_tu[:3, :3], rt_tu[:3, 3]
    r = rc @ rtu.T
    return r, scale * tc - r @ ttu


def transform(points: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ r.T + t


def project(points_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    uvw = points_cam @ k[:3, :3].T
    return uvw[:, :2] / uvw[:, 2:3]


def camera_center_tu(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera centre in TU mm for a TU->camera map (R, t)."""
    return -r.T @ t


def view_angles(center_tu: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch (degrees) and distance (mm) of a camera centre seen from the TU origin."""
    d = float(np.linalg.norm(center_tu))
    x, y, z = center_tu / d
    return float(np.degrees(np.arctan2(x, z))), float(np.degrees(np.arcsin(y))), d


def camera_table(params: dict, scale: float, rt_tu: np.ndarray) -> list[dict]:
    """One row per camera in a params.json: index, validity, yaw/pitch/distance, size."""
    rows = []
    for v in range(len(params) // 9):
        r, t = tu_to_camera(np.asarray(params[f"{v}_Rt"], dtype=np.float64), scale, rt_tu)
        yaw, pitch, dist = view_angles(camera_center_tu(r, t))
        rows.append({"view": v, "valid": bool(params[f"{v}_valid"]), "yaw": yaw, "pitch": pitch,
                     "distance_mm": dist, "width": int(params[f"{v}_width"]),
                     "height": int(params[f"{v}_height"])})
    return rows


def select_view(rows: list[dict], target_yaw: float) -> dict | None:
    """Closest valid, roughly level camera to the target yaw; prefers 12 MP cameras on near-ties."""
    best, best_cost = None, np.inf
    for row in rows:
        yaw_err = abs(row["yaw"] - target_yaw)
        if not row["valid"] or yaw_err > VIEW_MAX_YAW_ERROR or abs(row["pitch"]) > VIEW_MAX_PITCH:
            continue
        cost = yaw_err + 0.5 * abs(row["pitch"]) - (2.0 if row["width"] * row["height"] > 8e6 else 0.0)
        if cost < best_cost:
            best, best_cost = row, cost
    return best
