"""NumPy projection for single-image RealDenseFace camera diagnostics."""

import numpy as np


def camera_matrices(camera):
    """Return the pinned PerspectiveCamera intrinsics and world-to-view matrix."""
    width, height = camera["image_width"], camera["image_height"]
    focal = height / (2 * np.tan(np.deg2rad(camera["camera_fov_y"]) / 2))
    cx, cy = camera.get("principal_point", (width / 2, height / 2))
    intrinsic = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.asarray(camera["camera_rot"]).T
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ camera["camera_pos"]
    return intrinsic, extrinsic


def project_points(vertices, camera):
    vertices = np.asarray(vertices, dtype=np.float64)
    intrinsic, extrinsic = camera_matrices(camera)
    view = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = view[:, 2]
    valid = np.isfinite(view).all(axis=1) & (depth > 0.01) & (depth < 100.0)
    points = np.full((len(view), 2), np.nan)
    points[valid] = (view[valid] @ intrinsic.T)[:, :2] / depth[valid, None]
    return points, valid


def crop_transform(box, size_hw):
    """Original pixel centers to OpenCV-resized crop pixel centers."""
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    height, width = size_hw
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    return np.array([[sx, 0, sx * (0.5 - x0) - 0.5],
                     [0, sy, sy * (0.5 - y0) - 0.5], [0, 0, 1]])

