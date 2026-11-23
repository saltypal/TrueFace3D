"""CPU previews with a shared view and visibility-tested photo color."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class View:
    focal: np.ndarray
    center: np.ndarray

    def project(self, xyz):
        return xyz[:, :2] / xyz[:, 2, None] * self.focal + self.center

    def framed(self, xyz, size, margin=1.16):
        xy = self.project(xyz)
        low, high = xy.min(axis=0), xy.max(axis=0)
        scale = size / (margin * max(high - low))
        center = (low + high) / 2
        return View(self.focal * scale, (self.center - center) * scale + (size - 1) / 2)


def rasterize(view, vertices, faces, size, *, surface=False):
    """Return camera depth, with optional visible triangle and barycentrics."""
    xy = view.project(vertices)
    depth = np.full((size, size), np.inf)
    triangles = np.full((size, size), -1, np.int32) if surface else None
    weights = np.zeros((size, size, 3), np.float32) if surface else None
    for index, face in enumerate(faces):
        z = vertices[face, 2]
        if np.any(z <= 0):
            continue
        a, b, c = xy[face]
        lo = np.maximum(np.ceil(np.minimum(np.minimum(a, b), c)).astype(int), 0)
        hi = np.minimum(np.floor(np.maximum(np.maximum(a, b), c)).astype(int) + 1, size)
        if np.any(hi <= lo):
            continue
        u, v = b - a, c - a
        determinant = u[0] * v[1] - u[1] * v[0]
        if abs(determinant) < 1e-8:
            continue
        x = np.arange(lo[0], hi[0])[None, :] - a[0]
        y = np.arange(lo[1], hi[1])[:, None] - a[1]
        beta = (v[1] * x - v[0] * y) / determinant
        gamma = (u[0] * y - u[1] * x) / determinant
        alpha = 1 - beta - gamma
        inv_depth = alpha / z[0] + beta / z[1] + gamma / z[2]
        local = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        old = depth[local]
        visible = ((np.minimum(np.minimum(alpha, beta), gamma) >= -1e-7)
                   & (inv_depth > 1 / old))
        if visible.any():
            new = 1 / inv_depth[visible]
            old[visible] = new
            if surface:
                triangles[local][visible] = index
                weights[local][visible] = np.column_stack(
                    (alpha[visible] / z[0], beta[visible] / z[1], gamma[visible] / z[2])) * new[:, None]
    return depth, triangles, weights


def render_pair(image, vertices, faces, normals, camera, reference, landmarks, size=320):
    """Render matching clay and photo-color views. Unknown areas retain clay."""
    pivot = landmarks.mean(axis=0)
    eye_line = landmarks[3] - landmarks[0]
    azimuth = np.arctan2(eye_line[2], eye_line[0])
    angle = np.clip(azimuth, -np.pi / 9, np.pi / 9)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    target = (vertices - pivot) @ rotation.T + pivot
    target_reference = (reference - pivot) @ rotation.T + pivot
    render_view = camera.framed(target_reference, size)
    visibility_size = size * 2
    source_view = camera.framed(reference, visibility_size, margin=1.30)
    source_depth, _, _ = rasterize(source_view, vertices, faces, visibility_size)
    _, triangles, weights = rasterize(render_view, target, faces, size, surface=True)
    surface = triangles >= 0
    if not surface.any():
        raise ValueError("The diagnostic camera did not see the mesh")
    indices = faces[triangles[surface]]
    barycentric = weights[surface][..., None]
    xyz = (vertices[indices] * barycentric).sum(axis=1)
    shade_normal = (normals[indices] * barycentric).sum(axis=1) @ rotation.T
    shade_normal /= np.maximum(np.linalg.norm(shade_normal, axis=1, keepdims=True), 1e-12)
    diffuse = np.maximum(shade_normal @ np.array([-0.35, -0.42, -0.837]), 0)
    gray = np.rint((0.43 + 0.52 * diffuse[:, None]) * [202, 211, 224]).astype(np.uint8)
    clay = np.full((size, size, 3), [25, 29, 35], dtype=np.uint8)
    clay[surface] = gray

    xy = camera.project(xyz)
    px = np.rint(source_view.project(xyz)).astype(int)
    in_source = ((px >= 0) & (px < visibility_size)).all(axis=1)
    height, width = image.shape[:2]
    in_source &= ((xy >= 0) & (xy <= [width - 1, height - 1])).all(axis=1)
    observed = np.zeros(len(xyz), dtype=bool)
    ix, iy = px[in_source].T
    z = source_depth[iy, ix]
    tolerance = max(np.ptp(vertices[:, 2]) * 0.015, 0.001)
    observed[in_source] = np.isfinite(z) & (np.abs(z - xyz[in_source, 2]) <= tolerance)
    sample = xy[observed]
    colors = gray.copy()
    for channel in range(3):
        colors[observed, channel] = map_coordinates(
            image[..., channel], sample[:, ::-1].T, order=1, mode="nearest")
    textured = clay.copy()
    textured[surface] = colors
    return Image.fromarray(clay), Image.fromarray(textured), observed.mean().item()
