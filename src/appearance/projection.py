"""Project a photo onto the FLAME UV atlas and render the textured mesh back into the photo camera.

Uses the CPU rasterizer in rendering/gallery.py (numpy; no CUDA needed). Camera dicts follow
geometry/camera.py (RealDenseFace scene.json "camera"). Untested on real fits on Kaggle.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ..geometry.camera import camera_matrices
from ..rendering.gallery import View, rasterize

_UV_VIEW = View(np.array([1.0, 1.0]), np.array([0.0, 0.0]))


def load_vit_bundle(fit_dir):
    """Mesh, UVs and camera from a ViT-S fit bundle (face.npz + scene.json)."""
    fit_dir = Path(fit_dir)
    with np.load(fit_dir / "face.npz", allow_pickle=False) as face:
        data = {k: face[k] for k in ("vertices", "faces", "uvs", "uv_faces")}
    data["camera"] = json.loads((fit_dir / "scene.json").read_text(encoding="utf-8"))["camera"]
    return data


def _camera_space(vertices, camera):
    intrinsic, extrinsic = camera_matrices(camera)
    view = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    return view, intrinsic


def _photo_view(intrinsic, size):
    return View(np.array([intrinsic[0, 0], intrinsic[1, 1]]), np.array([intrinsic[0, 2], intrinsic[1, 2]]))


def _scaled(view, scale):
    return View(view.focal * scale, view.center * scale)


def _sample(image, xy):
    """Bilinear colour at pixel-centre coordinates xy (N,2) -> (N,3) float."""
    cols = [map_coordinates(image[..., c].astype(np.float32), [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
            for c in range(image.shape[2])]
    return np.stack(cols, axis=1)


def face_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def project_to_uv(image, mesh, tex_size=1024, depth_size=768, tolerance=0.012, min_cos=0.15):
    """Return (uv_rgb uint8 HxWx3, visibility bool HxW, info dict).

    image: HxWx3 uint8 photo. mesh: dict from load_vit_bundle (or same keys).
    A texel is visible when its 3D point projects inside the photo, passes a z-buffer test
    against the fitted surface and faces the camera (cosine > min_cos).
    """
    h, w = image.shape[:2]
    vertices, faces = mesh["vertices"].astype(np.float64), mesh["faces"]
    view_xyz, intrinsic = _camera_space(vertices, mesh["camera"])
    # z-buffer at reduced size covering the whole photo.
    scale = depth_size / max(w, h)
    zview = _scaled(_photo_view(intrinsic, (w, h)), scale)
    depth, _, _ = rasterize(zview, view_xyz, faces, depth_size)

    # UV space rasterization: z is constant 1 so barycentrics come out directly.
    uv = np.asarray(mesh["uvs"], dtype=np.float64)
    uv_xyz = np.column_stack([uv[:, 0] * tex_size, (1 - uv[:, 1]) * tex_size, np.ones(len(uv))])
    _, tri, bary = rasterize(_UV_VIEW, uv_xyz, mesh["uv_faces"], tex_size, surface=True)
    covered = tri >= 0
    ids = tri[covered]
    weights = bary[covered]
    corners = faces[ids]
    point = (view_xyz[corners] * weights[..., None]).sum(axis=1)
    xy = point[:, :2] / point[:, 2:3] * intrinsic[[0, 1], [0, 1]] + intrinsic[:2, 2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1) & (point[:, 2] > 0)
    px = np.clip(np.rint(xy * scale).astype(int), 0, depth_size - 1)
    z = depth[px[:, 1], px[:, 0]]
    span = max(np.ptp(view_xyz[:, 2]), 1e-3)
    unobstructed = np.isfinite(z) & (np.abs(z - point[:, 2]) <= tolerance * span)
    normals = face_normals(view_xyz, faces)[ids]
    facing = (-(normals * point).sum(axis=1) / np.maximum(np.linalg.norm(point, axis=1), 1e-12)) > min_cos
    seen = inside & unobstructed & facing

    uv_rgb = np.zeros((tex_size, tex_size, 3), np.uint8)
    visibility = np.zeros((tex_size, tex_size), bool)
    rows, cols = np.nonzero(covered)
    colours = _sample(image, xy[seen])
    uv_rgb[rows[seen], cols[seen]] = np.clip(np.rint(colours), 0, 255).astype(np.uint8)
    visibility[rows[seen], cols[seen]] = True
    info = {"tex_size": tex_size, "uv_area_texels": int(covered.sum()), "visible_texels": int(seen.sum()),
            "uv_coverage": float(seen.sum() / max(covered.sum(), 1))}
    return uv_rgb, visibility, info


def render_textured(uv_rgb, visibility, mesh, image_size_wh, max_side=1024, fallback=(150, 150, 150)):
    """Render the textured mesh into the photo camera.

    Returns (rgb uint8 HxWx3 at photo aspect, mesh_mask bool, texel_valid bool). Texels without a
    photo observation use ``fallback`` gray so holes are visible rather than black.
    """
    w, h = image_size_wh
    scale = max_side / max(w, h)
    vertices, faces = mesh["vertices"].astype(np.float64), mesh["faces"]
    view_xyz, intrinsic = _camera_space(vertices, mesh["camera"])
    zview = _scaled(_photo_view(intrinsic, (w, h)), scale)
    size = max_side
    _, tri, bary = rasterize(zview, view_xyz, faces, size, surface=True)
    ow, oh = max(1, round(w * scale)), max(1, round(h * scale))
    tri, bary = tri[:oh, :ow], bary[:oh, :ow]
    mask = tri >= 0
    uv = np.asarray(mesh["uvs"], dtype=np.float64)
    corner_uv = uv[mesh["uv_faces"][tri[mask]]]
    pos = (corner_uv * bary[mask][..., None]).sum(axis=1)
    tex = uv_rgb.shape[0]
    xy = np.column_stack([pos[:, 0] * tex - 0.5, (1 - pos[:, 1]) * tex - 0.5])
    colours = _sample(uv_rgb, xy)
    valid = map_coordinates(visibility.astype(np.float32), [xy[:, 1], xy[:, 0]], order=1, mode="nearest") > 0.5
    colours[~valid] = fallback
    out = np.zeros((oh, ow, 3), np.uint8)
    out[mask] = np.clip(np.rint(colours), 0, 255).astype(np.uint8)
    texel_valid = np.zeros((oh, ow), bool)
    texel_valid[mask] = valid
    return out, mask, texel_valid


def save_projection(out_dir, image_path, mesh, tex_size=1024):
    """Project, render back and write uv_texture.png, uv_visibility.png, render.png, projection.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"))
    uv_rgb, visibility, info = project_to_uv(image, mesh, tex_size)
    Image.fromarray(uv_rgb).save(out_dir / "uv_texture.png")
    Image.fromarray((visibility * 255).astype(np.uint8)).save(out_dir / "uv_visibility.png")
    render, mask, valid = render_textured(uv_rgb, visibility, mesh, (image.shape[1], image.shape[0]))
    Image.fromarray(render).save(out_dir / "render.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "render_mask.png")
    (out_dir / "projection.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"uv_texture": str(out_dir / "uv_texture.png"), "uv_visibility": str(out_dir / "uv_visibility.png"),
            "render": str(out_dir / "render.png"), "render_mask": str(out_dir / "render_mask.png"), **info}


def export_textured_glb(mesh, uv_rgb, path):
    """Textured GLB (seams split per UV corner) through trimesh."""
    import trimesh
    from trimesh.visual import TextureVisuals

    vertices, faces = mesh["vertices"], mesh["faces"]
    corners = np.column_stack((faces.ravel(), mesh["uv_faces"].ravel()))
    pairs, inverse = np.unique(corners, axis=0, return_inverse=True)
    solid = trimesh.Trimesh(vertices[pairs[:, 0]], inverse.reshape(-1, 3), process=False)
    solid.visual = TextureVisuals(uv=np.asarray(mesh["uvs"])[pairs[:, 1]], image=Image.fromarray(uv_rgb))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    solid.export(path)
    return str(path)
