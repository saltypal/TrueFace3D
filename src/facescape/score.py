"""Score stored benchmark fits against the FaceScape TU ground truth (CPU).

Protocol (NoW-style, applied to FaceScape):
- reference = TU vertices (mm) inside facial_mask_v10 (face region, no ears/neck/back of head);
- 7 landmarks (eye corners, nose bottom, mouth corners; iBUG 36, 39, 42, 45, 33, 48, 54) give an initial
  similarity alignment of the prediction to the TU mesh, refined by similarity ICP;
- error = distance from each reference vertex to the aligned predicted surface, in mm.
Two views of the same fit:
- "posed": the model's expression-posed mesh against the GT of that expression (MICA has no
  expression, so its neutral mesh is used and the row is marked);
- "neutral": the model's neutral mesh, scored only on expression 1 (neutral) photos.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import trimesh

from ..evaluation import closest_surface, fit_similarity
from .dataset import FaceScapeC

IBUG7 = [36, 39, 42, 45, 33, 48, 54]


@lru_cache(maxsize=4)
def face_mask_vertices(root: str) -> np.ndarray:
    """Boolean per TU vertex: inside facial_mask_v10.png (sampled at the vertex UV)."""
    root = Path(root)
    topo = np.load(root / "topology.npz")
    mask = cv2.imread(str(root / "facial_mask_v10.png"), cv2.IMREAD_GRAYSCALE)
    n = int(topo["n_vertices"])
    uv = np.full((n, 2), np.nan)
    uv[topo["faces"].ravel()] = topo["uvs"][topo["face_uvs"].ravel()]
    h, w = mask.shape
    px = np.clip(np.nan_to_num(uv[:, 0]) * (w - 1), 0, w - 1).round().astype(int)
    py = np.clip((1 - np.nan_to_num(uv[:, 1])) * (h - 1), 0, h - 1).round().astype(int)
    return (mask[py, px] > 127) & ~np.isnan(uv[:, 0])


def _obj(path):
    m = trimesh.load(str(path), process=False, force="mesh")
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def transfer_landmarks(src_v, faces, lmk, dst_v):
    """Same topology: express each landmark barycentrically on its closest face of src, rebuild on dst."""
    mesh = trimesh.Trimesh(src_v, faces, process=False)
    pts, _, tri = trimesh.proximity.closest_point(mesh, np.asarray(lmk, float))
    bary = trimesh.triangles.points_to_barycentric(src_v[faces[tri]], pts)
    return np.einsum("nk,nkd->nd", bary, dst_v[faces[tri]])


def prediction(model: str, folder: Path, which: str):
    """Return (vertices, faces, landmarks7, note) from a stored fit bundle, or raise FileNotFoundError."""
    folder = Path(folder)
    if model in ("vits", "vitb"):
        z = np.load(folder / "fit" / "face.npz")
        if which == "posed":
            return z["vertices"], z["faces"], np.load(folder / "fit" / "posed_landmarks.npy"), ""
        return z["neutral_vertices"], z["faces"], np.load(folder / "fit" / "neutral_landmarks.npy"), ""
    if model == "deca" and which == "posed":
        v, f = _obj(folder / "native" / "posed.obj")
        nv, nf = _obj(folder / "neutral.obj")      # landmarks68 live on the neutral mesh
        return v, f, transfer_landmarks(nv, nf, np.load(folder / "native" / "landmarks68.npy")[IBUG7], v), ""
    v, f = _obj(folder / "neutral.obj")
    note = "neutral-only model" if (model == "mica" and which == "posed") else ""
    return v, f, np.load(folder / "neutral_landmarks.npy"), note


def align_and_measure(pred_v, pred_f, pred_lmk, ref_pts, ref_lmk, iterations=15, icp_points=3000, seed=0):
    """Landmark similarity init, similarity ICP on a fixed subsample, then the full-face distance once."""
    mesh = trimesh.Trimesh(pred_v.astype(np.float64), pred_f, process=False)
    t = fit_similarity(np.asarray(pred_lmk, float), ref_lmk, scale=True)
    mesh.apply_transform(t)
    rng = np.random.default_rng(seed)
    sub = ref_pts[rng.choice(len(ref_pts), min(icp_points, len(ref_pts)), replace=False)]
    closest, dist, _ = closest_surface(mesh, sub)
    for _ in range(iterations):
        step = fit_similarity(closest, sub, scale=True)
        cand = mesh.copy(); cand.apply_transform(step)
        c2, d2, _ = closest_surface(cand, sub)
        gain = float(np.mean(dist ** 2) - np.mean(d2 ** 2))
        if gain < 0:
            break
        mesh, closest, dist, t = cand, c2, d2, step @ t
        if gain < 1e-4:
            break
    _, full, _ = closest_surface(mesh, ref_pts)
    lmk_after = trimesh.transform_points(np.asarray(pred_lmk, float), t)
    return full, float(np.linalg.norm(lmk_after - ref_lmk, axis=1).mean()), t


def score_record(ds: FaceScapeC, fits_root: Path, rec: dict, which: str) -> dict:
    row = {"model": rec["model"], "image_id": rec["image_id"], "condition": rec["condition"], "protocol": which}
    if rec.get("fit_status") != "success":
        return {**row, "score_status": "fit_unavailable"}
    subject, expression, view = rec["image_id"].split("_")
    if which == "neutral" and int(expression) != 1:
        return {**row, "score_status": "skipped_not_neutral"}
    case = ds.cases(subject=int(subject), expression=int(expression), view=view)[0]
    gt = case.gt_vertices_tu()
    ref_pts = gt[face_mask_vertices(str(ds.root))]
    ref_lmk = gt[ds.topology["landmarks68"][IBUG7]]
    folder = Path(fits_root) / rec["model"] / rec["condition"] / rec["image_id"]
    try:
        v, f, lmk, note = prediction(rec["model"], folder, which)
        dist, lmk_err, t = align_and_measure(v, f, lmk, ref_pts, ref_lmk)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {**row, "score_status": f"failed: {type(exc).__name__}: {exc}"[:200]}
    return {**row, "score_status": "success", "note": note, "n_points": len(dist),
            "mean_mm": float(dist.mean()), "median_mm": float(np.median(dist)), "std_mm": float(dist.std()),
            "p90_mm": float(np.percentile(dist, 90)), "landmark7_mm": lmk_err,
            "scale": float(np.cbrt(abs(np.linalg.det(t[:3, :3]))))}


def _score_one(args):
    root, fits_root, rec, protocols = args
    ds = FaceScapeC(root)
    return [score_record(ds, Path(fits_root), rec, which) for which in protocols]


def score_folder(fits_root, out_csv=None, protocols=("posed", "neutral"), workers=None, progress_every=200):
    """Score every record.json under fits_root/<model>/<condition>/<image_id>/ in parallel processes."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    ds = FaceScapeC.find()
    fits_root = Path(fits_root)
    recs = [json.loads(p.read_text()) for p in sorted(fits_root.glob("*/*/*/record.json"))]
    jobs = [(str(ds.root), str(fits_root), r, protocols) for r in recs]
    rows = []
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, part in enumerate(pool.map(_score_one, jobs, chunksize=4), 1):
            rows.extend(part)
            if i % progress_every == 0 or i == len(jobs):
                print(f"[score] {i}/{len(jobs)}", flush=True)
    table = pd.DataFrame(rows)
    if out_csv:
        table.to_csv(out_csv, index=False)
    return table
