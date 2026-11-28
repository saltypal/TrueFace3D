"""Checks and figures for the FaceScape-C inspection notebook (notebooks/30_dataset_inspection.ipynb).

Every function returns a table (pandas) or a figure, so the notebook stays a thin list of calls.
Faces are drawn only for publishable subjects (constants.PUBLISHABLE) unless allow_private=True.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from . import constants as C
from .dataset import Case, FaceScapeC
from .geometry import camera_table, select_view, transform
from .mesh import read_obj
from .sources import Sources

EYE_L, EYE_R = range(36, 42), range(42, 48)


def _check_public(subject: int, allow_private: bool) -> None:
    if not allow_private and int(subject) not in C.PUBLISHABLE:
        raise PermissionError(f"subject {subject} is not publishable; pass allow_private=True for local-only use")


# ------------------------------------------------------------------ sources
def sources_table(src: Sources) -> pd.DataFrame:
    rows = []
    for kind, ranges in (("photos", src.image_zips), ("mview meshes", src.shape_zips), ("TU models", src.tu_zips)):
        for a, b, p in ranges:
            z = src._zip(p)
            rows.append({"kind": kind, "archive": p.name, "subjects": f"{a}-{b}", "GB": round(p.stat().st_size / 1e9, 2),
                         "members": len(z.namelist())})
    return pd.DataFrame(rows)


def expected_vs_present(ds: FaceScapeC, src: Sources) -> pd.DataFrame:
    rows = []
    for s in range(321, 360):
        rows.append({"subject": s, "photos": src.has_photos(s), "TU": src.has_tu(s),
                     "mview mesh": src._find(src.shape_zips, s) is not None,
                     "in dataset": any(int(r["subject"]) == s for r in ds.subjects),
                     "invalid": s in C.INVALID})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ splits
def split_summary(ds: FaceScapeC) -> tuple[pd.DataFrame, dict]:
    t = pd.DataFrame(ds.subjects)
    t["age"] = pd.to_numeric(t["age"], errors="coerce")
    summary = t.groupby("split").agg(people=("subject", "count"), female=("gender", lambda g: (g == "f").sum()),
                                     male=("gender", lambda g: (g == "m").sum()), age_min=("age", "min"),
                                     age_median=("age", "median"), age_max=("age", "max"))
    test = {int(s) for s in t.loc[t.split == "test", "subject"]}
    train = {int(s) for s in t.loc[t.split == "train", "subject"]}
    clean = pd.DataFrame(ds.clean_rows)
    leak = set(clean.loc[clean.split == "train", "subject"].astype(int)) & test
    checks = {"test and train people disjoint": not (test & train) and not leak,
              "pilot is a test subject": C.PILOT_SUBJECT in test or C.PILOT_SUBJECT not in {int(s) for s in t.subject},
              "test subjects missing so far": sorted(set(C.TEST_SUBJECTS) - test)}
    return summary.reset_index(), checks


def coverage(ds: FaceScapeC) -> pd.DataFrame:
    clean = pd.DataFrame(ds.clean_rows)
    return clean.pivot_table(index="subject", columns="view", values="image_id", aggfunc="count", fill_value=0)


# ------------------------------------------------------------------ cameras
def rig_table(src: Sources, subject: int, expression: int) -> pd.DataFrame:
    s, rt_tu = src.rt_scale(subject, expression)
    rows = camera_table(src.params(subject, expression), s, rt_tu)
    chosen = {tag: (select_view(rows, yaw) or {}).get("view") for tag, yaw in C.VIEWS.items()}
    for r in rows:
        r["selected_as"] = next((tag for tag, v in chosen.items() if v == r["view"]), "")
        r["megapixels"] = round(r["width"] * r["height"] / 1e6, 1)
    return pd.DataFrame(rows)


def plot_rig(table: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for valid, marker in ((True, "o"), (False, "x")):
        sub = table[table.valid == valid]
        ax.scatter(sub.yaw, sub.pitch, s=sub.megapixels * 8, marker=marker, alpha=0.6,
                   label="valid" if valid else "invalid (ignored)")
    for _, r in table[table.selected_as != ""].iterrows():
        ax.scatter(r.yaw, r.pitch, s=260, facecolors="none", edgecolors="red", linewidths=2)
        ax.annotate(f"{r.selected_as} (#{r.view})", (r.yaw, r.pitch), xytext=(6, 8), textcoords="offset points")
    for yaw in C.VIEWS.values():
        ax.axvspan(yaw - C.VIEW_MAX_YAW_ERROR, yaw + C.VIEW_MAX_YAW_ERROR, color="green", alpha=0.06)
    ax.axhspan(-C.VIEW_MAX_PITCH, C.VIEW_MAX_PITCH, color="grey", alpha=0.06)
    ax.set(xlabel="yaw (deg, + = subject's left)", ylabel="pitch (deg)", title="Camera rig seen from the face")
    ax.legend(loc="lower right")
    return ax


def camera_stats(ds: FaceScapeC) -> pd.DataFrame:
    rows = []
    for case in ds.cases():
        cam = case.camera
        rows.append({"subject": int(case.subject), "expression": int(case.expression), "view": case.view,
                     "source_view": cam["source_view"], "yaw": cam["yaw"], "yaw_error": cam["yaw"] - C.VIEWS[case.view],
                     "pitch": cam["pitch"], "distance_mm": cam["distance_mm"], "face_px_source": cam["face_px_source"],
                     "megapixels": cam["source_size"][0] * cam["source_size"][1] / 1e6, "padded": cam["padded"],
                     "face_width_mm": cam["face_width_mm"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ geometry / units
def units_table(ds: FaceScapeC) -> pd.DataFrame:
    lm = ds.topology["landmarks68"]
    rows = []
    for case in ds.cases(view="front", expression=1):
        v = case.gt_vertices_tu()[lm]
        rows.append({"subject": int(case.subject),
                     "inter_ocular_mm": float(np.linalg.norm(v[list(EYE_L)].mean(0) - v[list(EYE_R)].mean(0))),
                     "jaw_width_mm": float(np.linalg.norm(v[0] - v[16])),
                     "nose_to_chin_mm": float(np.linalg.norm(v[33] - v[8])),
                     "camera_distance_mm": case.camera["distance_mm"]})
    return pd.DataFrame(rows)


def overlay(case: Case, landmarks: np.ndarray, *, every: int = 25, allow_private: bool = False) -> np.ndarray:
    _check_public(case.subject, allow_private)
    img = case.image().copy()
    uv = case.project(case.gt_vertices_camera())
    r = max(1, img.shape[0] // 300)
    for x, y in uv[::every].astype(int):
        cv2.circle(img, (int(x), int(y)), r, (0, 255, 0), -1)
    for x, y in uv[landmarks].astype(int):
        cv2.circle(img, (int(x), int(y)), 2 * r + 1, (255, 0, 0), -1)
    return img


def overlay_grid(ds: FaceScapeC, subject: int, expressions, views=("right45", "front", "left45"), *, size=256,
                 allow_private=False) -> np.ndarray:
    lm = ds.topology["landmarks68"]
    rows = []
    for e in expressions:
        tiles = []
        for v in views:
            cs = ds.cases(subject=subject, expression=e, view=v)
            tile = cv2.resize(overlay(cs[0], lm, allow_private=allow_private), (size, size)) if cs \
                else np.zeros((size, size, 3), np.uint8)
            cv2.putText(tile, f"{C.EXPRESSIONS[e]} {v}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate(rows, 0)


def silhouette_iou(case: Case, faces: np.ndarray) -> float:
    """IoU between the projected GT face silhouette and a skin mask from the photo (YCrCb).

    A rough, label-free alignment score: well-aligned GT covers the skin it should cover."""
    img = case.image()
    uv = case.project(case.gt_vertices_camera()).astype(np.int32)
    proj = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(proj, list(uv[faces]), 1)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135)) > 0
    face_region = cv2.dilate(proj, np.ones((15, 15), np.uint8)) > 0
    skin &= face_region
    p = proj > 0
    return float((p & skin).sum() / max(1, (p | skin).sum()))


def mview_alignment(src: Sources, subject: int, expression: int, max_points: int = 20000) -> dict:
    """Distance (mm) from TU vertices to the multi-view scan after the toolkit's Rt/scale alignment."""
    from scipy.spatial import cKDTree
    import trimesh

    data = src.mview_mesh_bytes(subject, expression)
    if data is None:
        return {"subject": subject, "expression": expression, "status": "no multi-view mesh archive"}
    scan = trimesh.load(io.BytesIO(data), file_type="ply", process=False)
    s, rt = src.rt_scale(subject, expression)
    pts = (rt[:3, :3] @ (np.asarray(scan.vertices) * s).T).T + rt[:3, 3]
    tu = read_obj(src.tu_obj_bytes(subject, expression)).vertices
    lm = tu[src.landmark_indices()]
    centre, radius = lm.mean(0), 0.75 * np.linalg.norm(lm[0] - lm[16])
    face = tu[np.linalg.norm(tu - centre, axis=1) < radius]
    face = face[np.random.default_rng(0).choice(len(face), min(max_points, len(face)), replace=False)]
    d, _ = cKDTree(pts).query(face)
    return {"subject": subject, "expression": C.EXPRESSIONS[expression], "status": "ok", "scan_points": len(pts),
            "median_mm": float(np.median(d)), "mean_mm": float(d.mean()), "p95_mm": float(np.percentile(d, 95))}


# ------------------------------------------------------------------ integrity
def integrity(ds: FaceScapeC, *, hash_every: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    problems, counts = [], Counter()
    for name, rows in (("clean", ds.clean_rows), ("degraded", ds.degraded_rows)):
        for i, r in enumerate(rows):
            p = ds.root / r["path"]
            counts[name, "rows"] += 1
            if not p.exists():
                problems.append({"set": name, "path": r["path"], "problem": "missing"})
                continue
            if i % hash_every == 0:
                counts[name, "hashed"] += 1
                if hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
                    problems.append({"set": name, "path": r["path"], "problem": "sha256 mismatch"})
            for extra in ("camera", "gt"):
                if not (ds.root / r[extra]).exists():
                    problems.append({"set": name, "path": r[extra], "problem": f"{extra} missing"})
    table = pd.DataFrame([{"set": k[0], "what": k[1], "count": v} for k, v in counts.items()])
    return table, problems


# ------------------------------------------------------------------ degradations
def degraded_sheet(ds: FaceScapeC, image_id: str, *, size=160, allow_private=False) -> np.ndarray:
    from src.robustness.profiles import PROFILES
    clean = next(r for r in ds.clean_rows if r["image_id"] == image_id)
    _check_public(clean["subject"], allow_private)
    deg = {r["condition"]: r for r in ds.degraded_rows if r["image_id"] == image_id}

    def tile(path, label):
        im = cv2.cvtColor(cv2.imread(str(ds.root / path)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST if im.shape[0] < size else cv2.INTER_AREA)
        cv2.putText(im, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return im

    rows = []
    for pid in PROFILES:
        tiles = [tile(clean["path"], "clean")]
        tiles += [tile(deg[f"{pid}_s{s}"]["path"], f"{pid} s{s}") if f"{pid}_s{s}" in deg
                  else np.zeros((size, size, 3), np.uint8) for s in (1, 2, 3)]
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate([np.concatenate(rows[i:i + 2], 1) for i in range(0, len(rows), 2)], 0)


def input_quality(ds: FaceScapeC, *, split="test", max_images=60) -> pd.DataFrame:
    """PSNR of every degraded image against its clean photo (degraded resized back to the crop size)."""
    clean = {r["image_id"]: r for r in ds.clean_rows}
    ids = sorted({r["image_id"] for r in ds.degraded_rows if r["split"] == split})[:max_images]
    keep = set(ids)
    out = []
    cache = {}
    for r in ds.degraded_rows:
        if r["image_id"] not in keep:
            continue
        if r["image_id"] not in cache:
            cache[r["image_id"]] = cv2.imread(str(ds.root / clean[r["image_id"]]["path"])).astype(np.float32)
        ref = cache[r["image_id"]]
        im = cv2.imread(str(ds.root / r["path"]))
        im = cv2.resize(im, ref.shape[1::-1], interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mse = float(((im - ref) ** 2).mean())
        out.append({"profile": r["profile"], "severity": int(r["severity"]),
                    "psnr_db": 10 * np.log10(255 ** 2 / max(mse, 1e-9)), "width": int(r["width"])})
    t = pd.DataFrame(out)
    return t.pivot_table(index="profile", columns="severity", values="psnr_db", aggfunc="mean").round(1)


def degraded_projection_check(ds: FaceScapeC, condition="C01_s1", n=6) -> pd.DataFrame:
    """Landmarks projected with K_image of a downsized image must equal the clean projection rescaled."""
    lm = ds.topology["landmarks68"]
    rows = []
    for case in ds.cases(split="test", condition=condition)[:n]:
        clean = ds.cases(subject=case.subject, expression=case.expression, view=case.view)[0]
        uv_d = case.project(case.gt_vertices_camera())[lm]
        uv_c = clean.project(clean.gt_vertices_camera())[lm]
        sc = int(case.width) / C.CROP_PX
        expected = (uv_c + 0.5) * sc - 0.5
        rows.append({"image_id": case.image_id, "size": f"{case.width}x{case.height}",
                     "max_px_diff": float(np.abs(uv_d - expected).max())})
    return pd.DataFrame(rows)


def train_mix(ds: FaceScapeC) -> pd.DataFrame:
    t = pd.DataFrame([r for r in ds.degraded_rows if r["split"] == "train"])
    if t.empty:
        return t
    return t.pivot_table(index="profile", columns="view", values="image_id", aggfunc="count", fill_value=0)


def sizes(ds: FaceScapeC) -> pd.DataFrame:
    rows = []
    for name, rs in (("clean", ds.clean_rows), ("degraded", ds.degraded_rows)):
        for split in ("train", "test"):
            sub = [r for r in rs if r["split"] == split]
            gb = sum((ds.root / r["path"]).stat().st_size for r in sub) / 1e9
            rows.append({"set": name, "split": split, "images": len(sub), "GB": round(gb, 2)})
    gt = sum(p.stat().st_size for p in (ds.root / "subjects").glob("*/gt/*")) / 1e9
    rows.append({"set": "ground truth (npz + test textures)", "split": "all", "images": "", "GB": round(gt, 2)})
    return pd.DataFrame(rows)


def checks_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame([{"check": k, "result": "PASS" if v is True else ("FAIL" if v is False else str(v))}
                         for k, v in results.items()])


def read_build_info(ds: FaceScapeC) -> dict:
    p = ds.root / "build_info.json"
    return json.loads(p.read_text()) if p.exists() else {}


__all__ = [n for n in dir() if not n.startswith("_") and n not in {"np", "pd", "cv2", "io", "json", "hashlib"}]
_ = transform  # re-exported helper for notebooks
