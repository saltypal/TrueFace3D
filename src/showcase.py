"""Faculty showcase helpers: load benchmark outputs (zip or folder), render meshes uniformly, plot results.

Mesh outputs: any run folder or Kaggle zip with runs/*/cases/<case>/<variant>/<model>/<attempt>/{record,score}.json
and fit/neutral.obj. Appearance outputs: Kaggle out_B folders with <method>/<subject>/<variant>/result.json and
appearance_pairs.json.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np

VARIANTS = ["original", "face-64", "face-32", "face-16", "compound"]
VARIANT_LABELS = {"original": "Original", "face-64": "64 px", "face-32": "32 px", "face-16": "16 px",
                  "compound": "Compound"}
MODELS = ["vits", "vitb", "deca", "mica", "pixel3dmm"]
MODEL_LABELS = {"vits": "ViT-S (RealDenseFace)", "vitb": "ViT-B (RealDenseFace)", "deca": "DECA", "mica": "MICA",
                "pixel3dmm": "Pixel3DMM"}
MODEL_COLORS = {"vits": "#1f5fae", "vitb": "#6a8fd0", "deca": "#2f7d55", "mica": "#a36a00", "pixel3dmm": "#8a4fb3"}


class _Source:
    """Read files from a zip or a folder with the same interface."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None

    def names(self):
        if self.zip:
            return self.zip.namelist()
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()


def load_mesh_results(*sources):
    """Rows of {subject, model, variant, mean_mm, ..., obj} from one or more outputs; later sources win."""
    rows = {}
    for src_path in sources:
        src = _Source(src_path)
        names = src.names()
        nameset = set(names)
        objs = {}
        for n in names:  # RealDenseFace writes <attempt>/fit/neutral.obj; baselines write <attempt>/neutral.obj
            if n.endswith("/record.json"):
                attempt = n.rsplit("/", 1)[0]
                for cand in (attempt + "/fit/neutral.obj", attempt + "/neutral.obj"):
                    if cand in nameset:
                        objs[attempt] = cand
                        break
        for n in names:
            if not n.endswith("/record.json") or "/cases/" not in n:
                continue
            attempt = n.rsplit("/", 1)[0]
            score_name = attempt + "/score.json"
            rec = json.loads(src.read(n))
            if rec.get("score_status") != "success" or score_name not in names:
                continue
            score = json.loads(src.read(score_name))
            key = (rec["subject_id"], rec["model_id"], rec["variant_id"])
            rows[key] = {"subject": key[0], "model": key[1], "variant": key[2], "image_key": rec["image_key"],
                         **{k: score[k] for k in ("mean_mm", "median_mm", "p95_mm", "within_2_5_mm") if k in score},
                         "obj": (src_path, objs.get(attempt))}
    return list(rows.values())


def read_obj(source_and_name):
    src_path, name = source_and_name
    if not name:
        return None, None
    text = _Source(src_path).read(name).decode("utf-8", "replace")
    v, f = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v), np.asarray(f)


def render_mesh(vertices, faces, ax, color=(0.82, 0.82, 0.84)):
    """Flat-shaded frontal render on a matplotlib axis (same camera for every model)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers 3d)
    from matplotlib.collections import PolyCollection
    v = vertices - vertices.mean(0)
    v = v / np.abs(v).max()
    tri = v[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    visible = normals[:, 2] > 0
    light = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])
    shade = np.clip(normals @ light, 0, 1)[visible]
    order = np.argsort(tri[visible][:, :, 2].mean(1))
    polys = tri[visible][order][:, :, :2]
    colors = np.clip(np.outer(0.25 + 0.75 * shade[order], color), 0, 1)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def summary_table(rows):
    """mean scan error per model x variant with n."""
    import pandas as pd
    df = pd.DataFrame([{k: r[k] for k in ("subject", "model", "variant", "mean_mm")} for r in rows])
    return df.pivot_table(index="model", columns="variant", values="mean_mm", aggfunc=["mean", "count"]).reindex(
        [m for m in MODELS if m in set(df.model)])


def plot_scan_error(rows, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for m in MODELS:
        means, xs = [], []
        for i, v in enumerate(VARIANTS):
            vals = [r["mean_mm"] for r in rows if r["model"] == m and r["variant"] == v]
            if len(vals) >= 3:
                xs.append(i)
                means.append(np.mean(vals))
        if xs:
            ax.plot(xs, means, "-o", color=MODEL_COLORS[m], label=MODEL_LABELS[m], lw=2)
    ax.set_xticks(range(len(VARIANTS)), [VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_ylabel("mean scan error (mm)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return ax


def load_appearance(*folders):
    """Pair scores and per-attempt images from one or more out_B folders."""
    pairs, images = [], {}
    for folder in folders:
        folder = Path(folder)
        if (folder / "appearance_pairs.json").is_file():
            pairs += json.loads((folder / "appearance_pairs.json").read_text())
        for res in folder.glob("*/*/*/result.json"):
            method, subject, variant = res.parts[-4:-1]
            data = json.loads(res.read_text())
            if data.get("status") == "ok" and (res.parent / "render.png").is_file():
                images[(method, subject, variant)] = res.parent / "render.png"
    return pairs, images


def pair_metrics(pairs, method, subject, variant, space="render"):
    for p in pairs:
        if p["method"] == method and p["subject"] == subject and p["condition"] == variant and p.get("status") == "ok":
            return p.get(space)
    return None
