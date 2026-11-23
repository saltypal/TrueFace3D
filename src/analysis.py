"""Tables and figures for the Kaggle benchmark (used by notebooks/20_results_analysis.ipynb).

Everything is built from plain rows (list of dicts) so no pandas is needed; render tables with
``reporting.table_markdown``. Inputs are the downloaded ``out_A`` and ``out_B`` folders.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .benchmark import _active_artifact, _score_values, result_records
from .reporting import table_markdown  # noqa: F401  (re-exported for notebooks)

CONDITIONS = ("original", "face-64", "face-32", "face-16", "compound")
MODELS = ("vits", "vitb", "deca", "pixel3dmm", "mica")
FACE_WIDTH = {"original": None, "face-64": 64, "face-32": 32, "face-16": 16, "compound": 48}
ORACLE_MODES = ("coord", "depth", "both")


# ---------------------------------------------------------------- loading
def load_mesh_records(out_a):
    """[(run_name, RunRecord)] from every runs/* results.json (active rows only)."""
    rows = []
    for run in sorted(Path(out_a, "runs").glob("*")):
        rows.extend((run.name, rec) for rec in result_records(run))
    return rows


def _local(record, out_a, run):
    """Re-anchor a record's folders under the downloaded output (paths were recorded on Kaggle)."""
    folder = Path(record.case_dir)
    parts = folder.parts
    if "runs" in parts:
        return Path(out_a, *parts[parts.index("runs"):])
    return folder


def score_rows(tagged, out_a, kinds=("clean", "real", "oracle")):
    """One row per attempt with its scan-score summary (None when unscored)."""
    rows = []
    for run, rec in tagged:
        kind = run.split("-")[0]
        if kind not in kinds:
            continue
        folder = _local(rec, out_a, run)
        score_file = folder / "score.json"
        score = json.loads(score_file.read_text()) if score_file.is_file() else {}
        rows.append({
            "run": run, "kind": kind, "model": rec.model_id, "condition": rec.variant_id,
            "image": rec.image_key, "subject": rec.subject_id, "category": rec.image_key.split("/")[1],
            "oracle_mode": run.split("-")[1] if kind == "oracle" else None,
            "fit": rec.fit_status, "score_status": rec.score_status, "folder": str(folder),
            "mean_mm": score.get("mean_mm"), "median_mm": score.get("median_mm"), "std_mm": score.get("std_mm"),
            "p95_mm": score.get("p95_mm"), "within_2_5_mm": score.get("within_2_5_mm")})
    return rows


# ---------------------------------------------------------------- stats
def _agg(values):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "std": None, "p95": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "p95": float(np.percentile(values, 95))}


def clean_mesh_table(rows):
    """B1: original photos vs scan, one row per model; stats are over per-subject mean errors."""
    table = []
    for model in MODELS:
        sel = [r for r in rows if r["kind"] == "clean" and r["model"] == model and r["condition"] == "original"]
        a = _agg([r["mean_mm"] for r in sel])
        table.append({"model": model, "subjects scored": a["n"], "mean (mm)": a["mean"], "median (mm)": a["median"],
                      "std (mm)": a["std"], "p95 (mm)": a["p95"],
                      "within 2.5 mm": _agg([r["within_2_5_mm"] for r in sel])["mean"],
                      "failed fits": sum(r["fit"] != "success" for r in sel)})
    return table


def wilcoxon(deltas):
    """Two-sided paired Wilcoxon on per-subject deltas; returns (statistic, p) or (None, None)."""
    from scipy.stats import wilcoxon as test
    deltas = np.asarray([d for d in deltas if d is not None], float)
    if len(deltas) < 6 or np.allclose(deltas, 0):
        return None, None
    stat, p = test(deltas)
    return float(stat), float(p)


def per_person(rows, kind="clean"):
    """{(model, condition): {subject: mean_mm}} for a kind of run."""
    out = defaultdict(dict)
    for r in rows:
        if r["kind"] == kind and r["mean_mm"] is not None:
            out[(r["model"], r["condition"])][r["subject"]] = r["mean_mm"]
    return out


def degradation_table(rows):
    """B2: error vs scan per condition, and the paired change versus the same person's clean result."""
    people = per_person(rows)
    table = []
    for model in MODELS:
        base = people.get((model, "original"), {})
        for cond in CONDITIONS:
            now = people.get((model, cond), {})
            common = sorted(set(base) & set(now))
            deltas = [now[s] - base[s] for s in common]
            stat, p = wilcoxon(deltas)
            table.append({"model": model, "condition": cond, "n": len(now),
                          "mean vs scan (mm)": _agg(now.values())["mean"], "paired n": len(common),
                          "mean delta (mm)": float(np.mean(deltas)) if deltas else None,
                          "median delta (mm)": float(np.median(deltas)) if deltas else None,
                          "wilcoxon p": p})
    return table


def per_person_deltas(rows):
    """Rows (model, subject, condition, delta_mm) for the paired-line plot."""
    people, out = per_person(rows), []
    for (model, cond), values in people.items():
        base = people.get((model, "original"), {})
        out.extend({"model": model, "subject": s, "condition": cond, "delta_mm": v - base[s]}
                   for s, v in values.items() if s in base and cond != "original")
    return out


def real_world_table(rows):
    sel = [r for r in rows if r["kind"] == "real"]
    table = []
    for model in MODELS:
        for category in sorted({r["category"] for r in sel}):
            part = [r for r in sel if r["model"] == model and r["category"] == category]
            a = _agg([r["mean_mm"] for r in part])
            table.append({"model": model, "category": category, "images": len(part), "scored": a["n"],
                          "mean (mm)": a["mean"], "median (mm)": a["median"], "p95 (mm)": a["p95"],
                          "failed fits": sum(r["fit"] != "success" for r in part)})
    return table


def oracle_table(rows):
    """Oracle gap recovered for ViT-S: (degraded - oracle) / (degraded - clean), per condition and mode."""
    people = per_person(rows)
    clean = people.get(("vits", "original"), {})
    oracle = defaultdict(dict)
    for r in rows:
        if r["kind"] == "oracle" and r["mean_mm"] is not None:
            oracle[(r["oracle_mode"], r["condition"])][r["subject"]] = r["mean_mm"]
    table = []
    for cond in CONDITIONS[1:]:
        degraded = people.get(("vits", cond), {})
        for mode in ORACLE_MODES:
            fixed = oracle.get((mode, cond), {})
            common = sorted(set(degraded) & set(fixed) & set(clean))
            if not common:
                table.append({"condition": cond, "oracle priors": mode, "n": 0})
                continue
            d = np.mean([degraded[s] for s in common])
            o = np.mean([fixed[s] for s in common])
            c = np.mean([clean[s] for s in common])
            gap = d - c
            table.append({"condition": cond, "oracle priors": mode, "n": len(common), "clean (mm)": c,
                          "degraded (mm)": d, "oracle (mm)": o, "gap (mm)": gap,
                          "gap recovered": (d - o) / gap if abs(gap) > 1e-9 else None,
                          "wilcoxon p (oracle vs degraded)": wilcoxon([fixed[s] - degraded[s] for s in common])[1]})
    return table


# ---------------------------------------------------------------- appearance
def load_appearance(out_b):
    """Rows from every <method>/<subject>/<condition>/result.json plus appearance_pairs.json."""
    results = []
    for path in sorted(Path(out_b).glob("*/*/*/result.json")):
        method, subject, condition = path.parts[-4], path.parts[-3], path.parts[-2]
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append({"method": method, "subject": subject, "condition": condition, "dir": str(path.parent), **data})
    pairs_file = Path(out_b, "appearance_pairs.json")
    pairs = json.loads(pairs_file.read_text(encoding="utf-8")) if pairs_file.is_file() else []
    return results, pairs


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def appearance_clean_table(results):
    """B3: clean re-render vs photo inside the face mask, mean over subjects."""
    table = []
    for method in sorted({r["method"] for r in results}):
        sel = [r for r in results if r["method"] == method and r["condition"] == "original"]
        ok = [r for r in sel if r["status"] == "ok" and r.get("metrics_vs_photo", {}).get("status") == "ok"]
        row = {"method": method, "subjects": len(sel), "ok": len(ok),
               "status": ", ".join(sorted({f"{r['status']}" + (f": {r.get('reason', '')[:60]}" if r["status"] != "ok" else "")
                                           for r in sel}))}
        for key in ("psnr", "ssim", "lpips", "ciede2000", "ita_diff_deg", "arcface_cos", "uv_coverage"):
            row[key] = _mean([r["metrics_vs_photo"].get(key) for r in ok])
        table.append(row)
    return table


def appearance_degraded_table(pairs):
    """B4: degraded vs the same method's clean result (render and UV), mean over subjects."""
    table = []
    for method in sorted({p["method"] for p in pairs}):
        for cond in CONDITIONS[1:]:
            sel = [p for p in pairs if p["method"] == method and p["condition"] == cond and p.get("status") == "ok"]
            row = {"method": method, "condition": cond, "n": len(sel)}
            for key in ("psnr", "ssim", "lpips", "ciede2000", "ita_diff_deg", "arcface_cos"):
                row[f"render {key}"] = _mean([p["render"].get(key) for p in sel])
                row[f"uv {key}"] = _mean([p.get("uv", {}).get(key) for p in sel])
            row["uv coverage common"] = _mean([p.get("uv_coverage_common") for p in sel])
            table.append(row)
    return table


# ---------------------------------------------------------------- cost / GLB
def glb_index_table(out_a, out_b=None):
    """Rows with relative links (open in a glTF viewer such as https://gltf-viewer.donmccurdy.com)."""
    rows = []
    for root, name in ((out_a, "A"),) + (((out_b, "B"),) if out_b else ()):
        for path in sorted(Path(root).rglob("*.glb")):
            rows.append({"source": name, "file": path.name, "path": str(path),
                         "link": f"[{path.name}]({path.as_posix()})", "MiB": path.stat().st_size / 2 ** 20})
    return rows


# ---------------------------------------------------------------- plots
def _fig(n=1, size=(7, 4)):
    import matplotlib.pyplot as plt
    return plt.subplots(1, n, figsize=(size[0] * n, size[1]), layout="constrained", squeeze=False)


def plot_error_vs_severity(table):
    """Mean scan error per model against face width (compound plotted at its 48 px width, marked)."""
    fig, ax = _fig()
    ax = ax[0][0]
    for model in MODELS:
        pts = sorted((FACE_WIDTH[r["condition"]] or 460, r["mean vs scan (mm)"], r["condition"]) for r in table
                     if r["model"] == model and r["mean vs scan (mm)"] is not None)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=model)
            for x, y, c in pts:
                if c == "compound":
                    ax.annotate("compound", (x, y), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set(xlabel="Face width in pixels (460 = original)", ylabel="Scan error (mm)")
    ax.legend()
    return fig


def plot_paired_lines(deltas, model="vits"):
    """One line per person: change in scan error from clean to each degraded condition."""
    fig, ax = _fig()
    ax = ax[0][0]
    by_subject = defaultdict(dict)
    for d in deltas:
        if d["model"] == model:
            by_subject[d["subject"]][d["condition"]] = d["delta_mm"]
    conds = CONDITIONS[1:]
    for s, vals in by_subject.items():
        ax.plot(range(len(conds)), [vals.get(c, np.nan) for c in conds], "-o", alpha=.35, color="#315f86")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xticks(range(len(conds)), conds)
    ax.set(ylabel="Degraded minus clean scan error (mm)", title=f"{model}: one line per person")
    return fig


def plot_ced_by_condition(rows, model="vits", conditions=CONDITIONS):
    """Cumulative error distribution of scan distances, pooled over subjects."""
    fig, ax = _fig()
    ax = ax[0][0]
    for cond in conditions:
        values = []
        for r in rows:
            if r["kind"] == "clean" and r["model"] == model and r["condition"] == cond:
                f = Path(r["folder"]) / "distances.npz"
                if f.is_file():
                    with np.load(f, allow_pickle=False) as d:
                        values.append(d["distances_mm"])
        if values:
            x = np.sort(np.concatenate(values))
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=cond)
    ax.set(xlabel="Scan point to mesh (mm)", ylabel="Cumulative proportion", title=model)
    ax.legend()
    return fig


def plot_cost_bars(summary):
    fig, axes = _fig(2)
    names = [r["model"] for r in summary]
    for ax, key, label in ((axes[0][0], "warm_median_wall_s", "Warm median wall time (s)"),
                           (axes[0][1], "peak_reserved_mib_max", "Peak reserved memory (MiB)")):
        ax.bar(names, [r[key] or 0 for r in summary], color="#315f86")
        ax.set_ylabel(label)
    return fig


def plot_appearance_curves(table, metric="render ciede2000"):
    fig, ax = _fig()
    ax = ax[0][0]
    for method in sorted({r["method"] for r in table}):
        pts = [(c, r[metric]) for c in CONDITIONS[1:] for r in table
               if r["method"] == method and r["condition"] == c and r.get(metric) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=method)
    ax.set(ylabel=metric, title=f"{metric}: degraded vs same method's clean result")
    ax.legend()
    return fig


# ---------------------------------------------------------------- image galleries
def clay_render(obj_path, size=256):
    """Frontal clay shading of a neutral mesh using the CPU rasterizer."""
    from .benchmark import PROJECT  # noqa: F401
    from .kaggle.export import _obj_mesh
    from .rendering.gallery import View, rasterize

    vertices, faces = _obj_mesh(obj_path)
    v = vertices.astype(np.float64)
    v = v - v.mean(axis=0)
    cam = np.column_stack([v[:, 0], -v[:, 1], np.ptp(v[:, 2]) * 3 - v[:, 2]])
    view = View(np.array([1.0, 1.0]), np.array([0.0, 0.0])).framed(cam, size)
    _, tri, bary = rasterize(view, cam, faces, size, surface=True)
    tri_n = np.cross(cam[faces[:, 1]] - cam[faces[:, 0]], cam[faces[:, 2]] - cam[faces[:, 0]])
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-12)
    shade = np.clip(np.abs(tri_n[:, 2]), 0, 1)
    image = np.full((size, size, 3), 30, np.uint8)
    seen = tri >= 0
    image[seen] = (shade[tri[seen]][:, None] * [210, 215, 222]).astype(np.uint8)
    return image


def scan_error_heatmap(scan_path, landmark_path, distances_mm, ax=None, vmax=5.0):
    """Scatter of the scored scan vertices (x, y) coloured by distance to the fitted mesh."""
    import matplotlib.pyplot as plt
    from .data_now import load_scan_landmarks
    from .evaluation import load_surface, now_face_mask

    scan = load_surface(scan_path)
    points = scan.vertices[now_face_mask(scan.vertices, load_scan_landmarks(landmark_path))]
    if len(points) != len(distances_mm):
        raise ValueError("Distance vector does not match the scan mask")
    ax = ax or plt.subplots(figsize=(3, 3))[1]
    ax.scatter(points[:, 0], points[:, 1], c=distances_mm, s=1, cmap="magma", vmin=0, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def show_image_grid(images, titles, columns=5, size=2.6):
    import matplotlib.pyplot as plt
    rows = -(-len(images) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(size * columns, size * rows + .4), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
    return fig
