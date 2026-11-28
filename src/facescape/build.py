"""Build the FaceScape-C folder from the downloaded zips.

    python -m src.facescape.build --stage clean   [--subjects 344 350] [--workers 6]
    python -m src.facescape.build --stage refine            (ICP of TU to the multi-view scan, if available)
    python -m src.facescape.build --stage degrade [--workers 6]
    python -m src.facescape.build --stage all

Layout (root = $FACESCAPE_C or /mnt/DataBunker/facescape_c):
    README.md  build_info.json  subjects.csv  topology.npz  facial_mask_v10.png
    manifest_clean.csv  manifest_degraded.csv  issues.csv
    subjects/<sid>/gt/<exp>.npz              TU vertices (float32, mm, canonical frame)
    subjects/<sid>/gt/<exp>_texture.jpg      TU texture (test subjects only)
    subjects/<sid>/<exp>/<view>/clean.png    real photo crop, undistorted
    subjects/<sid>/<exp>/<view>/camera.json  crop intrinsics K and rigid TU->camera (R, t) in mm
    subjects/<sid>/<exp>/<view>/degraded/<profile>_s<k>.png
Every stage prints progress with rate and ETA, and writes manifests with SHA-256 per file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from . import constants as C
from .crop import crop_photo, face_frame
from .geometry import camera_table, select_view, tu_to_camera
from .mesh import read_obj
from .sources import Sources
from .splits import read_info_list, subject_table

DEFAULT_ROOT = Path(os.environ.get("FACESCAPE_C", "/mnt/DataBunker/facescape_c"))
_SRC: Sources | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"could not write {path}")


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Progress:
    """Prints 'label: done/total, rate, ETA' at most every `every` seconds (and at the end)."""

    def __init__(self, label: str, total: int, every: float = 20.0):
        self.label, self.total, self.every = label, total, every
        self.done, self.t0, self.last = 0, time.time(), 0.0
        print(f"[{label}] start: {total} items", flush=True)

    def step(self, n: int = 1, note: str = "") -> None:
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            el = now - self.t0
            rate = self.done / el if el else 0.0
            eta = (self.total - self.done) / rate if rate else float("nan")
            print(f"[{self.label}] {self.done}/{self.total} ({100 * self.done / max(1, self.total):.0f}%), "
                  f"{rate:.2f}/s, elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min {note}", flush=True)
            self.last = now


# ---------------------------------------------------------------- clean stage
def _init_worker():
    global _SRC
    _SRC = Sources()


def _clean_tuple(job: tuple[int, int, str, str]) -> tuple[list[dict], list[dict]]:
    sid, exp, split, root = job
    root, src = Path(root), _SRC
    rows, issues = [], []
    sdir = root / "subjects" / str(sid)
    try:
        params = src.params(sid, exp)
        scale, rt_tu = src.rt_scale(sid, exp)
        mesh = read_obj(src.tu_obj_bytes(sid, exp))
    except (KeyError, FileNotFoundError) as err:
        return [], [{"subject": sid, "expression": exp, "view": "", "issue": f"missing source: {err}"}]
    gt_rel = f"subjects/{sid}/gt/{C.expression_dir(exp)}.npz"
    (root / gt_rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / gt_rel, vertices=mesh.vertices.astype(np.float32))
    if split == "test":
        (root / gt_rel.replace(".npz", "_texture.jpg")).write_bytes(src.tu_texture_bytes(sid, exp))
    centre, width_mm = face_frame(mesh.vertices, src.landmark_indices())
    table = camera_table(params, scale, rt_tu)
    for tag, target in C.VIEWS.items():
        row = select_view(table, target)
        if row is None:
            issues.append({"subject": sid, "expression": exp, "view": tag, "issue": "no valid camera near target"})
            continue
        v = row["view"]
        photo = cv2.imdecode(np.frombuffer(src.photo_bytes(sid, exp, v), np.uint8), cv2.IMREAD_COLOR)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        r, t = tu_to_camera(np.asarray(params[f"{v}_Rt"], dtype=np.float64), scale, rt_tu)
        crop = crop_photo(photo, np.asarray(params[f"{v}_K"]), np.asarray(params[f"{v}_distortion"]), r, t,
                          centre, width_mm)
        if crop.face_px_source < C.FACE_PX:
            issues.append({"subject": sid, "expression": exp, "view": tag,
                           "issue": f"source face {crop.face_px_source:.0f}px < {C.FACE_PX}px (upscaled)"})
        vdir = sdir / C.expression_dir(exp) / tag
        write_png(vdir / "clean.png", crop.image)
        cam = {"K": crop.k.tolist(), "R": r.tolist(), "t": t.tolist(), "width": C.CROP_PX, "height": C.CROP_PX,
               "units": "mm", "frames": "X_cam = R @ X_tu + t; pixel = K @ X_cam (OpenCV, undistorted)",
               "view_tag": tag, "source_view": v, "yaw": row["yaw"], "pitch": row["pitch"],
               "distance_mm": row["distance_mm"], "source_size": [row["width"], row["height"]],
               "face_px_source": crop.face_px_source, "face_px": crop.face_px, "face_width_mm": width_mm,
               "crop_box_source": list(crop.box), "crop_scale": crop.scale, "padded": crop.padded}
        (vdir / "camera.json").write_text(json.dumps(cam, indent=1))
        rel = vdir.relative_to(root)
        rows.append({"image_id": f"{sid}_{exp:02d}_{tag}", "subject": sid, "expression": exp,
                     "expression_name": C.EXPRESSIONS[exp], "view": tag, "split": split,
                     "condition": "clean", "profile": "clean", "severity": 0, "seed": 0,
                     "path": f"{rel}/clean.png", "width": C.CROP_PX, "height": C.CROP_PX,
                     "sha256": sha256(vdir / "clean.png"), "camera": f"{rel}/camera.json", "gt": gt_rel,
                     "meta": ""})
    return rows, issues


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_clean(root: Path, subjects: list[int] | None, workers: int) -> None:
    src = Sources()
    available = [s for s in src.photo_subjects() if src.has_tu(s)]
    subjects = [s for s in (subjects or available) if s in available]
    table = subject_table(subjects, read_info_list(src.info_list_path()))
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "subjects.csv", table)
    print(f"[clean] photo archives: {[p.name for _, _, p in src.image_zips]}", flush=True)
    print(f"[clean] subjects with photos + TU: {len(subjects)}; splits: "
          + ", ".join(f"{k}={sum(r['split'] == k for r in table)}" for k in ("train", "test", "excluded")), flush=True)
    # shared topology, landmarks and mask (identical for every TU mesh)
    first = next(r["subject"] for r in table if r["split"] != "excluded")
    mesh = read_obj(src.tu_obj_bytes(first, 1))
    np.savez_compressed(root / "topology.npz", faces=mesh.faces, uvs=mesh.uvs, face_uvs=mesh.face_uvs,
                        landmarks68=src.landmark_indices(), n_vertices=len(mesh.vertices))
    shutil.copy(src.facial_mask_path(), root / "facial_mask_v10.png")
    jobs = [(r["subject"], e, r["split"], str(root)) for r in table if r["split"] != "excluded"
            for e in C.EXPRESSIONS]
    prog = Progress("clean", len(jobs))
    rows, issues = [], []
    with Pool(workers, initializer=_init_worker) as pool:
        for r, i in pool.imap_unordered(_clean_tuple, jobs, chunksize=1):
            rows += r
            issues += i
            prog.step(note=f"images {len(rows)}, issues {len(issues)}")
    rows.sort(key=lambda r: r["image_id"])
    write_csv(root / "manifest_clean.csv", rows)
    write_csv(root / "issues.csv", issues or [{"subject": "", "expression": "", "view": "", "issue": "none"}])
    info = {"stage_clean": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _git_commit(),
                            "images": len(rows), "issues": len(issues),
                            "archives": [p.name for _, _, p in src.image_zips + src.tu_zips],
                            "crop_px": C.CROP_PX, "face_px": C.FACE_PX, "views": C.VIEWS}}
    _update_info(root, info)
    print(f"[clean] done: {len(rows)} images, {len(issues)} issues -> {root}", flush=True)


def _update_info(root: Path, new: dict) -> None:
    p = root / "build_info.json"
    info = json.loads(p.read_text()) if p.exists() else {}
    info.update(new)
    p.write_text(json.dumps(info, indent=1))


# -------------------------------------------------------------- degrade stage
def degraded_jobs(clean_rows: list[dict]) -> list[tuple[dict, str, int]]:
    """Test grid (all conditions) for test subjects' test expressions and the pilot's every expression;
    TRAIN_DEGRADED_PER_IMAGE random conditions for every training photo."""
    from src.robustness.profiles import conditions

    grid = [c for c in conditions(include_restorer=True) if c[0] != "clean"]
    jobs = []
    for row in clean_rows:
        sid, exp = int(row["subject"]), int(row["expression"])
        if row["split"] == "test" and (exp in C.TEST_EXPRESSIONS or sid == C.PILOT_SUBJECT):
            jobs += [(row, p, s) for p, s in grid]
        elif row["split"] == "train":
            rng = random.Random(f"{C.BASE_SEED}:{row['image_id']}")
            jobs += [(row, p, s) for p, s in rng.sample(grid, C.TRAIN_DEGRADED_PER_IMAGE)]
    return jobs


def _degrade_one(job: tuple[dict, str, int, str]) -> dict:
    from src.robustness.profiles import apply, condition_seed

    row, pid, sev, root = job
    root = Path(root)
    clean = read_rgb(root / row["path"])
    seed = condition_seed(C.BASE_SEED, row["image_id"], pid, sev)
    face_px = json.loads((root / row["camera"]).read_text())["face_px"]
    img, meta = apply(clean, pid, sev, face_px=face_px, seed=seed)
    rel = Path(row["path"]).parent / "degraded" / f"{pid}_s{sev}.png"
    write_png(root / rel, img)
    out = dict(row)
    out.update({"condition": f"{pid}_s{sev}", "profile": pid, "severity": sev, "seed": seed, "path": str(rel),
                "width": img.shape[1], "height": img.shape[0], "sha256": sha256(root / rel),
                "meta": json.dumps(meta, separators=(",", ":"))})
    return out


def stage_degrade(root: Path, workers: int) -> None:
    clean_rows = read_csv(root / "manifest_clean.csv")
    jobs = [(r, p, s, str(root)) for r, p, s in degraded_jobs(clean_rows)]
    n_test = sum(1 for j in jobs if j[0]["split"] == "test")
    print(f"[degrade] {len(clean_rows)} clean photos -> {len(jobs)} degraded images "
          f"(test grid {n_test}, train {len(jobs) - n_test})", flush=True)
    prog = Progress("degrade", len(jobs))
    rows = []
    with Pool(workers) as pool:
        for out in pool.imap_unordered(_degrade_one, jobs, chunksize=8):
            rows.append(out)
            prog.step()
    rows.sort(key=lambda r: (r["image_id"], r["condition"]))
    write_csv(root / "manifest_degraded.csv", rows)
    _update_info(root, {"stage_degrade": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _git_commit(),
                                          "images": len(rows), "base_seed": C.BASE_SEED,
                                          "train_degraded_per_image": C.TRAIN_DEGRADED_PER_IMAGE}})
    print(f"[degrade] done: {len(rows)} images", flush=True)


# --------------------------------------------------------------- refine stage
def _refine_tuple(job: tuple[int, int, str]) -> dict:
    from .refine import face_points, icp, scan_in_tu

    sid, exp, root = job
    root, src = Path(root), _SRC
    row = {"subject": sid, "expression": exp, "status": "no multi-view mesh"}
    ply = src.mview_mesh_bytes(sid, exp)
    if ply is None:
        return row
    scale, rt_tu = src.rt_scale(sid, exp)
    tu = np.load(root / "subjects" / str(sid) / "gt" / f"{C.expression_dir(exp)}.npz")["vertices"].astype(np.float64)
    res = icp(face_points(tu, src.landmark_indices()), scan_in_tu(ply, scale, rt_tu))
    for cam_path in (root / "subjects" / str(sid) / C.expression_dir(exp)).glob("*/camera.json"):
        cam = json.loads(cam_path.read_text())
        r0 = np.asarray(cam.get("R_toolkit", cam["R"]))
        t0 = np.asarray(cam.get("t_toolkit", cam["t"]))
        cam.update({"R_toolkit": r0.tolist(), "t_toolkit": t0.tolist(), "R": (r0 @ res["R"]).tolist(),
                    "t": (r0 @ res["t"] + t0).tolist(), "refined": True,
                    "refine": {k: v for k, v in res.items() if k not in ("R", "t")}})
        cam_path.write_text(json.dumps(cam, indent=1))
    row.update({"status": "refined", **{k: round(v, 3) for k, v in res.items() if k not in ("R", "t")}})
    return row


def stage_refine(root: Path, workers: int) -> None:
    tuples = sorted({(int(r["subject"]), int(r["expression"])) for r in read_csv(root / "manifest_clean.csv")})
    print(f"[refine] {len(tuples)} subject x expression tuples; ICP of TU to the multi-view scan", flush=True)
    prog = Progress("refine", len(tuples))
    rows = []
    with Pool(workers, initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_refine_tuple, [(s, e, str(root)) for s, e in tuples]):
            rows.append(row)
            prog.step(note=f"refined {sum(r['status'] == 'refined' for r in rows)}")
    rows.sort(key=lambda r: (r["subject"], r["expression"]))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("subject", "expression", "status"), k))
    write_csv(root / "refine.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    done = [r for r in rows if r["status"] == "refined"]
    if done:
        print(f"[refine] {len(done)}/{len(rows)} refined; median residual "
              f"{np.median([r['median_after_mm'] for r in done]):.2f} mm (before "
              f"{np.median([r['median_before_mm'] for r in done]):.2f} mm)", flush=True)
    _update_info(root, {"stage_refine": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "refined": len(done), "tuples": len(rows)}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["clean", "refine", "degrade", "all"], default="all")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    if a.stage in ("clean", "all"):
        stage_clean(a.root, a.subjects, a.workers)
    if a.stage in ("refine", "all"):
        stage_refine(a.root, a.workers)
    if a.stage in ("degrade", "all"):
        stage_degrade(a.root, a.workers)


if __name__ == "__main__":
    main()
