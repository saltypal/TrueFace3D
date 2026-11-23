"""Appearance jobs for notebook 12 (account B) and the raw-projection step of notebook 11."""
import json
from pathlib import Path

from ..appearance import backends, evaluate
from ..benchmark import KAGGLE_VARIANTS, _input
from ..data_now import installed_cases
from . import paths
from .mesh_jobs import case_sets

METHODS = ("deca", "opendelight", "trust")
VARIANTS = {v.name: v for v in KAGGLE_VARIANTS}


def jobs(out, methods=METHODS, count=20):
    keys = [c.image_key for c in case_sets(count=count)["clean"]]
    return [{"method": m, "key": k, "variant": v.name, "out": str(Path(out) / m)}
            for m in methods for k in keys for v in KAGGLE_VARIANTS]


def attempt_dir(job):
    return Path(job["out"]) / Path(job["key"]).parts[0] / job["variant"]


def run_job(job):
    """Run one backend on one input; skip if a result.json already exists (resumable)."""
    folder = attempt_dir(job)
    done = folder / "result.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    case = next(c for c in installed_cases(paths.data_root()) if c.image_key == job["key"])
    image, meta = _input(case, VARIANTS[job["variant"]], Path(job["out"]) / "inputs" / case.subject_id)
    result = backends.load(job["method"]).run(image, folder, box=meta.get("face_box"))
    result.update(job=job, input=str(image), input_metadata=meta)
    if result["status"] == "ok" and job["variant"] == "original":
        result["metrics_vs_photo"] = evaluate.score_clean(folder, case.image_path)
    folder.mkdir(parents=True, exist_ok=True)
    evaluate.write(done, result)
    return result


def score_pairs(out, methods=METHODS):
    """Mode 2 for every degraded attempt that has a clean result of the same method."""
    rows = []
    for method in methods:
        for folder in sorted(Path(out, method).glob("*/*")):
            if folder.name in {"original", "inputs"} or not (folder / "result.json").is_file():
                continue
            ref = folder.parent / "original"
            if (ref / "result.json").is_file() and json.loads((ref / "result.json").read_text())["status"] == "ok":
                score = evaluate.score_pair(ref, folder)
                rows.append({"method": method, "subject": folder.parent.name, "condition": folder.name, **score})
    evaluate.write(Path(out) / "appearance_pairs.json", rows)
    return rows


def raw_projection_from_vits(tagged_records, out):
    """Project each input photo onto its already-fitted ViT-S mesh (no second fit); then score like the others."""
    from ..benchmark import _active_artifact
    cases = {c.image_key: c for c in installed_cases(paths.data_root())}
    method = backends.load("raw_projection")
    results = []
    for run, rec in tagged_records:
        if rec.model_id != "vits" or rec.fit_status != "success" or not run.startswith("clean-"):
            continue
        face = _active_artifact(Path(rec.case_dir), "face.npz")
        if face is None:
            continue
        job = {"method": "raw_projection", "key": rec.image_key, "variant": rec.variant_id, "out": str(Path(out) / "raw_projection")}
        folder = attempt_dir(job)
        if (folder / "result.json").is_file():
            continue
        case = cases[rec.image_key]
        image, meta = _input(case, VARIANTS[rec.variant_id], Path(out) / "raw_projection" / "inputs" / case.subject_id)
        result = method.run(image, folder, fit_dir=face.parent)
        result.update(job=job, input=str(image), input_metadata=meta)
        if result["status"] == "ok" and rec.variant_id == "original":
            result["metrics_vs_photo"] = evaluate.score_clean(folder, case.image_path)
        evaluate.write(folder / "result.json", result)
        results.append(result)
    return results


def export_obj_glbs(out):
    """GLB for every attempt that produced a textured mesh.obj but no GLB yet (trimesh; best effort)."""
    import trimesh
    made = []
    for obj in sorted(Path(out).glob("*/*/*/mesh.obj")):
        target = obj.with_name("mesh.glb")
        if not target.exists():
            try:
                trimesh.load(obj, process=False).export(target)
                made.append(str(target))
            except Exception as exc:  # noqa: BLE001
                print("GLB export failed for", obj, exc)
    return made
