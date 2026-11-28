"""FaceScape-C mesh benchmark jobs (notebook K10). Fits only; scoring against the TU ground truth is a
separate CPU step (src.facescape.score / notebook 32) so a GPU session never fails on evaluation.

One job = one model x one condition over every selected image, so progress lines arrive often and a
deadline cut leaves whole conditions done. Every fit keeps its full bundle (meshes, priors, renders,
logs) under out/fits/<model>/<condition>/<image_id>/, plus record.json. Re-running skips finished fits.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from ..facescape import constants as C
from ..facescape.dataset import FaceScapeC

MODELS_ON_KAGGLE = ("vits", "vitb", "deca", "emoca", "mica")   # SMIRK, HRN: adapters pending


def benchmark_cases(ds: FaceScapeC, subjects=None, expressions=None, views=None, conditions=None):
    """Clean + degraded rows of the test grid: test subjects x test expressions x 3 views x 43 conditions."""
    subjects = set(int(s) for s in (subjects or [r["subject"] for r in ds.subjects if r["split"] == "test"]))
    expressions = set(int(e) for e in (expressions or C.TEST_EXPRESSIONS))
    views = set(views or C.VIEWS)
    rows = []
    for r in ds.clean_rows + ds.degraded_rows:
        if (int(r["subject"]) in subjects and int(r["expression"]) in expressions and r["view"] in views
                and r["split"] == "test" and (conditions is None or r["condition"] in conditions)):
            rows.append(r)
    return rows


def condition_order(rows):
    names = sorted({r["condition"] for r in rows} - {"clean"})
    return (["clean"] if any(r["condition"] == "clean" for r in rows) else []) + names


def make_jobs(rows, models, out):
    jobs = []
    for model in models:
        for cond in condition_order(rows):
            items = [[r["image_id"], cond, r["path"]] for r in rows if r["condition"] == cond]
            jobs.append({"kind": "facescape", "model": model, "variant": cond, "items": items, "out": str(out)})
    return jobs


def _files(folder: Path) -> dict:
    return {str(p.relative_to(folder)): p.stat().st_size for p in sorted(folder.rglob("*")) if p.is_file()}


def run_job(job):
    """Runs inside a GPU worker. Never raises for a single failed fit; the record says what happened."""
    from ..benchmark import _fit, model_specs
    ds_root = FaceScapeC.find().root
    spec = model_specs((job["model"],))[0]
    deadline = float(os.environ.get("TRUEFACE3D_DEADLINE", "inf"))
    gpu = os.environ.get("TRUEFACE3D_GPU_INDEX", "0")
    log = Path(job["out"]) / "records" / f"{job['model']}_gpu{gpu}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    for image_id, cond, rel in job["items"]:
        folder = Path(job["out"]) / "fits" / job["model"] / cond / image_id
        record_path = folder / "record.json"
        if record_path.is_file():
            continue
        record = {"model": job["model"], "image_id": image_id, "condition": cond, "input": rel, "gpu": gpu,
                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if time.time() > deadline:
            record.update(fit_status="not_run", error_stage="session_deadline")
        else:
            if folder.exists():                       # leftovers of an interrupted attempt
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            start = time.time()
            try:
                result = _fit(spec, ds_root / rel, folder, timeout=600)
            except Exception as exc:  # noqa: BLE001
                result = {"fit_status": "failed", "error_stage": "exception", "message": f"{type(exc).__name__}: {exc}"}
            record.update({k: v for k, v in result.items()})
            record["wall_seconds"] = round(time.time() - start, 2)
            record["files"] = _files(folder)
            record["has_priors"] = any(name.endswith("priors.npz") for name in record["files"])
        if record["fit_status"] != "not_run":
            record_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in record.items() if k != "files"}, default=str) + "\n")
        print(f"[fit] {job['model']} {cond} {image_id} {record['fit_status']} "
              f"{record.get('wall_seconds', 0):.0f}s {record.get('message', '')[:120]}", flush=True)


def read_records(out) -> list[dict]:
    rows = []
    for p in sorted(Path(out, "fits").glob("*/*/*/record.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows
