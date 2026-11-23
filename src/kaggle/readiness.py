"""Readiness check for notebook 10: environment info, then a one-photo smoke test per model/backend per GPU.

Untested on Kaggle. Failures keep their stage and message instead of stopping the run.
"""
import json
from pathlib import Path
import shutil
import subprocess

from . import paths, runner
from .mesh_jobs import MODELS, case_sets

APPEARANCE = ("raw_projection", "deca", "opendelight", "trust")


def environment():
    info = {"paths": paths.describe()}
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        info["nvidia_smi"] = f"unavailable: {exc}"
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                         "available": torch.cuda.is_available(),
                         "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                                   "capability": torch.cuda.get_device_capability(i),
                                   "total_mib": torch.cuda.get_device_properties(i).total_memory // 2 ** 20}
                                  for i in range(torch.cuda.device_count())]}
    except Exception as exc:  # noqa: BLE001
        info["torch"] = {"error": str(exc)}
    return info


def smoke_jobs(out):
    key = case_sets(count=1)["clean"][0].image_key
    jobs = [{"what": "mesh", "name": m, "key": key, "out": str(Path(out) / "smoke")} for m in MODELS]
    jobs += [{"what": "appearance", "name": a, "key": key, "out": str(Path(out) / "smoke")} for a in APPEARANCE]
    return jobs


def smoke_job(job):
    """Runs inside a GPU worker (CUDA_VISIBLE_DEVICES already set); writes one result json."""
    import os
    gpu = os.environ.get("TRUEFACE3D_GPU_INDEX", "0")
    target = Path(job["out"]) / f"gpu{gpu}" / f"{job['what']}-{job['name']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {"what": job["what"], "name": job["name"], "gpu": gpu, "status": "failed", "detail": ""}
    try:
        if job["what"] == "mesh":
            from ..benchmark import ORIGINAL, model_specs, run_matrix
            from .mesh_jobs import _cases_by_key
            rows = run_matrix(_cases_by_key([job["key"]]), model_specs((job["name"],)), [ORIGINAL],
                              target.with_suffix(""), timeout=900)
            row = rows[0]
            result.update(status="ok" if row.fit_status == "success" else "failed",
                          detail=f"fit={row.fit_status} score={row.score_status} stage={row.error_stage} {row.message[:200]}",
                          seconds=row.cost.get("end_to_end_wall_seconds"))
        else:
            from ..appearance import backends
            from .mesh_jobs import _cases_by_key
            case = _cases_by_key([job["key"]])[0]
            out = backends.load(job["name"]).run(case.image_path, target.with_suffix(""))
            result.update(status={"ok": "ok", "not_run": "not_run"}.get(out["status"], "failed"),
                          detail=out.get("reason", ""), seconds=out.get("seconds"))
    except Exception as exc:  # noqa: BLE001
        result["detail"] = f"{type(exc).__name__}: {exc}"[:300]
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_smoke(out, gpus=(0, 1)):
    jobs = smoke_jobs(out)
    for gpu in gpus:
        runner.launch("src.kaggle.readiness:smoke_job", jobs, Path(out) / "smoke_logs", gpu_ids=[gpu])
    rows = []
    for path in sorted(Path(out, "smoke").glob("gpu*/*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def readiness_table(rows):
    """Ready only if the item passed on every GPU that ran it; not_run is reported separately."""
    names = sorted({(r["what"], r["name"]) for r in rows})
    table = []
    for what, name in names:
        mine = [r for r in rows if (r["what"], r["name"]) == (what, name)]
        states = {r["gpu"]: r["status"] for r in mine}
        status = "ready" if mine and all(s == "ok" for s in states.values()) else (
            "not_run" if all(s == "not_run" for s in states.values()) else "NOT READY")
        table.append({"kind": what, "name": name, "status": status,
                      **{f"gpu{g}": s for g, s in sorted(states.items())},
                      "detail": next((r["detail"] for r in mine if r["detail"]), "")})
    return table


def save(out, env, rows, wheels="/kaggle/working/wheels"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readiness.json").write_text(json.dumps({"environment": env, "smoke": rows,
                                                    "table": readiness_table(rows)}, indent=2), encoding="utf-8")
    if Path(wheels).is_dir():
        shutil.copytree(wheels, out / "wheels", dirs_exist_ok=True)
