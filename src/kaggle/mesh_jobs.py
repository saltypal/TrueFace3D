"""Mesh benchmark jobs for notebook 11 (account A). Reuses benchmark.run_matrix.

Job kinds (dicts, JSON-serialisable so src.kaggle.runner can ship them to GPU workers):
  {"kind": "matrix", "set": "clean"|"real", "model": id, "keys": [...], "variants": [...], "out": dir}
  {"kind": "oracle", "key": image_key, "mode": coord|depth|both, "out": dir, "clean_priors": path}
"""
import os
from pathlib import Path

from ..benchmark import (COMPOUND, FACE_16, KAGGLE_VARIANTS, ORIGINAL, Variant, _active_artifact, _fit,
                         model_specs, read_records, run_matrix)
from ..data_now import installed_cases, pilot_cases, real_world_cases
from . import paths

VARIANTS = {v.name: v for v in KAGGLE_VARIANTS}
MODELS = ("vits", "vitb", "deca", "pixel3dmm", "mica")
ORACLE_MODES = ("coord", "depth", "both")


def case_sets(data_root=None, count=20):
    cases = installed_cases(data_root or paths.data_root())
    return {"clean": pilot_cases(cases, count=count), "real": real_world_cases(cases)}


def chunk(items, parts):
    return [items[i::parts] for i in range(parts) if items[i::parts]]


def matrix_jobs(out, sets, models=MODELS, chunks_clean=4, chunks_real=4):
    jobs = []
    for set_name, parts, variants in (("clean", chunks_clean, [v.name for v in KAGGLE_VARIANTS]),
                                      ("real", chunks_real, ["original"])):
        keys = [c.image_key for c in sets[set_name]]
        for model in models:
            for index, part in enumerate(chunk(keys, parts)):
                jobs.append({"kind": "matrix", "set": set_name, "model": model, "keys": part,
                             "variants": variants, "out": str(Path(out) / "runs" / f"{set_name}-{model}-c{index}")})
    return jobs


def _cases_by_key(keys):
    wanted = set(keys)
    cases = installed_cases(paths.data_root())
    return [c for c in cases if c.image_key in wanted]


def run_matrix_job(job):
    cases = _cases_by_key(job["keys"])
    variants = [VARIANTS[name] for name in job["variants"]]
    return run_matrix(cases, model_specs((job["model"],)), variants, job["out"], timeout=900)


def clean_vits_priors(out, image_key):
    """Locate priors.npz of the clean ViT-S fit for one image (searches the clean vits run dirs)."""
    for run in sorted(Path(out, "runs").glob("clean-vits-c*")):
        for row in read_records(run):
            if (row.image_key == image_key and row.variant_id == "original" and row.fit_status == "success"):
                found = _active_artifact(Path(row.case_dir), "priors.npz")
                if found:
                    return str(found)
    return None


def oracle_jobs(out, sets):
    jobs = []
    for case in sets["clean"]:
        priors = clean_vits_priors(out, case.image_key)
        for mode in ORACLE_MODES:
            jobs.append({"kind": "oracle", "key": case.image_key, "mode": mode, "clean_priors": priors,
                         "out": str(Path(out) / "runs" / f"oracle-{mode}-{case.subject_id}")})
    return jobs


def run_oracle_job(job):
    """Fit the degraded variants of one image with clean priors substituted (ViT-S only)."""
    if not job.get("clean_priors"):
        raise RuntimeError("No clean ViT-S priors for " + job["key"])
    env = {"TRUEFACE3D_PRIORS_IN": job["clean_priors"], "TRUEFACE3D_PRIORS_MODE": job["mode"]}

    def runner(spec, input_path, folder, limit):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return _fit(spec, input_path, folder, limit)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    cases = _cases_by_key([job["key"]])
    degraded = [v for v in KAGGLE_VARIANTS if v.name != "original"]
    return run_matrix(cases, model_specs(("vits",)), degraded, job["out"], runner=runner, timeout=900)


def run_job(job):
    return run_oracle_job(job) if job["kind"] == "oracle" else run_matrix_job(job)


def collect_records(out):
    """All attempts (in order) under out/runs/*, tagged with their run name."""
    rows = []
    for run in sorted(Path(out, "runs").glob("*")):
        for rec in read_records(run):
            rows.append((run.name, rec))
    return rows
