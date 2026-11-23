"""Cost table for every attempt: device, GPU index, times, memory, params, GFLOPs, dtype."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..benchmark import _active_artifact


def cost_rows(tagged_records):
    """tagged_records: [(run_name, RunRecord)] in attempt order. Flags the first call per (model, GPU)."""
    seen, rows = set(), []
    for run, rec in tagged_records:
        c = rec.cost or {}
        gpu = c.get("gpu_index")
        key = (rec.model_id, gpu)
        cold = key not in seen
        seen.add(key)
        stages = {k: (v[0] if isinstance(v, (list, tuple)) and v else v) for k, v in c.get("stage_seconds", {}).items()}
        rows.append({
            "run": run, "model": rec.model_id, "condition": rec.variant_id, "image": rec.image_key,
            "subject": rec.subject_id, "fit_status": rec.fit_status, "error_stage": rec.error_stage,
            "message": (rec.message or "")[:300],
            "device": c.get("device"), "gpu_name": c.get("gpu"), "gpu_index": gpu,
            "first_call_on_this_gpu": cold, "cold_or_warm": "cold" if cold else "warm",
            "model_seconds": c.get("seconds"), "stage_seconds": stages,
            "end_to_end_wall_seconds": c.get("end_to_end_wall_seconds", c.get("cli_wall_seconds")),
            "peak_allocated_mib": _mib(c.get("peak_allocated_bytes")),
            "peak_reserved_mib": _mib(c.get("peak_reserved_bytes")),
            "dtype": c.get("precision"), "runtime": rec.runtime_id,
        })
    return rows


def _mib(value):
    return None if value is None else value / 2 ** 20


def profile_models(out_dir, vit_priors, sample_image, models=("vits", "vitb", "deca", "mica")):
    """Counted GFLOPs (via geometry.profile_flops) and parameter counts, one row per model.

    Runs profile_flops as a subprocess so a crash only marks that model. Pixel3DMM needs its
    preprocessed folders, so it is reported as 'not profiled'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for model in models:
        source = vit_priors if model.startswith("vit") else sample_image
        target = out_dir / f"{model}.json"
        if target.exists():
            target.unlink()
        run = subprocess.run([sys.executable, "-m", "src.geometry.profile_flops", model, str(source), str(target)],
                             capture_output=True, text=True)
        row = {"model": model, "gflops": None, "parameters": None, "status": "ok"}
        if run.returncode == 0 and target.is_file():
            row["gflops"] = json.loads(target.read_text(encoding="utf-8"))["counted_flops"] / 1e9
        else:
            row["status"] = "failed: " + (run.stderr.strip().splitlines() or ["?"])[-1][:200]
        if model.startswith("vit") and Path(vit_priors).is_file():
            with np.load(vit_priors, allow_pickle=False) as p:
                row["parameters"] = int(p["network_parameters"])
        rows[model] = row
    rows["pixel3dmm"] = {"model": "pixel3dmm", "gflops": None, "parameters": None, "status": "not profiled"}
    return rows


def summarize_cost(rows, profiles):
    """Per-model medians over warm attempts joined with the static GFLOPs/params."""
    groups = defaultdict(list)
    for r in rows:
        if r["fit_status"] == "success":
            groups[r["model"]].append(r)
    table = []
    for model, items in groups.items():
        warm = [r for r in items if not r["first_call_on_this_gpu"]] or items
        table.append({
            "model": model, "n_success": len(items),
            "cold_first_call_s": next((r["end_to_end_wall_seconds"] for r in items if r["first_call_on_this_gpu"]), None),
            "warm_median_wall_s": _median([r["end_to_end_wall_seconds"] for r in warm]),
            "peak_allocated_mib_max": _max([r["peak_allocated_mib"] for r in items]),
            "peak_reserved_mib_max": _max([r["peak_reserved_mib"] for r in items]),
            "gflops": profiles.get(model, {}).get("gflops"), "parameters": profiles.get(model, {}).get("parameters"),
            "dtype": items[0]["dtype"], "gpu_name": next((r["gpu_name"] for r in items if r["gpu_name"]), None),
        })
    return table


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None
