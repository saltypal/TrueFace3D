"""Run bounded, restartable NoW matrices and compare matched observations."""

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image

from .data_now import PROJECT, load_face_box
from .evaluation import Evaluator, LOCAL_SIMILARITY
from .robustness.degrade import Degradation, make_compound_variant, make_face_width_variant, make_variant

MODEL_IDS = ("vits", "vitb", "deca", "pixel3dmm", "mica")
WSL_DISTRO = "Ubuntu-26.04"
WSL_PYTHON = "/home/Satya/.venvs/trueface3d-vit/bin/python"


@dataclass(frozen=True)
class Variant:
    name: str
    width_px: int | None = None
    blur_fraction: float | None = None
    compound: bool = False


ORIGINAL = Variant("original")
DEGRADED = (
    Variant("face-64", width_px=64),
    Variant("blur-mild", blur_fraction=0.005),
    Variant("blur-heavy", blur_fraction=0.015),
    Variant("face-128", width_px=128),
    Variant("face-32", width_px=32),
)
# Kaggle benchmark conditions; the names above keep their original behaviour.
FACE_16 = Variant("face-16", width_px=16)
COMPOUND = Variant("compound", width_px=48, compound=True)
KAGGLE_VARIANTS = (ORIGINAL, Variant("face-64", width_px=64), Variant("face-32", width_px=32), FACE_16, COMPOUND)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    asset_digest: str
    config_digest: str
    worker_argv: tuple[str, ...] = ()
    runtime_id: str = ""
    device: str = ""
    precision: str = ""
    mesh_unit_to_mm: float | None = None

    @property
    def config_id(self):
        return _digest(asdict(self))


@dataclass
class RunRecord:
    run_id: str
    attempt_id: str
    image_key: str
    subject_id: str
    variant_id: str
    model_id: str
    model_config_id: str
    fit_status: str
    score_status: str
    timing_scope: str
    runtime_id: str
    cost: dict
    case_dir: str
    score_file: str | None
    error_stage: str | None
    input_sha256: str
    reference_sha256: str
    input_metadata: dict
    message: str = ""


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_table(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (tuple, list, dict))
                       else value for key, value in row.items()} for row in rows)


def _wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}" if os.name == "nt" else str(path)


ASSETS = {
    "vits": ("RealDenseFace", ("weights/realdenseface/vits.pth", "weights/facebox/face_box.pth",
                               "weights/flame/flame2023.pkl", "weights/flame/flame_assets.npz",
                               "configs/model/vits.yaml", "configs/fitting/single_image.yaml")),
    "vitb": ("RealDenseFace", ("weights/realdenseface/vitb.pth", "weights/facebox/face_box.pth",
                               "weights/flame/flame2023.pkl", "weights/flame/flame_assets.npz",
                               "configs/model/vitb.yaml", "configs/fitting/single_image.yaml")),
    "deca": ("DECA", ("data/deca_model.tar", "data/generic_model.pkl", "data/landmark_embedding.npy")),
    "pixel3dmm": ("Pixel3DMM", ("pretrained_weights/uv.ckpt", "pretrained_weights/normals.ckpt",)),
    "mica": ("Pixel3DMM/src/pixel3dmm/preprocessing/MICA",
             ("data/pretrained/mica.tar", "data/FLAME2020/generic_model.pkl",
              "data/FLAME2020/landmark_embedding.npy")),
}


@lru_cache(maxsize=None)
def _asset_hashes(model_id):
    upstream, members = ASSETS[model_id]
    root = PROJECT / "third_party" / upstream
    revision = "checkout-unavailable"
    if root.is_dir():
        try:
            owner = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                                            text=True, stderr=subprocess.DEVNULL).strip()
            if Path(owner).resolve() == root.resolve():
                revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                                   text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    paths = {name: root / name for name in members}
    if model_id in {"deca", "mica", "pixel3dmm"}:
        from .geometry.baselines import SPECS
        paths.update({str(path.relative_to(PROJECT)): path for path in SPECS[model_id][2]})
    return revision, _digest({name: file_digest(path) if path.is_file() else None for name, path in paths.items()})


def _runtime_digest(model_id):
    names = ["benchmark.py", "data_now.py", "cli.py", "evaluation.py", "robustness/degrade.py"]
    geometry = ("backend.py", "worker.py", "artifacts.py", "camera.py") if model_id.startswith("vit") else (
        "baselines.py", "baseline_worker.py")
    if model_id == "pixel3dmm":
        geometry += ("baseline_pixel3dmm.py",)
    names.extend(f"geometry/{name}" for name in geometry)
    sources = {name: file_digest(PROJECT / "src" / name) for name in names
               if (PROJECT / "src" / name).is_file()}
    runtime = {name: os.getenv(name) for name in (
        f"TRUEFACE3D_{model_id.upper()}_PYTHON", "TRUEFACE3D_COMPARATOR_DEVICE",
        "TRUEFACE3D_PIXEL3DMM_WSL_DISTRO", "TRUEFACE3D_PIXEL3DMM_WSL_PYTHON",
        "TRUEFACE3D_PIXEL3DMM_WSL_PYTHONPATH",
    )}
    return _digest({"source": sources, "runtime": runtime})


def model_specs(model_ids=MODEL_IDS):
    specs = []
    for model_id in model_ids:
        revision, assets = _asset_hashes(model_id)
        if model_id in {"vits", "vitb"}:
            argv = (WSL_PYTHON, "-m", "src.cli", "--model", model_id, "reconstruct",
                    "--device", "cuda", "--python", WSL_PYTHON, "--precision", "fp32-compat")
            if os.name == "nt":
                argv = ("wsl", "-d", WSL_DISTRO, "--cd", _wsl_path(PROJECT), "--", *argv)
            runtime, device, precision = "wsl-mx330", "cuda", "fp32-compat"
            if os.name != "nt":
                # Linux (Kaggle): run the worker directly, no wsl.exe.
                python = os.getenv("TRUEFACE3D_VIT_PYTHON", sys.executable)
                argv = (python, "-m", "src.cli", "--model", model_id, "reconstruct",
                        "--device", "cuda", "--python", python, "--precision", "fp32-compat")
                runtime = "linux-cuda"
        else:
            argv = ()
            runtime, device, precision = "windows-comparator", "cpu", "fp32"
            if model_id in {"deca", "mica"}:
                device = os.getenv("TRUEFACE3D_COMPARATOR_DEVICE", device)
                runtime += "-" + device
            elif os.getenv("TRUEFACE3D_PIXEL3DMM_WSL_DISTRO"):
                device, runtime = "cuda", "wsl-pixel3dmm"
        specs.append(ModelSpec(model_id, revision, assets, _runtime_digest(model_id), argv, runtime, device, precision))
    return specs


def _input(case, variant, input_dir):
    with Image.open(case.image_path) as opened:
        if opened.getexif().get(274, 1) != 1:
            raise ValueError("Photo and crop coordinates need a verified EXIF orientation policy")
        box = load_face_box(case.box_path, opened.size)
        width = box[2] - box[0]
        if variant.compound:
            generated, meta = make_compound_variant(opened, box, target_px=variant.width_px)
        elif variant.width_px is not None:
            generated, meta = make_face_width_variant(opened, box, variant.width_px)
        elif variant.blur_fraction is not None:
            radius = width * variant.blur_fraction
            generated, meta = make_variant(opened, Degradation(blur=radius), seed=0)
            meta.update(achieved_face_width_px=width, blur_radius_px=radius)
        else:
            return case.image_path, {"original_size": opened.size, "variant_size": opened.size,
                                     "achieved_face_width_px": width, "face_box": box}
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{variant.name}.png"
    if path.is_file():
        with Image.open(path) as saved:
            if saved.size != generated.size or saved.tobytes() != generated.tobytes():
                raise ValueError(f"Existing generated input disagrees with this case: {path}")
    else:
        with path.open("xb") as stream:
            generated.save(stream, format="PNG")
    return path, meta


def _fit(spec, input_path, attempt_dir, timeout):
    if not spec.worker_argv:
        from .geometry.baselines import run_baseline
        return run_baseline(spec.model_id, input_path, attempt_dir, timeout=timeout)

    bundle = attempt_dir / "fit"
    first = spec.worker_argv.index("--") + 1 if spec.worker_argv[0] == "wsl" else 0
    command = (*spec.worker_argv[:first], "timeout", "--kill-after=10s", f"{round(timeout, 2)}s",
               *spec.worker_argv[first:], "--input", _wsl_path(input_path), "--output", _wsl_path(bundle))
    _json(attempt_dir / "command.json", list(command))
    start = time.perf_counter()
    with (attempt_dir / "worker.log").open("wb") as stream:
        result = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout + 25, check=False)
    run = _read_json(bundle / "run.json", {})
    measurement = _read_json(bundle / "timing.json", {})
    stages = measurement.get("seconds", {})
    scope = "cold-cli-wall"
    cost = {**measurement.get("runtime", {}), "stage_seconds": stages,
            "cli_wall_seconds": time.perf_counter() - start,
            "gpu_index": os.getenv("TRUEFACE3D_GPU_INDEX"), "end_to_end_wall_seconds": time.perf_counter() - start}
    measured = ("prior_detection_and_inference", "camera_search_and_solver", "mesh_decode")
    if all(stages.get(name) for name in measured):
        scope = "synchronized-first-fit-through-mesh"
        cost["seconds"] = sum(stages[name][0] for name in measured)
    if result.returncode == 0 and (bundle / "face.npz").is_file():
        return {"fit_status": "success", "timing_scope": scope, "cost": cost, "mesh_unit_to_mm": 1000.0}
    log = bundle / "upstream.log"
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:] if log.is_file() else []
    detail = " | ".join(tail[-4:]) or run.get("error") or f"Exited {result.returncode}"
    stage = "cuda_oom" if "out of memory" in detail.lower() else "timeout" if result.returncode == 124 else "worker"
    return {"fit_status": "failed", "error_stage": stage, "message": detail, "cost": cost}


def _active_artifact(root, name):
    for file in (root / name, root / "fit" / name):
        if file.is_file():
            return file
    return None


def _score(case, root, result, evaluator, protocol, fit_dir=None):
    if result["fit_status"] != "success":
        return "fit_unavailable", None, ""
    if case.scan_landmark_path is None:
        return "missing_scan_landmarks", None, ""
    neutral = _active_artifact(fit_dir or root, "neutral.obj")
    points = _active_artifact(fit_dir or root, "neutral_landmarks.npy")
    if neutral is None or points is None:
        return "missing_neutral_mesh_or_landmarks", None, ""
    try:
        score = evaluator.score_now(case.scan_path, case.scan_landmark_path, neutral, points,
                                    protocol=protocol, unit_to_mm=result.get("mesh_unit_to_mm"))
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        return "evaluation_failed", None, str(exc)
    stats = {key: value for key, value in asdict(score).items()
             if key not in {"distances_mm", "alignment"}}
    with (root / "distances.npz").open("wb") as stream:
        np.savez_compressed(stream, distances_mm=score.distances_mm,
                            alignment=score.alignment if score.alignment is not None else np.eye(0))
    path = root / "score.json"
    _json(path, stats)
    return "success", str(path), ""


def read_records(run_dir):
    path = Path(run_dir) / "attempts.jsonl"
    return [RunRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def run_matrix(cases, specs, variants, run_dir, *, evaluator=None, protocol=LOCAL_SIMILARITY,
               budget_seconds=None, timeout=300, runner=None):
    """Resume identical success rows and append a new attempt for anything else."""
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "request.json"
    cases, specs, variants = tuple(cases), tuple(specs), tuple(variants)
    if not cases or not specs or not variants:
        raise ValueError("Cases, specs and variants must be nonempty")
    identity = {
        "cases": [{"key": case.image_key, "image": file_digest(case.image_path),
                   "box": file_digest(case.box_path), "scan": file_digest(case.scan_path),
                   "landmarks": file_digest(case.scan_landmark_path) if case.scan_landmark_path else None}
                  for case in cases],
        "models": [spec.config_id for spec in specs],
        # The compound flag is omitted when false so older run requests keep matching.
        "variants": [{k: v for k, v in asdict(variant).items() if not (k == "compound" and not v)}
                     for variant in variants],
        "protocol": protocol,
    }
    if manifest.is_file() and _read_json(manifest) != identity:
        raise ValueError("Run request changed; use a new run directory")
    if not manifest.is_file():
        _json(manifest, identity)
    previous = read_records(root)
    latest = {(row.image_key, row.model_id, row.variant_id): row for row in previous}
    evaluator = evaluator or Evaluator()
    runner = runner or _fit
    deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
    active = []
    for variant in variants:
        for case, source in zip(cases, identity["cases"]):
            slug = _digest(case.image_key)[:16]
            ref = _digest([source["scan"], source["landmarks"]])
            try:
                input_path, metadata = _input(case, variant, root / "inputs" / slug)
                input_sha = file_digest(input_path)
                input_error = None
            except (OSError, ValueError) as exc:
                input_error, metadata, input_sha = str(exc), {}, ""
            for spec in specs:
                key = case.image_key, spec.model_id, variant.name
                last = latest.get(key)
                old_fit = Path(last.case_dir) if last else None
                reuse = (last and last.fit_status == "success" and last.model_config_id == spec.config_id
                         and last.input_sha256 == input_sha and any(
                             _active_artifact(old_fit, name) for name in ("face.npz", "neutral.obj")))
                if reuse and last.score_status == "success" and not (
                        last.score_file and Path(last.score_file).is_file()
                        and Path(last.score_file).with_name("distances.npz").is_file()):
                    reuse = False
                if reuse and last.score_status != "evaluation_failed":
                    active.append(last)
                    continue
                attempt_id = uuid.uuid4().hex[:12]
                folder = root / "cases" / slug / variant.name / spec.model_id / attempt_id
                folder.mkdir(parents=True, exist_ok=False)
                print(f"{attempt_id} {spec.model_id} {variant.name} {case.image_key}", flush=True)
                if input_error:
                    result = {"fit_status": "input_failed", "error_stage": "variant", "message": input_error}
                elif deadline is not None and time.monotonic() >= deadline:
                    result = {"fit_status": "unattempted_budget", "error_stage": "budget"}
                else:
                    limit = timeout if deadline is None else min(timeout, max(1, deadline - time.monotonic()))
                    try:
                        result = runner(spec, input_path, folder, limit)
                    except (ValueError, OSError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
                        result = {"fit_status": "failed", "error_stage": "worker", "message": str(exc)}
                _json(folder / "fit_result.json", result)
                score_status, score_file, warning = _score(case, folder, result, evaluator, protocol)
                costs = {**result.get("cost", {})}
                if result.get("seconds") is not None:
                    costs["seconds"] = result["seconds"]
                for name in ("inference_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                             "detector_device", "detector", "memory_scope", "input_size", "failure_elapsed_seconds"):
                    if result.get(name) is not None:
                        costs[name] = result[name]
                costs.setdefault("gpu_index", os.getenv("TRUEFACE3D_GPU_INDEX"))
                costs.update(device=result.get("fit_device", result.get("device", spec.device)),
                             precision=result.get("precision", spec.precision))
                record = RunRecord(
                    root.name, attempt_id, case.image_key, case.subject_id, variant.name, spec.model_id,
                    spec.config_id, result["fit_status"], score_status,
                    result.get("timing_scope", costs.get("timing_scope", "cold-cli-wall")),
                    result.get("runtime_id", spec.runtime_id), costs, str(folder), score_file,
                    result.get("error_stage"), input_sha, ref, metadata,
                    " | ".join(filter(None, [result.get("message", ""), warning]))
                )
                _json(folder / "record.json", asdict(record))
                with (root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")
                previous.append(record)
                active.append(record)
    table = [asdict(row) for row in active]
    _json(root / "results.json", table)
    write_table(root / "results.csv", table)
    write_table(root / "summary.csv", summarize_runs(active, protocol))
    return active


def result_records(run_dir):
    path = Path(run_dir) / "results.json"
    return [RunRecord(**row) for row in _read_json(path, [])]


def result_with_retries(run_dir, *retry_dirs):
    """Use separately recorded retry rows only when they ran the identical inputs."""
    def key(row):
        return row.model_id, row.image_key, row.variant_id

    rows = {key(row): row for row in result_records(run_dir)}
    requests = {}
    for directory in retry_dirs:
        for row in result_records(directory):
            prior = rows.get(key(row))
            if (prior is None or prior.input_sha256 != row.input_sha256
                    or prior.reference_sha256 != row.reference_sha256
                    or _case_request(prior, requests) != _case_request(row, requests)):
                raise ValueError(f"Retry does not match the original matrix inputs: {key(row)}")
            rows[key(row)] = row
    return list(rows.values())


def rescore_run(run_dir, output_dir, cases, *, evaluator=None, protocol=LOCAL_SIMILARITY,
                retry_dirs=()):
    """Score retained meshes against newly available annotations in a new evidence folder."""
    source = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    retries = tuple(Path(path).resolve() for path in retry_dirs)
    records = result_with_retries(source, *retries)
    if not records:
        raise ValueError(f"No saved results in {source}")
    case_map = {case.image_key: case for case in cases}
    evaluator = evaluator or Evaluator()
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "request.json", {
        "replays": str(source), "retries": [str(path) for path in retries], "protocol": protocol,
    })
    fingerprints, requests, scored = {}, {}, []
    for record in records:
        case = case_map[record.image_key]
        if record.image_key not in fingerprints:
            fingerprints[record.image_key] = {"image": file_digest(case.image_path),
                                              "box": file_digest(case.box_path),
                                              "scan": file_digest(case.scan_path)}
        request = _case_request(record, requests)
        if not request or any(request[key] != val for key, val in fingerprints[record.image_key].items()):
            raise ValueError(f"Retained fit no longer matches the source/scan: {record.image_key}")
        folder = output / "scores" / record.attempt_id
        folder.mkdir(parents=True, exist_ok=False)
        fit = Path(record.case_dir)
        proof = _read_json(fit / "fit_result.json") or _read_json(fit / "baseline_result.json", {})
        status, path, message = _score(case, folder, {**proof, "fit_status": record.fit_status},
                                       evaluator, protocol, fit_dir=fit)
        ref = _digest([fingerprints[record.image_key]["scan"],
                       file_digest(case.scan_landmark_path) if case.scan_landmark_path else None])
        scored.append(replace(record, run_id=output.name, score_status=status, score_file=path,
                              reference_sha256=ref, message=" | ".join(filter(None, [record.message, message]))))
    table = [asdict(row) for row in scored]
    _json(output / "results.json", table)
    write_table(output / "results.csv", table)
    write_table(output / "summary.csv", summarize_runs(scored, protocol))
    return scored


def _score_values(record):
    return _read_json(Path(record.score_file), {}) if record.score_file else {}


def summarize_runs(records, protocol=LOCAL_SIMILARITY):
    groups = defaultdict(list)
    for row in records:
        groups[(row.model_id, row.variant_id)].append(row)
    results = []
    for (model, variant), rows in groups.items():
        subjects = defaultdict(list)
        for row in rows:
            score = _score_values(row)
            if score.get("protocol") == protocol:
                subjects[row.subject_id].append(score["mean_mm"])
        results.append({
            "model": model, "variant": variant, "requested": len(rows),
            "attempted": sum(row.fit_status in {"success", "failed"} for row in rows),
            "fitted": sum(row.fit_status == "success" for row in rows),
            "scored": sum(map(len, subjects.values())),
            "subject_mean_mm": float(np.mean([np.mean(values) for values in subjects.values()])) if subjects else None,
        })
    return results


def _case_request(record, requests):
    if len(Path(record.case_dir).parents) < 5:
        return None
    root = Path(record.case_dir).parents[4]
    if root not in requests:
        manifest = _read_json(root / "request.json", {})
        requests[root] = {row["key"]: row for row in manifest.get("cases", [])}
    return requests[root].get(record.image_key)


def pair_variants(records, original_records):
    originals = {(r.model_id, r.image_key): r for r in original_records if r.variant_id == "original"}
    pairs = []
    requests = {}
    for row in records:
        source = originals.get((row.model_id, row.image_key))
        original_score, variant_score = _score_values(source) if source else {}, _score_values(row)
        status = "scored"
        if source is None:
            status = "no_original"
        elif source.model_config_id != row.model_config_id:
            status = "model_config_mismatch"
        else:
            start, end = (_case_request(record, requests) for record in (source, row))
            if (not start or not end or any(start[field] != end[field] for field in ("image", "box", "scan"))
                    or source.input_sha256 != start["image"]):
                status = "source_request_mismatch"
            elif source.reference_sha256 != row.reference_sha256:
                status = "reference_mismatch"
            elif source.fit_status != "success" or row.fit_status != "success":
                status = "fit_unavailable"
            elif not original_score or not variant_score:
                status = "score_unavailable"
            elif original_score.get("protocol") != variant_score.get("protocol"):
                status = "protocol_mismatch"
        pairs.append({
            "model": row.model_id, "image": row.image_key, "subject": row.subject_id,
            "condition": row.variant_id, "face_width_px": row.input_metadata.get("achieved_face_width_px"),
            "original_mm": original_score.get("mean_mm"), "degraded_mm": variant_score.get("mean_mm"),
            "delta_mm": variant_score["mean_mm"] - original_score["mean_mm"] if status == "scored" else None,
            "pair_status": status,
            "fit_status": row.fit_status, "score_status": row.score_status,
            "original_run": source.run_id if source else None, "degraded_run": row.run_id,
        })
    return pairs
