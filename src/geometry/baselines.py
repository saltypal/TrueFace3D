"""Run the Chapter 1 comparators in isolated native-model processes."""

import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from time import time


ROOT = Path(__file__).resolve().parents[2]
PIXEL = ROOT / "third_party" / "Pixel3DMM"
MICA = PIXEL / "src/pixel3dmm/preprocessing/MICA"
DECA = ROOT / "third_party" / "DECA"
SCRFD = ROOT / "checkpoints/insightface/models/antelopev2/scrfd_10g_bnkps.onnx"
RUNTIME = ROOT / "checkpoints/comparator_runtime"
# EMOCA v2 (Danecek et al. 2022), EMOCA_v2_lr_mse_20 detail checkpoint; licensed, never in git.
EMOCA_CKPT = Path(os.environ.get("TRUEFACE3D_EMOCA_CKPT", ROOT / "checkpoints/emoca/EMOCA_v2_lr_mse_20.ckpt"))
DEADLINE_ENV = "TRUEFACE3D_COMPARATOR_DEADLINE"

SPECS = {
    "deca": (
        DECA, "a11554ae2a2b0f3998cf1fa94dd4db03babb34a2",
        (DECA / "data/deca_model.tar", DECA / "data/generic_model.pkl",
         DECA / "data/landmark_embedding.npy", SCRFD),
        ("torch", "torchvision", "cv2", "onnxruntime", "skimage.transform", "chumpy"),
        "cpu",
    ),
    "emoca": (
        DECA, "a11554ae2a2b0f3998cf1fa94dd4db03babb34a2",
        (EMOCA_CKPT, DECA / "data/generic_model.pkl", DECA / "data/landmark_embedding.npy", SCRFD),
        ("torch", "torchvision", "cv2", "onnxruntime", "skimage.transform", "chumpy"),
        "cpu",
    ),
    "mica": (
        MICA, "af22e7a5810d474bc28a1433db533723d6bd2b07",
        (MICA / "data/pretrained/mica.tar", MICA / "data/FLAME2020/generic_model.pkl",
         MICA / "data/FLAME2020/landmark_embedding.npy", SCRFD),
        ("torch", "cv2", "onnxruntime", "skimage.transform", "chumpy", "loguru"),
        "cpu",
    ),
    "pixel3dmm": (
        PIXEL, "fcd1fa973c7715b02a8948dfc679dff53cf85924",
        (PIXEL / "pretrained_weights/uv.ckpt", PIXEL / "pretrained_weights/normals.ckpt",
         PIXEL / "src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth",
         PIXEL / "src/pixel3dmm/preprocessing/PIPNet/snapshots/WFLW/"
         "pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth",
         MICA / "data/pretrained/mica.tar", MICA / "data/FLAME2020/generic_model.pkl",
         MICA / "data/FLAME2020/landmark_embedding.npy", SCRFD),
        ("torch", "torchvision", "cv2", "onnxruntime", "skimage.transform", "skimage.io",
         "loguru", "chumpy", "environs", "mediapy", "tyro", "distinctipy", "tqdm",
         "pytorch_lightning", "timm", "einops", "wandb", "tensorboard", "trimesh",
         "pyvista", "dreifus", "pytorch3d._C", "nvdiffrast.torch"),
        "cuda",
    ),
}


def requested_device(model_id):
    if model_id == "pixel3dmm":
        return "cuda"
    device = os.environ.get("TRUEFACE3D_COMPARATOR_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("TRUEFACE3D_COMPARATOR_DEVICE must be cpu or cuda")
    return device


def _runtime(model_id):
    override = os.environ.get(f"TRUEFACE3D_{model_id.upper()}_PYTHON")
    windows_python = ROOT / ".venv-vit/Scripts/python.exe"
    python = override or (str(windows_python)
                          if os.name == "nt" and windows_python.is_file() else sys.executable)
    environment = dict(os.environ)
    paths = [str(ROOT)]
    if os.name == "nt":
        paths.append(str(RUNTIME))
    if model_id == "pixel3dmm":
        paths.extend((str(PIXEL / "src"),
                      str(PIXEL / "src/pixel3dmm/preprocessing/facer")))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["ORT_DISABLE_TELEMETRY"] = "1"
    environment["WANDB_MODE"] = "disabled"
    environment["HF_HUB_OFFLINE"] = "1"
    return python, environment


def _wsl_settings(model_id):
    distribution = os.environ.get("TRUEFACE3D_PIXEL3DMM_WSL_DISTRO")
    if model_id == "pixel3dmm" and os.name == "nt" and distribution:
        return distribution, os.environ.get("TRUEFACE3D_PIXEL3DMM_WSL_PYTHON", "python3")
    return None


def _wsl_path(path):
    path = PureWindowsPath(path)
    if not path.is_absolute() or len(path.drive) != 2 or path.drive[1] != ":":
        raise ValueError(f"Expected a Windows drive path for WSL: {path}")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def _invoke(model_id, *args, deadline=None, **kwargs):
    python, environment = _runtime(model_id)
    if deadline is not None:
        environment[DEADLINE_ENV] = str(deadline)
    if wsl := _wsl_settings(model_id):
        distribution, python = wsl
        paths = [_wsl_path(path) for path in
                 (ROOT, PIXEL / "src", PIXEL / "src/pixel3dmm/preprocessing/facer")]
        if extra := os.environ.get("TRUEFACE3D_PIXEL3DMM_WSL_PYTHONPATH"):
            paths.append(extra)
        names = ("ORT_DISABLE_TELEMETRY", "WANDB_MODE", "HF_HUB_OFFLINE", DEADLINE_ENV)
        options = [f"{key}={environment[key]}" for key in names if key in environment]
        command = ["wsl.exe", "-d", distribution, "--cd", _wsl_path(ROOT), "--exec", "env",
                   "PYTHONPATH=" + ":".join(paths), *options, python, "-m",
                   "src.geometry.baseline_worker", model_id,
                   *(_wsl_path(arg) if isinstance(arg, Path) else str(arg) for arg in args)]
        return subprocess.run(command, cwd=ROOT, check=False, **kwargs)
    return subprocess.run(
        [python, "-m", "src.geometry.baseline_worker", model_id, *map(str, args)],
        cwd=ROOT, env=environment, check=False, **kwargs,
    )


def _model_metadata(model_id):
    source, expected_revision, assets, _, _ = SPECS[model_id]
    try:
        git = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True,
            text=True, check=False,
        )
        revision = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    missing = [str(path) for path in assets if not path.is_file() or not path.stat().st_size]
    wsl = _wsl_settings(model_id)
    return {
        "model_id": model_id, "revision": revision, "expected_revision": expected_revision,
        "device": requested_device(model_id), "precision": "float32",
        "python": wsl[1] if wsl else _runtime(model_id)[0],
        "runtime_id": f"wsl:{wsl[0]}" if wsl else f"native:{os.name}",
        "assets": {str(path.relative_to(ROOT)): path.stat().st_size
                   for path in assets if str(path) not in missing},
        "missing_assets": missing,
    }


def qualify_model(model_id: str) -> dict:
    """Check locally staged source, assets and executable dependencies."""
    metadata = _model_metadata(model_id)
    try:
        probe = _invoke(model_id, "--qualify", capture_output=True, text=True, timeout=120)
        runtime = json.loads(probe.stdout) if probe.returncode == 0 else {
            "runtime_error": probe.stderr.strip() or f"Worker exited with {probe.returncode}",
        }
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        runtime = {"runtime_error": str(error)}
    ready = (metadata["revision"] == metadata["expected_revision"]
             and not metadata["missing_assets"] and runtime.get("runtime_ready", False))
    return {**metadata, **runtime, "ready": ready,
            "status": "ready_to_attempt" if ready else "blocked"}


def run_baseline(model_id: str, image_path: Path, run_dir: Path, *, timeout=2400) -> dict:
    """Fit one source image and persist diagnostics, native output and neutral geometry."""
    metadata = _model_metadata(model_id)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / "baseline_result.json"
    if any((run_dir / name).exists() for name in ("native", "neutral.obj", result_file.name)):
        raise FileExistsError(f"Comparator attempt already exists in {run_dir}")
    result = {
        **metadata, "fit_status": "failed", "seconds": None, "mesh_unit_to_mm": None,
        "timing_scope": "single_image_preprocess_inference_export_including_model_load",
    }
    with (run_dir / "comparator.stdout.log").open("wb") as stdout, (
        run_dir / "comparator.stderr.log"
    ).open("wb") as stderr:
        try:
            worker = _invoke(model_id, Path(image_path).resolve(), run_dir, stdout=stdout,
                             stderr=stderr, timeout=timeout, deadline=time() + timeout)
            if not result_file.is_file():
                result.update(error_stage="worker", message=f"Worker exited with {worker.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            result.update(error_stage="worker", message=str(error))
    if result_file.is_file():
        result.update(json.loads(result_file.read_text(encoding="utf-8")))
    temp = result_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(result_file)
    return result


def run_baseline_batch(model_id: str, items, log_dir: Path, *, timeout=3600) -> list[dict]:
    """Fit many (image_path, run_dir) pairs in ONE worker process, so the model loads once.

    Each run_dir ends with the same baseline_result.json / native / neutral.obj files as run_baseline.
    """
    metadata = _model_metadata(model_id)
    items = [(Path(i).resolve(), Path(r).resolve()) for i, r in items]
    for _, run_dir in items:
        run_dir.mkdir(parents=True, exist_ok=True)
        if any((run_dir / n).exists() for n in ("native", "neutral.obj", "baseline_result.json")):
            raise FileExistsError(f"Comparator attempt already exists in {run_dir}")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_file = log_dir / f"batch_{model_id}_{int(time() * 1000)}.json"
    batch_file.write_text(json.dumps([[str(i), str(r)] for i, r in items]), encoding="utf-8")
    error = None
    with (log_dir / "comparator.stdout.log").open("ab") as stdout, (log_dir / "comparator.stderr.log").open("ab") as stderr:
        try:
            _invoke(model_id, "--batch", batch_file, stdout=stdout, stderr=stderr,
                    timeout=timeout, deadline=time() + timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
    results = []
    for _, run_dir in items:
        result_file = run_dir / "baseline_result.json"
        result = {**metadata, "fit_status": "failed", "seconds": None, "mesh_unit_to_mm": None}
        if result_file.is_file():
            result.update(json.loads(result_file.read_text(encoding="utf-8")))
        else:
            result.update(error_stage="worker", message=error or "batch worker stopped before this image")
        result_file.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        results.append(result)
    return results
