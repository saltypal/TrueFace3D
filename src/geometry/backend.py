"""Run the pinned upstream programs in their own Python environment."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

REVISION = "7d855a29931c95cc9016f75d49cead60adbcf0a3"
PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT / "third_party" / "RealDenseFace"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_backend(upstream, model):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected RealDenseFace {REVISION}, found {revision}")
    paths = {
        "flame": upstream / "weights/flame/flame2023.pkl",
        "support": upstream / "weights/flame/flame_assets.npz",
        "detector": upstream / "weights/facebox/face_box.pth",
        "checkpoint": upstream / f"weights/realdenseface/{model}.pth",
        "model_config": upstream / f"configs/model/{model}.yaml",
        "fit_config": upstream / "configs/fitting/single_image.yaml",
    }
    return {"revision": revision, "assets": {
        name: {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
        for name, path in paths.items()
    }}


def run_upstream(python, upstream, image, output, model, device, capture, precision="default"):
    if precision not in {"default", "fp32-compat"} or (precision != "default" and not capture):
        raise ValueError("fp32-compat requires the isolated capture worker")
    metadata = audit_backend(upstream, model)
    missing = [v["path"] for v in metadata["assets"].values() if v["sha256"] is None]
    if missing:
        raise ValueError("Missing required assets: " + ", ".join(missing))
    if not image.is_file():
        raise FileNotFoundError(image)
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("worker.py") if capture else upstream / "fit_single_image.py"
    command = [
        python, str(script), "--input", str(image),
        "--output_npz", str(output / "native_fit.npz"),
        "--model_config", f"configs/model/{model}.yaml",
        "--model_weights", f"weights/realdenseface/{model}.pth",
        "--device", device, "--no_compile",
    ]
    if capture:
        command += ["--capture-dir", str(output), "--precision", precision]
    else:
        command += ["--output_image", str(output / "native_panels.png")]
    run = {"status": "running", "command": command, "model": model, "precision": precision,
           "image_sha256": sha256(image), **metadata}
    manifest = output / "run.json"
    manifest.write_text(json.dumps(run, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        with (output / "upstream.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=upstream, stdout=log, stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        run.update(status="failed", error=str(exc))
        raise
    else:
        run["status"] = "upstream_complete"
    finally:
        run["worker_wall_seconds"] = time.perf_counter() - start
        manifest.write_text(json.dumps(run, indent=2), encoding="utf-8")


def run_fit(python, upstream, image, output, model="vits", device="cuda", precision="fp32-compat"):
    """Fit in the selected native runtime, then bundle its saved geometry."""
    from .artifacts import save_bundle

    output = Path(output).resolve()
    run_upstream(python, Path(upstream).resolve(), Path(image).resolve(), output,
                 model, device, capture=True, precision=precision)
    save_bundle(output, Path(image).resolve())
    return output

