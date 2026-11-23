"""Shared pieces for appearance backends.

Interface for every backend module:
    available() -> (bool, reason)
    run(image_path, out_dir) -> dict  with status in {"ok", "not_run", "failed"}
Backends never raise; a missing repo/weight gives status "not_run" with a reason.
"""
from pathlib import Path
import subprocess
import sys
import time

from ...kaggle import paths


def third_party(name):
    return paths.third_party_root() / name


def not_run(reason):
    return {"status": "not_run", "reason": reason}


def guarded(run):
    """Decorator: turn exceptions into a failed result that keeps the stage and message."""
    def wrapper(image_path, out_dir, **kwargs):
        start = time.perf_counter()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = run(Path(image_path), Path(out_dir), **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            result = {"status": "failed", "stage": getattr(exc, "stage", type(exc).__name__), "reason": str(exc)[:500]}
        result.setdefault("seconds", time.perf_counter() - start)
        return result
    return wrapper


def run_cli(command, cwd, log_path, timeout=1800):
    """Run an upstream command, keep a log, raise with the log tail on failure."""
    with open(log_path, "wb") as log:
        done = subprocess.run([str(c) for c in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False)
    if done.returncode:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        error = RuntimeError(" | ".join(tail) or f"exit {done.returncode}")
        error.stage = "upstream_cli"
        raise error
    return done


def upstream_ready(repo_name, weight_hints=()):
    repo = third_party(repo_name)
    if not repo.is_dir():
        return False, f"{repo_name} not cloned in {repo.parent}"
    missing = [str(w) for w in weight_hints if not Path(w).exists()]
    if missing:
        return False, "missing weights: " + ", ".join(missing)
    return True, "ok"


PYTHON = sys.executable
