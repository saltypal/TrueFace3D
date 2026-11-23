"""Split a job list over two GPUs with two worker processes (CUDA_VISIBLE_DEVICES).

A job is a JSON-serialisable dict; each worker runs
``python -m src.kaggle.runner --worker module:function --jobs jobs.json --gpu i``.
Resumability comes from the callee (run_matrix skips identical successful rows).
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def split_jobs(jobs, parts):
    """Round-robin so each GPU gets a similar mix of subjects/conditions."""
    return [jobs[i::parts] for i in range(parts)]


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return max(1, sum(line.startswith("GPU ") for line in out.splitlines()))
    except (OSError, subprocess.TimeoutExpired):
        return 1


def launch(worker, jobs, workdir, gpus=None, extra_env=None, cwd=None, gpu_ids=None):
    """Start one process per GPU (or per id in ``gpu_ids``), wait, return exit codes. ``worker`` is 'module:function'."""
    gpu_ids = list(gpu_ids) if gpu_ids is not None else list(range(gpus or gpu_count()))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for index, shard in zip(gpu_ids, split_jobs(list(jobs), len(gpu_ids))):
        file = workdir / f"jobs_gpu{index}.json"
        file.write_text(json.dumps(shard), encoding="utf-8")
        env = {**os.environ, **(extra_env or {}), "CUDA_VISIBLE_DEVICES": str(index),
               "TRUEFACE3D_GPU_INDEX": str(index), "TRUEFACE3D_COMPARATOR_DEVICE": "cuda",
               "PYTHONUNBUFFERED": "1"}
        log = (workdir / f"worker_gpu{index}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "src.kaggle.runner", "--worker", worker,
                                 "--jobs", str(file), "--gpu", str(index)],
                                env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((proc, log))
    codes = []
    for proc, log in procs:
        codes.append(proc.wait())
        log.close()
    return codes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    module, function = args.worker.split(":")
    fn = getattr(importlib.import_module(module), function)
    failures = 0
    for job in json.loads(args.jobs.read_text(encoding="utf-8")):
        try:
            fn(job)
        except Exception:  # one failed job must not stop the shard; callees record their own failures
            failures += 1
            traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
