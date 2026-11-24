"""Local OpenDelight appearance benchmark: 20 NoW people x 5 conditions (DECA/TRUST run on Kaggle A).

Usage: .venv-vit/Scripts/python experiments/appearance/run_local_opendelight.py
Stops after the first job if it fails, so a broken setup costs one job, not 100.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.kaggle import appearance_jobs  # noqa: E402

OUT = Path("experiments/appearance/local-opendelight")
OUT.mkdir(parents=True, exist_ok=True)
jobs = appearance_jobs.jobs(OUT, methods=("opendelight",))
start = time.time()
counts = {}
for index, job in enumerate(jobs, 1):
    folder = appearance_jobs.attempt_dir(job)
    done = folder / "result.json"
    if done.is_file() and json.loads(done.read_text()).get("status") == "ok":
        counts["ok"] = counts.get("ok", 0) + 1
        continue
    t = time.time()
    appearance_jobs.run_job(job)
    result = json.loads(done.read_text()) if done.is_file() else {"status": "no result.json"}
    status = result.get("status")
    counts[status] = counts.get(status, 0) + 1
    print(f"[{(time.time() - start) / 60:6.1f} min] {index}/{len(jobs)} {job['variant']:9} {job['key'][:40]} "
          f"{status} {time.time() - t:.0f}s {str(result.get('reason', ''))[-160:]}", flush=True)
    if index == 1 and status != "ok":
        sys.exit("first job failed; stopping")
print("counts", counts, flush=True)
pairs = appearance_jobs.score_pairs(OUT, methods=("opendelight",))
print(len(pairs), "pair scores", flush=True)
