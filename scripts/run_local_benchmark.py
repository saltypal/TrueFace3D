"""Run the FaceScape-C mesh benchmark locally on CPU (DECA / EMOCA v2 / MICA in batch mode), then score.

    .venv-linux/bin/python scripts/run_local_benchmark.py --models mica emoca deca --out /mnt/DataBunker/benchmark_local/k10

Same job code, same 1,548 test images and same record format as the Kaggle K10 notebooks; re-running
skips finished fits. Writes quality.csv (inputs) and scores.csv (TU scan error) next to fits/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["mica", "emoca", "deca"])
    ap.add_argument("--out", type=Path, default=Path("/mnt/DataBunker/benchmark_local/k10"))
    ap.add_argument("--expressions", nargs="+", type=int, default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--skip-score", action="store_true")
    a = ap.parse_args(argv)

    import pandas as pd
    from src.facescape import quality
    from src.facescape.dataset import FaceScapeC
    from src.kaggle import facescape_jobs as fj

    os.environ.setdefault("TRUEFACE3D_COMPARATOR_DEVICE", "cpu")
    ds = FaceScapeC.find()
    rows = fj.benchmark_cases(ds, subjects=a.subjects, expressions=a.expressions)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"[local] dataset {ds.root}; {len(rows)} images x {a.models}", flush=True)

    if not (a.out / "quality.csv").exists():
        clean_by = {r["image_id"]: r for r in ds.clean_rows}
        pd.DataFrame([quality.measure(ds.root, r, clean_by[r["image_id"]]) for r in rows]).to_csv(
            a.out / "quality.csv", index=False)
        print("[local] quality.csv written", flush=True)

    t0 = time.time()
    for model in a.models:
        jobs = fj.make_jobs(rows, [model], a.out)
        for i, job in enumerate(jobs, 1):
            fj.run_job(job)
            done = sum(1 for _ in (a.out / "fits" / model).glob("*/*/record.json"))
            print(f"[local] {model} job {i}/{len(jobs)} ({job['variant']}); records {done}; "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    recs = pd.DataFrame(fj.read_records(a.out)).drop(columns=["files"], errors="ignore")
    recs.to_csv(a.out / "fits.csv", index=False)
    print("[local] fit status:", recs.groupby(["model", "fit_status"]).size().to_dict(), flush=True)
    if not a.skip_score:
        from src.facescape.score import score_folder
        table = score_folder(a.out / "fits", a.out / "scores.csv", workers=6)
        ok = table[table.score_status == "success"]
        print(ok.groupby(["model", "protocol"])[["mean_mm", "median_mm"]].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
