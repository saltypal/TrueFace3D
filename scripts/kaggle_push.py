"""Push a generated notebook to Kaggle with the CLI wrappers kaggle-a / kaggle-b.

    python scripts/kaggle_push.py notebooks/kaggle/K00_readiness.ipynb --account a --slug tf3d-k00-readiness
    python scripts/kaggle_push.py ... --account b --dataset satyapaladugu/facescape-c --dry-run

Stages the notebook and kernel-metadata.json (pinned image, GPU, private) in kaggle_staging/<slug>/ and runs
`kaggle-<account> kernels push`. Reminder (lesson 16): the CLI cannot cancel runs; cancel old versions on the
website first, and each account allows only 2 GPU sessions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kaggle.notebook_kit import DATASET, WHEELS, kernel_metadata, write_kernel_folder  # noqa: E402

USERS = {"a": "satyapaladugu", "b": "satyapaladuguplus"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--account", choices=USERS, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dataset", action="append", help="extra dataset sources (default: FaceScape-C + wheels)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    # each account reads its own private copy of the FaceScape-C data; B cannot see A's wheels dataset
    own = DATASET.replace("satyapaladugu/", USERS[a.account] + "/")
    datasets = [own] + ([WHEELS, "satyapaladugu/trueface3d-private"] if a.account == "a" else ["satyapaladugu/trueface3d-private"])
    datasets += [d for d in (a.dataset or []) if d not in datasets]
    meta = kernel_metadata(USERS[a.account], a.slug, a.notebook.name, datasets, gpu=not a.no_gpu)
    folder = write_kernel_folder(ROOT / "kaggle_staging" / a.slug, a.notebook, meta)
    print("staged", folder)
    if a.dry_run:
        print((folder / "kernel-metadata.json").read_text())
        return 0
    return subprocess.run([f"kaggle-{a.account}", "kernels", "push", "-p", str(folder)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
