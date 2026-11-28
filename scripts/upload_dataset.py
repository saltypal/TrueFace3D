"""Upload the FaceScape-C folder as a PRIVATE Kaggle dataset (account A), or add a new version.

    python scripts/upload_dataset.py --root /mnt/DataBunker/facescape_c --dry-run
    python scripts/upload_dataset.py --root /mnt/DataBunker/facescape_c                 # first upload
    python scripts/upload_dataset.py --root /mnt/DataBunker/facescape_c --version "add 321-340"

The `subjects/` folder is sent as one zip (Kaggle extracts it on mount; find files with glob, lesson 7).
Account B is added as a collaborator on the website (Dataset -> Settings -> Sharing); the CLI cannot do that.
FaceScape is licensed: the dataset must stay private and is never shared beyond the two accounts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DATASET_ID = "satyapaladugu/facescape-c"

README = """# FaceScape-C (private; FaceScape license applies)

Real FaceScape multi-view photos (subjects 321-359) paired with TU-model ground truth, plus the FaceScape-C
degradations. Built by `python -m src.facescape.build` in github.com/saltypal/TrueFace3D (branch facescape-c);
see build_info.json for the exact commit. Read it with `src.facescape.dataset.FaceScapeC.find()`.

- manifest_clean.csv / manifest_degraded.csv: one row per image (subject, expression, view, split, condition,
  path, sha256, camera, gt, meta)
- subjects/<sid>/<exp>/<view>/clean.png + camera.json (K, and rigid TU->camera R, t in mm)
- subjects/<sid>/<exp>/<view>/degraded/<profile>_s<k>.png
- subjects/<sid>/gt/<exp>.npz (TU vertices, mm) ; topology.npz (faces, uvs, 68 landmarks)
Only subjects 340 and 344 may be shown in publications (FaceScape publishable list).
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--version", help="version notes; omit for the first upload")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not (a.root / "manifest_clean.csv").exists():
        raise SystemExit(f"{a.root} is not a built FaceScape-C folder")
    (a.root / "README.md").write_text(README)
    meta = {"title": "FaceScape-C", "id": DATASET_ID, "licenses": [{"name": "other"}],
            "subtitle": "Private: FaceScape photos + TU ground truth + degradations"}
    (a.root / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in a.root.rglob("*") if p.is_file()) / 1e9
    cmd = ["kaggle-a", "datasets"]
    cmd += (["version", "-m", a.version] if a.version else ["create"]) + ["-p", str(a.root), "-r", "zip"]
    print(f"{size:.2f} GB in {a.root}\n{' '.join(cmd)}")
    if a.dry_run:
        return 0
    return subprocess.run(cmd).returncode   # `datasets create` is private unless --public is passed


if __name__ == "__main__":
    raise SystemExit(main())
