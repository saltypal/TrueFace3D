"""Build kaggle_staging/trueface3d-private (local, Windows). Never uploads.

Dry run (default) prints a manifest and total size; ``--build`` copies the files.
"""
import argparse
import json
from pathlib import Path
import shutil

from ..data_now import installed_cases, pilot_cases, real_world_cases
from . import assets, paths

DATASET_SLUG = "trueface3d-private"


def selected_cases(data_root, count=20):
    cases = installed_cases(data_root)
    clean = pilot_cases(cases, count=count)
    real = real_world_cases(cases)
    return clean, real


def manifest(project=paths.PROJECT, data_root=None, include_optional=False, count=20):
    """Return list of (source Path, dataset-relative posix path, kind)."""
    data_root = Path(data_root or project / "Data")
    clean, real = selected_cases(data_root, count)
    rows, seen = [], set()

    def add(src, rel, kind):
        if rel not in seen:
            seen.add(rel)
            rows.append((Path(src), rel, kind))

    for case in [*clean, *real]:
        for src, kind in ((case.image_path, "photo"), (case.box_path, "face box"), (case.scan_path, "scan"),
                          (case.scan_landmark_path, "scan landmarks")):
            if src is not None:
                add(src, "Data/" + Path(src).resolve().relative_to(data_root.resolve()).as_posix(), kind)
    for rel in assets.REQUIRED_WEIGHTS + (assets.OPTIONAL_WEIGHTS if include_optional else []):
        add(project / rel, f"weights/{rel}", "weights")
    return rows, clean, real


def write_metadata(out, username, clean, real):
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({c.image_key for c in clean} | {c.image_key for c in real})
    data = out / "Data"
    data.mkdir(exist_ok=True)
    (data / "imagepathsvalidation.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "TrueFace3D private", "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}], "isPrivate": True}, indent=2), encoding="utf-8")
    (out / paths.MARKER).write_text(json.dumps({
        "clean_images": [c.image_key for c in clean], "real_world_images": [c.image_key for c in real]},
        indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=paths.PROJECT / "kaggle_staging" / DATASET_SLUG)
    parser.add_argument("--username", default="<KAGGLE_USER_A>")
    parser.add_argument("--build", action="store_true", help="copy files (default is a dry run)")
    parser.add_argument("--with-optional", action="store_true", help="also stage OpenDelight/TRUST extras")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    rows, clean, real = manifest(data_root=args.data_root, include_optional=args.with_optional)
    total = 0
    for src, rel, kind in rows:
        size = src.stat().st_size if src.is_file() else 0
        total += size
        flag = "" if src.is_file() else "  MISSING"
        print(f"{size / 2**20:10.2f} MiB  {kind:15s} {rel}{flag}")
    print(f"\n{len(rows)} files, {total / 2**30:.2f} GiB; {len(clean)} clean-set images, "
          f"{len(real)} real-world images, {len({c.subject_id for c in clean})} subjects")
    for name, needed in assets.MISSING_FOR_APPEARANCE.items():
        print(f"NOT AVAILABLE locally ({name}): {', '.join(needed)}")
    if not args.build:
        print("Dry run: nothing copied. Re-run with --build to create", args.output)
        return rows
    for src, rel, _ in rows:
        dst = args.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
    write_metadata(args.output, args.username, clean, real)
    print("Staged at", args.output, "(not uploaded)")
    return rows


if __name__ == "__main__":
    main()
