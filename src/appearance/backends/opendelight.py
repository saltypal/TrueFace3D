"""OpenDelight removes baked lighting from a face photo (delighting).

UNVERIFIED: the entry script and arguments are searched for, not known. The upstream repo is cloned by
src.kaggle.setup and the weights come from the private dataset (checkpoints/opendelight.tar). If the repo, weights or an
entry script are absent the result is not_run with a reason.
"""
from pathlib import Path

from ...kaggle import paths
from .base import PYTHON, guarded, not_run, run_cli, third_party, upstream_ready

REPO = "OpenDelight"
WEIGHT = paths.PROJECT / "checkpoints/opendelight.tar"
ENTRY_CANDIDATES = ("demo.py", "test.py", "inference.py", "run.py")


def available():
    return upstream_ready(REPO, [WEIGHT])


def _entry():
    repo = third_party(REPO)
    return next((repo / n for n in ENTRY_CANDIDATES if (repo / n).is_file()), None)


@guarded
def _run(image_path, out_dir):
    entry = _entry()
    if entry is None:
        return not_run(f"no entry script among {ENTRY_CANDIDATES} in {REPO}")
    run_cli([PYTHON, entry, "--input", image_path, "--output", out_dir], third_party(REPO), out_dir / "upstream.log")
    found = {p.suffix.lstrip("."): str(p) for p in sorted(Path(out_dir).glob("*")) if p.suffix in {".png", ".obj", ".jpg"}}
    result = {"status": "ok", "outputs": found}
    for name in ("uv_texture", "render"):
        if (Path(out_dir) / f"{name}.png").is_file():
            result[name] = str(Path(out_dir) / f"{name}.png")
    return result


def run(image_path, out_dir):
    ok, why = available()
    return _run(image_path, out_dir) if ok else not_run(why)
