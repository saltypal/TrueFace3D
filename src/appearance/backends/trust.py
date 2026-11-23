"""TRUST (BalanceAlb): skin-tone-aware albedo with scene + face lighting. UNVERIFIED on Kaggle."""
from pathlib import Path

from .base import PYTHON, guarded, not_run, run_cli, third_party, upstream_ready

REPO = "TRUST"
WORKER = Path(__file__).resolve().parents[1] / "trust_worker.py"


def _weights():
    data = third_party(REPO) / "data"
    return [data / "TRUST_models_BalanceAlb_version/E_albedo_BalanceAlb.tar", data / "BalanceAlb_model.npz",
            data / "generic_model.pkl", data / "deca_model.tar"]


def available():
    return upstream_ready(REPO, _weights())


@guarded
def _run(image_path, out_dir, box=None, **_):
    if box is None:
        return not_run("TRUST needs the face box from the input metadata")
    run_cli([PYTHON, WORKER, Path(image_path).resolve(), Path(out_dir).resolve(), *map(float, box)],
            third_party(REPO), Path(out_dir) / "upstream.log")
    names = ("albedo_uv", "render", "render_mask", "albedo_render", "clay", "reference_photo")
    return {"status": "ok", **{n: str(Path(out_dir) / f"{n}.png") for n in names}}


def run(image_path, out_dir, **kwargs):
    ok, why = available()
    return _run(image_path, out_dir, **kwargs) if ok else not_run(why)
