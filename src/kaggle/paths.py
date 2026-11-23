"""One path contract for local Windows and Kaggle Linux runs.

Env vars override everything:
  TRUEFACE3D_DATA         NoW data root (the folder that holds NoWDataset/, scans/, ...).
  TRUEFACE3D_WEIGHTS      weights root mirroring project-relative paths (third_party/..., checkpoints/...).
  TRUEFACE3D_THIRD_PARTY  where upstream checkouts live.
"""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
DATASET_NAME = "trueface3d-private"
MARKER = "TRUEFACE3D_DATASET.json"


def on_kaggle():
    return os.name != "nt" and (KAGGLE_INPUT.is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ)


def dataset_root():
    """Find the private dataset mount (Kaggle mounts differ between /kaggle/input/<name> and nested)."""
    if os.environ.get("TRUEFACE3D_DATASET"):
        return Path(os.environ["TRUEFACE3D_DATASET"])
    default = KAGGLE_INPUT / DATASET_NAME
    if (default / MARKER).is_file() or not KAGGLE_INPUT.is_dir():
        return default
    for depth in ("*", "*/*", "*/*/*"):
        for hit in sorted(KAGGLE_INPUT.glob(f"{depth}/{MARKER}")):
            return hit.parent
    return default


def data_root():
    if os.environ.get("TRUEFACE3D_DATA"):
        return Path(os.environ["TRUEFACE3D_DATA"])
    return dataset_root() / "Data" if on_kaggle() else PROJECT / "Data"


def weights_root():
    """Root whose layout mirrors project-relative paths; locally that is the project itself."""
    if os.environ.get("TRUEFACE3D_WEIGHTS"):
        return Path(os.environ["TRUEFACE3D_WEIGHTS"])
    return dataset_root() / "weights" if on_kaggle() else PROJECT


def third_party_root():
    if os.environ.get("TRUEFACE3D_THIRD_PARTY"):
        return Path(os.environ["TRUEFACE3D_THIRD_PARTY"])
    return Path("/kaggle/working/third_party_src") if on_kaggle() else PROJECT / "third_party"


def describe():
    return {"project": str(PROJECT), "on_kaggle": on_kaggle(), "dataset": str(dataset_root()),
            "data": str(data_root()), "weights": str(weights_root()), "third_party": str(third_party_root())}
