"""Collect and summarise FaceScape-C benchmark scores (local CPU runs and downloaded Kaggle zips)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL = Path("/mnt/DataBunker/benchmark_local")
KAGGLE = Path("/mnt/DataBunker/kaggle_outputs")
MODEL_NAMES = {"vits": "RealDenseFace-S", "vitb": "RealDenseFace-B", "deca": "DECA", "emoca": "EMOCA v2",
               "mica": "MICA", "smirk": "SMIRK", "hrn": "HRN"}


def _read_member(zip_path: Path, suffix: str) -> list[pd.DataFrame]:
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(suffix) and name.count("/") <= 1:
                out.append(pd.read_csv(io.BytesIO(zf.read(name))))
    return out


def collect(name: str = "scores.csv", local=LOCAL, kaggle=KAGGLE) -> pd.DataFrame:
    """Every <name> found: local run folders, Kaggle output folders and inside Kaggle result zips."""
    frames = []
    for p in sorted(Path(local).glob(f"*/{name}")):
        frames.append(pd.read_csv(p).assign(source=f"local:{p.parent.name}"))
    for p in sorted(Path(kaggle).glob(f"*/*/{name}")):
        frames.append(pd.read_csv(p).assign(source=f"kaggle:{p.parent.parent.name}"))
    for z in sorted(Path(kaggle).glob("*/*.zip")):
        for f in _read_member(z, "/" + name):
            frames.append(f.assign(source=f"kaggle:{z.parent.name}"))
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return add_keys(table)


def add_keys(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    if "image_id" in t:
        parts = t.image_id.str.split("_", expand=True)
        t["subject"], t["expression"], t["view"] = parts[0].astype(int), parts[1].astype(int), parts[2]
    if "condition" in t:
        t["profile"] = t.condition.str.split("_").str[0]
        t["severity"] = t.condition.str.extract(r"_s(\d)", expand=False).fillna(0).astype(int)
    if "model" in t:
        t["model_name"] = t.model.map(MODEL_NAMES).fillna(t.model)
    return t


def prefer_local(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, image, condition, protocol): local CPU result first, then Kaggle."""
    if t.empty:
        return t
    t = t.assign(_rank=t.source.str.startswith("local").map({True: 0, False: 1}))
    keys = ["model", "image_id", "condition"] + (["protocol"] if "protocol" in t else [])
    return t.sort_values("_rank").drop_duplicates(keys).drop(columns="_rank")


def paired_degradation(ok: pd.DataFrame, metric="mean_mm") -> pd.DataFrame:
    """Error of each degraded image minus the clean error of the same image (same model, protocol)."""
    clean = ok[ok.condition == "clean"].set_index(["model", "protocol", "image_id"])[metric]
    deg = ok[ok.condition != "clean"].copy()
    deg["clean_" + metric] = [clean.get((m, p, i), np.nan) for m, p, i in zip(deg.model, deg.protocol, deg.image_id)]
    deg["delta_" + metric] = deg[metric] - deg["clean_" + metric]
    deg["ratio_" + metric] = deg[metric] / deg["clean_" + metric]
    return deg


def mce(deg: pd.DataFrame, metric="mean_mm", reference="deca") -> pd.Series:
    """Corruption error per model (mean over conditions), normalised by the reference model (ImageNet-C style)."""
    per = deg.groupby(["model", "profile"])[metric].mean().unstack()
    if reference not in per.index:
        return per.mean(axis=1).rename("mean_corruption_error_mm")
    return (per.div(per.loc[reference])).mean(axis=1).rename(f"mCE_vs_{reference}")
