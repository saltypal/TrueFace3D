"""Read-side API for FaceScape-C, used by notebooks and Kaggle jobs.

    ds = FaceScapeC.find()                      # local folder, or /kaggle/input/** on Kaggle
    for case in ds.cases(split="test", condition="C07_s2"):
        img = case.image()                       # uint8 RGB
        verts_cam = case.gt_vertices_camera()    # (N, 3) mm, in this image's camera
        uv = case.project(verts_cam)             # pixel coords in this image
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .geometry import project, transform


@dataclass
class Case:
    root: Path
    row: dict

    def __getattr__(self, key):
        try:
            return self.__dict__["row"][key]
        except KeyError as err:
            raise AttributeError(key) from err

    def image(self) -> np.ndarray:
        import cv2
        bgr = cv2.imread(str(self.root / self.row["path"]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @cached_property
    def camera(self) -> dict:
        cam = json.loads((self.root / self.row["camera"]).read_text())
        k = np.asarray(cam["K"])
        meta = json.loads(self.row["meta"]) if self.row.get("meta") else {}
        if "clean_to_degraded" in meta:          # degraded image may be smaller than the clean crop
            m = np.asarray(meta["clean_to_degraded"], dtype=np.float64)   # pixel-edge convention scale
            sx, sy = m[0, 0], m[1, 1]
            # OpenCV pixel centres sit at integers: u' = (u + 0.5) * s - 0.5
            k = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]]) @ k
        cam["K_image"] = k
        return cam

    def gt_vertices_tu(self) -> np.ndarray:
        return np.load(self.root / self.row["gt"])["vertices"].astype(np.float64)

    def gt_vertices_camera(self) -> np.ndarray:
        cam = self.camera
        return transform(self.gt_vertices_tu(), np.asarray(cam["R"]), np.asarray(cam["t"]))

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        return project(points_cam, self.camera["K_image"])


class FaceScapeC:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        if not (self.root / "manifest_clean.csv").exists():
            raise FileNotFoundError(f"not a FaceScape-C folder: {self.root}")

    @classmethod
    def find(cls) -> "FaceScapeC":
        env = os.environ.get("FACESCAPE_C")
        candidates = [Path(env)] if env else []
        candidates += [Path("/mnt/DataBunker/facescape_c")]
        if Path("/kaggle/input").is_dir():
            candidates += [p.parent for p in Path("/kaggle/input").glob("**/manifest_clean.csv")]
        for c in candidates:
            if (c / "manifest_clean.csv").exists():
                return cls(c)
        raise FileNotFoundError("FaceScape-C folder not found; set FACESCAPE_C")

    def _rows(self, name: str) -> list[dict]:
        p = self.root / name
        if not p.exists():
            return []
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))

    @cached_property
    def clean_rows(self) -> list[dict]:
        return self._rows("manifest_clean.csv")

    @cached_property
    def degraded_rows(self) -> list[dict]:
        return self._rows("manifest_degraded.csv")

    @cached_property
    def subjects(self) -> list[dict]:
        return self._rows("subjects.csv")

    @cached_property
    def topology(self) -> dict:
        return dict(np.load(self.root / "topology.npz"))

    def cases(self, *, split=None, subject=None, expression=None, view=None, condition=None) -> list[Case]:
        rows = self.clean_rows if condition in (None, "clean") else self.degraded_rows
        out = []
        for r in rows:
            if split and r["split"] != split:
                continue
            if subject is not None and int(r["subject"]) != int(subject):
                continue
            if expression is not None and int(r["expression"]) != int(expression):
                continue
            if view and r["view"] != view:
                continue
            if condition not in (None, "clean") and r["condition"] != condition:
                continue
            out.append(Case(self.root, r))
        return out
