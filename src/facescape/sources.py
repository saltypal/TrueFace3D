"""Read FaceScape files straight from the downloaded zips (no full extraction).

Expected archives (any folder; default ~/Downloads, override with FACESCAPE_ZIPS):
  fsmview_trainset_images_<a>-<b>.zip   multi-view photos + params.json
  fsmview_trainset_shape_<a>-<b>.zip    multi-view meshes (optional)
  facescape_trainset_<a>_<b>.zip        TU models (mesh, texture, displacement)
and the toolkit clone (FACESCAPE_TOOLKIT, default ~/Downloads/facescape_repo) for
predef/Rt_scale_dict.json, landmark indices and the facial mask.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import expression_dir

_RANGE = re.compile(r"(\d+)[-_](\d+)\.zip$")


def _zip_ranges(folder: Path, prefix: str) -> list[tuple[int, int, Path]]:
    out = []
    for p in sorted(folder.glob(prefix + "*.zip")):
        m = _RANGE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return out


@dataclass
class Sources:
    zips: Path = field(default_factory=lambda: Path(os.environ.get("FACESCAPE_ZIPS", Path.home() / "Downloads")))
    toolkit: Path = field(default_factory=lambda: Path(os.environ.get(
        "FACESCAPE_TOOLKIT", Path.home() / "Downloads" / "facescape_repo")))

    def __post_init__(self):
        self.zips, self.toolkit = Path(self.zips), Path(self.toolkit)
        self.image_zips = _zip_ranges(self.zips, "fsmview_trainset_images_")
        self.shape_zips = _zip_ranges(self.zips, "fsmview_trainset_shape_")
        self.tu_zips = _zip_ranges(self.zips, "facescape_trainset_")
        self._open: dict[Path, zipfile.ZipFile] = {}
        self._rt_scale = None

    # ---- archive handling -------------------------------------------------
    def _zip(self, path: Path) -> zipfile.ZipFile:
        if path not in self._open:
            self._open[path] = zipfile.ZipFile(path)
        return self._open[path]

    @staticmethod
    def _find(ranges, subject: int):
        for a, b, p in ranges:
            if a <= subject <= b:
                return p
        return None

    def photo_subjects(self) -> list[int]:
        return sorted(s for a, b, _ in self.image_zips for s in range(a, b + 1))

    def has_photos(self, subject: int) -> bool:
        return self._find(self.image_zips, subject) is not None

    def has_tu(self, subject: int) -> bool:
        return self._find(self.tu_zips, subject) is not None

    def _read(self, ranges, subject: int, member: str) -> bytes:
        path = self._find(ranges, subject)
        if path is None:
            raise FileNotFoundError(f"no archive covers subject {subject}")
        return self._zip(path).read(member)

    # ---- multi-view photos ------------------------------------------------
    def params(self, subject: int, expression: int) -> dict:
        member = f"fsmview_trainset/{subject}/{expression_dir(expression)}/params.json"
        return json.loads(self._read(self.image_zips, subject, member))

    def photo_bytes(self, subject: int, expression: int, view: int) -> bytes:
        member = f"fsmview_trainset/{subject}/{expression_dir(expression)}/{view}.jpg"
        return self._read(self.image_zips, subject, member)

    def mview_mesh_bytes(self, subject: int, expression: int) -> bytes | None:
        if self._find(self.shape_zips, subject) is None:
            return None
        return self._read(self.shape_zips, subject, f"fsmview_trainset/{subject}/{expression_dir(expression)}.ply")

    # ---- TU models ----------------------------------------------------------
    def tu_obj_bytes(self, subject: int, expression: int) -> bytes:
        return self._read(self.tu_zips, subject, f"{subject}/models_reg/{expression_dir(expression)}.obj")

    def tu_texture_bytes(self, subject: int, expression: int) -> bytes:
        return self._read(self.tu_zips, subject, f"{subject}/models_reg/{expression_dir(expression)}.jpg")

    def tu_dpmap_bytes(self, subject: int, expression: int) -> bytes:
        return self._read(self.tu_zips, subject, f"{subject}/dpmap/{expression_dir(expression)}.png")

    # ---- toolkit ------------------------------------------------------------
    def rt_scale(self, subject: int, expression: int) -> tuple[float, np.ndarray]:
        """Scale s and 4x4/3x4 Rt that map the multi-view mesh into the TU frame (mm)."""
        if self._rt_scale is None:
            self._rt_scale = json.loads((self.toolkit / "toolkit/predef/Rt_scale_dict.json").read_text())
        s, rt = self._rt_scale[str(subject)][str(expression)]
        return float(s), np.asarray(rt, dtype=np.float64)

    def landmark_indices(self) -> np.ndarray:
        return np.load(self.toolkit / "toolkit/predef/landmark_indices.npz")["v10"]

    def facial_mask_path(self) -> Path:
        return self.toolkit / "toolkit/predef/facial_mask_v10.png"

    def info_list_path(self) -> Path:
        hits = sorted(self.zips.glob("**/info_list_v2.txt"))
        if not hits:
            raise FileNotFoundError("info_list_v2.txt not found under " + str(self.zips))
        return hits[0]

    def close(self):
        for z in self._open.values():
            z.close()
        self._open.clear()
