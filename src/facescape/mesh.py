"""Minimal OBJ reader for FaceScape TU meshes.

trimesh splits vertices along UV seams, which breaks the shared TU topology, so we parse
positions, UVs and the two index sets ourselves. Polygons are fan-triangulated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjMesh:
    vertices: np.ndarray    # (N, 3) float64, mm
    faces: np.ndarray       # (F, 3) int32, into vertices
    uvs: np.ndarray         # (M, 2) float32
    face_uvs: np.ndarray    # (F, 3) int32, into uvs


def read_obj(data: bytes | str) -> ObjMesh:
    text = data.decode() if isinstance(data, bytes) else data
    v, vt, f, ft = [], [], [], []
    for line in text.splitlines():
        if line.startswith("v "):
            v.append(line[2:].split()[:3])
        elif line.startswith("vt "):
            vt.append(line[3:].split()[:2])
        elif line.startswith("f "):
            corners = [c.split("/") for c in line[2:].split()]
            vi = [int(c[0]) - 1 for c in corners]
            ti = [int(c[1]) - 1 if len(c) > 1 and c[1] else -1 for c in corners]
            for k in range(1, len(corners) - 1):
                f.append((vi[0], vi[k], vi[k + 1]))
                ft.append((ti[0], ti[k], ti[k + 1]))
    return ObjMesh(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32),
                   np.asarray(vt, dtype=np.float32).reshape(-1, 2), np.asarray(ft, dtype=np.int32))
