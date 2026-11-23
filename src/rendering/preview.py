"""Export a self-contained neutral mesh as a standards-compatible GLB."""

import json
import struct

import numpy as np


def export_glb(vertices, faces, path, uvs=None, uv_faces=None):
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite Vx3 vertices")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not faces.size
            or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Expected valid Fx3 triangle indices")
    normals = np.zeros_like(vertices)
    surface = vertices[faces]
    cross = np.cross(surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    length = np.linalg.norm(normals, axis=1)
    normals[length > 0] /= length[length > 0, None]
    normals[length == 0] = [0, 0, 1]
    arrays = []
    if uvs is not None:
        uvs, uv_faces = np.asarray(uvs, dtype="<f4"), np.asarray(uv_faces)
        if (uvs.ndim != 2 or uvs.shape[1] != 2 or not np.isfinite(uvs).all()
                or uv_faces.shape != faces.shape or uv_faces.dtype.kind not in "iu"
                or uv_faces.min() < 0 or uv_faces.max() >= len(uvs)):
            raise ValueError("Invalid face-corner UV indexing")
        corners = np.column_stack((faces.ravel(), uv_faces.ravel()))
        pairs, indices = np.unique(corners, axis=0, return_inverse=True)
        vertices, faces = vertices[pairs[:, 0]], indices.reshape(-1, 3)
        normals = normals[pairs[:, 0]]
        arrays.append(("TEXCOORD_0", uvs[pairs[:, 1]], "VEC2", 5126, 34962))
    arrays.insert(0, ("POSITION", vertices, "VEC3", 5126, 34962))
    arrays.append(("NORMAL", normals, "VEC3", 5126, 34962))
    arrays.append(("indices", np.asarray(faces, dtype="<u4").ravel(), "SCALAR", 5125, 34963))
    primitive = {"attributes": {}, "material": 0}
    views, accessors = [], []
    binary = bytearray()
    for name, values, kind, dtype, target in arrays:
        index = len(views)
        views.append(dict(buffer=0, byteOffset=len(binary), byteLength=values.nbytes, target=target))
        accessor = dict(bufferView=index, componentType=dtype, count=len(values), type=kind)
        if name == "POSITION":
            accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        accessors.append(accessor)
        binary.extend(values.tobytes())
        if name == "indices":
            primitive["indices"] = index
        else:
            primitive["attributes"][name] = index
    model = dict(asset={"version": "2.0"}, scene=0, scenes=[{"nodes": [0]}],
                 nodes=[{"mesh": 0}], meshes=[{"primitives": [primitive]}],
                 materials=[{"name": "Neutral clay", "pbrMetallicRoughness": {
                     "baseColorFactor": [0.65, 0.72, 0.78, 1.0], "metallicFactor": 0.0,
                     "roughnessFactor": 0.9}}],
                 buffers=[{"byteLength": len(binary)}], bufferViews=views, accessors=accessors)
    header = json.dumps(model, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(header) + len(binary))
                     + struct.pack("<I4s", len(header), b"JSON") + header
                     + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    return path

