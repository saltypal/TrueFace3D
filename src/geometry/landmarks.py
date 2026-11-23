"""FLAME surface landmark transfer for a verified shared topology."""

import numpy as np


NOW_LANDMARK_INDICES = [36, 39, 42, 45, 33, 48, 54]


def transfer_now_landmarks(vertices, faces, source_faces, embedding):
    """Apply MICA's seven ordered indices after exact face-index agreement."""
    if not np.array_equal(faces, source_faces):
        raise ValueError("FLAME source and output face tables differ")
    face_ids = np.asarray(embedding["full_lmk_faces_idx"]).reshape(68)[NOW_LANDMARK_INDICES]
    barycentric = np.asarray(embedding["full_lmk_bary_coords"]).reshape(68, 3)[NOW_LANDMARK_INDICES]
    if (face_ids.dtype.kind not in "iu" or np.any(face_ids < 0) or np.any(face_ids >= len(faces))
            or not np.isfinite(barycentric).all() or np.any(barycentric < -1e-6)
            or not np.allclose(barycentric.sum(axis=1), 1)):
        raise ValueError("Invalid FLAME landmark face indices or barycentric coordinates")
    return np.einsum("kji,kj->ki", vertices[faces[face_ids]], barycentric)
