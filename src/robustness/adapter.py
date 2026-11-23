"""Directional diagnostics for aligned fitting targets."""

import numpy as np


def _aligned_targets(observed, comparison, valid):
    observed = np.asarray(observed, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (observed.ndim != 2 or observed.shape[1] not in (1, 2)
            or comparison.shape != observed.shape or valid.shape != observed.shape[:1]):
        raise ValueError("Expected matching (vertices, 1 or 2) targets and a vertex mask")
    usable = valid & np.isfinite(observed).all(axis=1) & np.isfinite(comparison).all(axis=1)
    return observed, comparison, usable


def directional_agreement(observed, probe, teacher, valid):
    """Compare the proposed correction direction with an aligned teacher residual."""
    observed, probe, usable = _aligned_targets(observed, probe, valid)
    _, teacher, usable = _aligned_targets(observed, teacher, usable)
    direction = (observed[usable] - probe[usable]).ravel()
    residual = (teacher[usable] - observed[usable]).ravel()
    norm = np.linalg.norm(direction) * np.linalg.norm(residual)
    return {
        "valid_vertices": int(usable.sum()),
        "cosine": float(np.clip(np.dot(direction, residual) / norm, -1, 1)) if norm else None,
    }
