"""Optional oracle diagnostic: swap clean-image priors into a degraded fit.

Off by default. ``--priors-in clean/priors.npz`` (or TRUEFACE3D_PRIORS_IN) tells the
ViT worker to replace the inferred per-vertex priors before the FLAME solve.

Parity design note: with ``mode="none"`` the worker must reproduce the unmodified fit
exactly (same face.npz); with the clean image's own priors and ``mode="both"`` a fit of the
clean image must also match its unmodified fit. Both properties are unit tested on the pure
substitution function; the end-to-end check needs the GPU and is untested locally.
"""
import os

import numpy as np

MODES = {
    "none": (),
    "coord": ("vertex_coord", "vertex_coord_log_var"),
    "depth": ("vertex_depth", "vertex_depth_log_var"),
    "both": ("vertex_coord", "vertex_coord_log_var", "vertex_depth", "vertex_depth_log_var"),
}


def oracle_settings(argv):
    """Pop --priors-in/--priors-mode from argv (mutating it); fall back to environment."""
    values = {"priors_in": os.environ.get("TRUEFACE3D_PRIORS_IN"),
              "mode": os.environ.get("TRUEFACE3D_PRIORS_MODE", "both")}
    for flag, key in (("--priors-in", "priors_in"), ("--priors-mode", "mode")):
        if flag in argv:
            i = argv.index(flag)
            values[key] = argv[i + 1]
            del argv[i:i + 2]
    if values["mode"] not in MODES:
        raise ValueError(f"--priors-mode must be one of {sorted(MODES)}")
    return values["priors_in"], values["mode"]


def map_coords(coord, src_bbox, dst_bbox):
    """Move per-vertex image coordinates from one face box to another (both x0, y0, x1, y1 in their own image).

    ``vertex_coord`` is stored in input-image pixels, so priors from the original photo must be mapped into the
    degraded photo's frame before substitution. Log-variances are resolution-independent and are not changed.
    """
    s, d = np.asarray(src_bbox, float), np.asarray(dst_bbox, float)
    scale = (d[2:] - d[:2]) / np.maximum(s[2:] - s[:2], 1e-6)
    return (np.asarray(coord, float) - s[:2]) * scale + d[:2]


def substitute_priors(result, clean, mode):
    """Replace the selected prior fields on ``result`` with arrays from ``clean``.

    Coordinates are mapped from ``clean['face_bbox']`` to the result's face box when both are known.
    Keeps each field's container type and dtype (numpy or torch). Returns the names replaced.
    """
    names = MODES[mode]
    dst_box = getattr(result, "face_bbox", None)
    for name in names:
        old, new = getattr(result, name), np.asarray(clean[name])
        if name == "vertex_coord" and "face_bbox" in clean and dst_box is not None:
            box = dst_box.detach().cpu().numpy() if hasattr(dst_box, "detach") else np.asarray(dst_box)
            new = map_coords(new.reshape(-1, 2), np.asarray(clean["face_bbox"]).reshape(-1)[:4],
                             box.reshape(-1)[:4]).reshape(new.shape)
        if tuple(old.shape) != new.shape:
            raise ValueError(f"Clean prior {name} has shape {new.shape}, expected {tuple(old.shape)}")
        if hasattr(old, "detach"):
            import torch
            replacement = torch.as_tensor(new, dtype=old.dtype, device=old.device)
        else:
            replacement = new.astype(old.dtype, copy=True)
        setattr(result, name, replacement)
    return list(names)
