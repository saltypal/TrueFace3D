"""Export a GLB for every fitted mesh and write an index (used by notebooks 11 and 20)."""
import json
from pathlib import Path
import shutil

import numpy as np

from ..benchmark import _active_artifact
from ..rendering.preview import export_glb


def _obj_mesh(path):
    verts, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def export_glbs(tagged_records, out_dir):
    """Neutral-shape GLB per successful attempt; ViT bundles also get their posed UV-mapped face.glb."""
    out_dir = Path(out_dir)
    index = []
    for run, rec in tagged_records:
        if rec.fit_status != "success":
            continue
        folder = Path(rec.case_dir)
        stem = f"{rec.model_id}/{rec.variant_id}/{rec.subject_id}_{Path(rec.image_key).stem}"
        entry = {"model": rec.model_id, "condition": rec.variant_id, "image": rec.image_key,
                 "subject": rec.subject_id, "run": run}
        neutral = _active_artifact(folder, "neutral.obj")
        if neutral:
            vertices, faces = _obj_mesh(neutral)
            target = out_dir / f"{stem}_neutral.glb"
            target.parent.mkdir(parents=True, exist_ok=True)
            export_glb(vertices, faces, target)
            entry["neutral_glb"] = str(target.relative_to(out_dir))
        posed = _active_artifact(folder, "face.glb")
        if posed:
            target = out_dir / f"{stem}_posed.glb"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(posed, target)
            entry["posed_glb"] = str(target.relative_to(out_dir))
        index.append(entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "glb_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
