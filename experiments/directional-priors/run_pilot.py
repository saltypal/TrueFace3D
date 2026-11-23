"""Run a fixed-center-crop diagnostic on one official portrait."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.robustness.adapter import directional_agreement
from src.robustness.degrade import Degradation, bicubic_control, make_variant


def load_targets(path, vertex_uv):
    with np.load(path, allow_pickle=False) as data:
        xy = data["vertex_coord"]
        depth_map = torch.from_numpy(data["uv_relative_depth"])[None, None]
        depth = F.grid_sample(depth_map, vertex_uv, align_corners=False).reshape(-1, 1).numpy()
        height, width = data["input_size_hw"]
        valid = ((xy >= 0) & (xy < [width, height])).all(axis=1)
        return xy, depth, valid, data["crop_bbox"], (width, height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with Image.open(args.input) as source:
        clean = source.convert("RGB")
    observed, _ = make_variant(clean, Degradation(divisor=8), seed=0)
    probe_size = tuple(max(1, round(size * 0.75)) for size in observed.size)
    probe = observed.resize(probe_size, Image.Resampling.BOX)
    images = (clean, bicubic_control(observed, clean.size), bicubic_control(probe, clean.size))

    with np.load(args.upstream / "weights/flame/flame_assets.npz", allow_pickle=False) as support:
        vertex_uv = torch.from_numpy(support["vertex_uvs"])[None, :, None].float() * 2 - 1

    samples = []
    for name, image in zip(("clean", "observed", "probe"), images):
        image_path = args.output / f"{name}.png"
        priors_path = args.output / f"{name}.npz"
        image.save(image_path)
        subprocess.run(
            [sys.executable, "-m", "src.geometry.cpu_priors", "--upstream", str(args.upstream),
             "--input", str(image_path), "--output", str(priors_path)],
            check=True,
        )
        samples.append(load_targets(priors_path, vertex_uv))

    teacher, lower, coarser = samples
    if any(sample[4] != teacher[4] or not np.array_equal(sample[3], teacher[3])
           for sample in samples):
        raise ValueError("Center crop or source canvas differed across observations")
    valid = teacher[2] & lower[2] & coarser[2]
    with args.input.open("rb") as source:
        source_hash = file_digest(source, "sha256").hexdigest()
    report = {
        "scope": "one CPU prior diagnostic; no scan, solver or visibility mask",
        "source_sha256": source_hash,
        "source_dimensions_wh": clean.size,
        "observed_dimensions_wh": observed.size,
        "probe_dimensions_wh": probe_size,
        "correspondence": directional_agreement(lower[0], coarser[0], teacher[0], valid),
        "relative_depth": directional_agreement(lower[1], coarser[1], teacher[1], valid),
    }
    serialized = json.dumps(report, indent=2)
    (args.output / "metrics.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
