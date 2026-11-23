"""Visualize sampled image correspondences and dense UV depth."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sample_uv(values, uv):
    """Bilinear sampling with PyTorch's align_corners=False and zero padding."""
    values, uv = np.asarray(values), np.asarray(uv, dtype=float)
    height, width = values.shape[:2]
    x, y = (uv * [width, height] - .5).T
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - left, y - top
    result = np.zeros((len(uv), *values.shape[2:]), dtype=np.result_type(values.dtype, float))
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        i, j = top + dy, left + dx
        valid = (i >= 0) & (i < height) & (j >= 0) & (j < width)
        w = weight[valid].reshape((-1,) + (1,) * (values.ndim - 2))
        result[valid] += values[i[valid], j[valid]] * w
    return result


def draw(input_path, priors_path):
    with np.load(priors_path, allow_pickle=False) as arrays:
        points = arrays["vertex_coord"]
        depth = arrays["uv_relative_depth"]
    with Image.open(input_path) as source:
        photo = source.convert("RGB")
    photo.thumbnail((700, 700))
    with Image.open(input_path) as source:
        scale = np.array(photo.size) / source.size
    sketch = ImageDraw.Draw(photo)
    for x, y in points * scale:
        if 0 <= x < photo.width and 0 <= y < photo.height:
            sketch.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(60, 255, 195))
    photo.save(priors_path.with_name("correspondence.png"))
    lo, hi = np.percentile(depth, [2, 98])
    intensity = np.round(np.clip((depth - lo) / max(float(hi - lo), 1e-8), 0, 1) * 255)
    Image.fromarray(intensity.astype(np.uint8)).save(priors_path.with_name("uv_depth.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("priors", type=Path)
    args = parser.parse_args()
    draw(args.image, args.priors)
