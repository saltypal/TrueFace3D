"""CPU-only ViT-S diagnostic using the pinned upstream network and crop code."""

import argparse
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream.resolve()
    sys.path.insert(0, str(root))
    from common.config import load_model_config
    from models.realdenseface import RealDenseFace
    spec = importlib.util.spec_from_file_location("rdf_crop", root / "tracker/inference/preprocess.py")
    crop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crop)

    image = cv2.imread(str(args.input))
    if image is None:
        raise ValueError(f"Cannot open {args.input}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    config = load_model_config(root / "configs/model/vits.yaml")
    config["dino_pretrained"] = False
    print("Loading ViT-S on CPU", flush=True)
    model = RealDenseFace(**config).eval()
    model.load_state_dict(torch.load(root / "weights/realdenseface/vits.pth",
                                     map_location="cpu", weights_only=True))
    model.build_inference_cache()
    bbox = crop.make_bbox_square(crop.create_default_face_bbox(image))
    rgb, bbox = crop.crop_and_resize_image(image, bbox, int(config["target_size"]))
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255
    x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    print("Predicting the four UV maps", flush=True)
    coord, log_c, depth, log_d = model(x)
    coord = coord.float() / config.get("scale_coord", 1.) * .5 + .5
    depth = depth.float() / config.get("scale_depth", 1.)
    log_c, log_d = log_c.float().clamp(-8, 3), log_d.float().clamp(-8, 3)
    with np.load(root / "weights/flame/flame_assets.npz", allow_pickle=False) as support:
        grid = torch.from_numpy(support["vertex_uvs"])[None, :, None].float() * 2 - 1
    sampled = F.grid_sample(coord, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T
    sampled = sampled.numpy() * (bbox[2:] - bbox[:2]) + bbox[:2]
    arrays = {"uv_correspondence": coord[0].permute(1, 2, 0).numpy(),
              "uv_relative_depth": depth[0, 0].numpy(),
              "uv_coord_log_variance": log_c[0, 0].numpy(),
              "uv_depth_log_variance": log_d[0, 0].numpy(),
              "vertex_coord": sampled, "crop_bbox": bbox,
              "input_size_hw": np.asarray(image.shape[:2]),
              "crop_policy": np.asarray("center crop; CPU prior diagnostic")}
    if not all(np.isfinite(x).all() for x in arrays.values() if x.dtype.kind == "f"):
        raise ValueError("Nonfinite prior values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        np.savez_compressed(file, **arrays)
    print("Saved", args.output, {key: value.shape for key, value in arrays.items()}, flush=True)


if __name__ == "__main__":
    main()
