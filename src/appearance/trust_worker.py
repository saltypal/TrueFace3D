"""Run TRUST on one image (BalanceAlb models). Executed with cwd = third_party/TRUST.

Usage: python trust_worker.py IMAGE OUT_DIR LEFT TOP RIGHT BOTTOM
The NoW face box (landmark box widened 10% per side, 30% top, 5% bottom) is converted back to the
landmark extent TRUST's own crop expects, then scaled by 1.6 (middle of TRUST's 1.4-1.8 range).
"""
import sys

import cv2
import numpy as np
import torch
from skimage.io import imread
from skimage.transform import estimate_transform, warp

sys.path.insert(0, ".")
from lib import util  # noqa: E402
from lib.model import TRUST  # noqa: E402
import test as trust_test  # noqa: E402  (reuses fixed_dict / config_test)


def main(image_path, out_dir, left, top, right, bottom):
    bw, bh = right - left, bottom - top
    lw, lh = bw / 1.2, bh / 1.35
    cx, cy = (left + right) / 2, top + 0.30 * lh + lh / 2
    size = (lw + lh) / 2 * 1.6
    src = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
    dst = np.array([[0, 0], [0, 223], [223, 0]])
    tform = estimate_transform("similarity", src, dst)
    image = imread(image_path)[:, :, :3] / 255.0
    crop = warp(image, tform.inverse, output_shape=(224, 224))
    h, w = image.shape[:2]
    side = min(h, w)
    scene = image[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
    scene = cv2.resize(scene, (224, 224), interpolation=cv2.INTER_AREA)

    config = trust_test.merge_config(util.dict2obj(trust_test.fixed_dict), trust_test.config_test)
    folder = "./data/TRUST_models_BalanceAlb_version"
    config.n_scenelight = 3
    config.pretrained_modelpath_scene = f"{folder}/E_scene_light_BalanceAlb.tar"
    config.pretrained_modelpath_facel = f"{folder}/E_face_light_BalanceAlb.tar"
    config.pretrained_modelpath_albedo = f"{folder}/E_albedo_BalanceAlb.tar"
    model = TRUST(config)
    for net in (model.E_flame, model.E_scene_light, model.E_face_light, model.E_albedo):
        net.eval()

    def tensor(array):
        return torch.from_numpy(array.transpose(2, 0, 1)).float()[None].cuda()

    with torch.no_grad():
        _, _, texcode, lightcode, albedo, ops, shape_images = model.encoding(tensor(crop), tensor(scene))

    def save(name, value):
        array = (value[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{name}", array[..., ::-1] if array.shape[2] == 3 else array[..., 0])

    save("albedo_uv.png", albedo)
    save("render.png", ops["images"] * ops["alpha_images"])
    save("render_mask.png", ops["alpha_images"])
    save("albedo_render.png", ops["albedo_images"] * ops["alpha_images"])
    save("clay.png", shape_images)
    cv2.imwrite(f"{out_dir}/reference_photo.png", (crop[..., ::-1] * 255).astype(np.uint8))
    np.savez(f"{out_dir}/codes.npz", texcode=texcode.cpu().numpy(), lightcode=lightcode.cpu().numpy())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7]))
