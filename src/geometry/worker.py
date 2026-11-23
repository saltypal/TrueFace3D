"""Capture native data in an isolated process without editing upstream source."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np


def prior_autocast(original, *, precision, **options):
    if (precision == "fp32-compat" and options.get("device_type") == "cuda"
            and options.get("enabled") and str(options.get("dtype")) == "torch.bfloat16"):
        return nullcontext()
    return original(**options)


def array(value):
    if not hasattr(value, "detach"):
        return np.asarray(value)
    value = value.detach()
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return value.cpu().numpy()


def main():
    upstream = Path.cwd()
    sys.path.insert(0, str(upstream))
    i = sys.argv.index("--capture-dir")
    output = Path(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    i = sys.argv.index("--precision")
    precision = sys.argv[i + 1]
    del sys.argv[i:i + 2]

    import torch
    import fit_single_image as native
    import tracker.inference.realdenseface_inferencer as inference_module
    from common.types import FittingOutput

    # Optional oracle diagnostic (off unless --priors-in or TRUEFACE3D_PRIORS_IN is given).
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.geometry.oracle import oracle_settings, substitute_priors
    priors_in, priors_mode = oracle_settings(sys.argv)
    clean_priors = dict(np.load(priors_in, allow_pickle=False)) if priors_in else None

    torch.set_grad_enabled(False)
    capture = {}
    timings = {}
    runtime = {"torch": torch.__version__, "python": str(sys.version), "precision": precision}
    device_name = sys.argv[sys.argv.index("--device") + 1]
    device = torch.device(device_name)
    cuda = device.type == "cuda"
    if cuda:
        runtime.update(gpu=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
                       total_bytes=torch.cuda.get_device_properties(device).total_memory,
                       native_bf16=torch.cuda.is_bf16_supported(including_emulation=False))
        torch.cuda.reset_peak_memory_stats(device)

    def timed(name, fn, *args, **kwargs):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if cuda:
                torch.cuda.synchronize(device)
            timings.setdefault(name, []).append(time.perf_counter() - start)

    class PriorTorch:
        """Only the pinned inferencer module sees this autocast override."""
        def __getattr__(self, name):
            return getattr(torch, name)

        def autocast(self, **options):
            return prior_autocast(torch.autocast, precision=precision, **options)

    if precision == "fp32-compat":
        inference_module.torch = PriorTorch()
    if cuda and runtime["capability"][0] < 7 and "--no_compile" in sys.argv:
        # The upstream CLI flag reaches the prior network but not FaceBoxes.
        from functools import partial

        inference_module.FaceDetector = partial(inference_module.FaceDetector, compile_model=False)
        runtime["facebox_compile"] = False

    original_infer = native.RealDenseFaceInferencer.infer_image
    original_crop = inference_module.crop_and_resize_image
    original_decode = native.GNFlameOptimizer.decode
    original_step = native.GNFlameOptimizer.fit_frame
    original_save = native.save_flame_npz
    original_camera = native.GNFlameOptimizer._build_config_camera
    original_energy = native.GNFlameOptimizer._evaluate_energy
    original_detector = native.RealDenseFaceInferencer._detect_face_bbox
    original_core = native.RealDenseFaceInferencer._run
    optimizer = None
    search = []

    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        capture["crop_bbox"] = np.asarray(result[1]).copy()
        capture["crop_size_hw"] = np.asarray(result[0].shape[:2])
        capture["crop_rgb"] = result[0].copy()
        return result

    def infer(self, *args, **kwargs):
        def save_maps(_model, _inputs, maps):
            names = ("raw_uv_coord", "raw_uv_coord_log_var", "raw_uv_depth", "raw_uv_depth_log_var")
            capture.update({name: array(value).copy() for name, value in zip(names, maps, strict=True)})
            capture["network_input"] = array(_inputs[0]).copy()

        def save_shape(name):
            def hook(_module, _inputs, outputs):
                tensors = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                capture[name] = np.asarray([list(item.shape) for item in tensors])
            return hook

        model = self.model
        encoder = model.encoder
        capture.update(encoder_taps=np.asarray(encoder.out_indices),
                       encoder_patch_hw=np.asarray(encoder.model.patch_embed.patch_size),
                       encoder_prefix_tokens=np.asarray(encoder.model.num_prefix_tokens),
                       network_parameters=np.asarray(sum(param.numel() for param in model.parameters())))
        for prefix in ("coord", "depth"):
            uv_layer = getattr(model, f"uv_feat_map_{prefix}")
            capture[f"uv_pyramid_{prefix}_shape"] = np.asarray([list(item.shape) for item in uv_layer.pyramid_params])

        hooks = [model.register_forward_hook(save_maps), encoder.register_forward_hook(save_shape("encoder_shapes"))]
        for name in ("uv_feat_map_coord", "uv_feat_map_depth", "decoder_coord", "decoder_depth",
                     "dpt_head_coord", "dpt_head_depth"):
            hooks.append(getattr(model, name).register_forward_hook(save_shape(f"{name}_shape")))
        try:
            result = timed("prior_detection_and_inference", original_infer, self, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if clean_priors is not None:
            for key in ("vertex_coord", "vertex_coord_log_var", "vertex_depth", "vertex_depth_log_var"):
                capture["inferred_" + key] = array(getattr(result, key)).copy()
            capture["oracle_replaced"] = np.asarray(substitute_priors(result, clean_priors, priors_mode))
        for key in ("vertex_coord", "vertex_coord_log_var", "vertex_depth", "vertex_depth_log_var", "face_bbox"):
            capture[key] = array(getattr(result, key)).copy()
        capture["input_size_hw"] = np.asarray([result.image_height, result.image_width])
        capture["uv_correspondence"] = capture["raw_uv_coord"][0].transpose(1, 2, 0) / self.scale_coord * .5 + .5
        capture["uv_relative_depth"] = capture["raw_uv_depth"][0, 0] / self.scale_depth
        for name in ("coord", "depth"):
            capture[f"uv_{name}_log_variance"] = capture[f"raw_uv_{name}_log_var"][0, 0].clip(-8, 3)
        np.savez_compressed(output / "priors.npz", **capture)
        return result

    def decode(self, result):
        vertices, joints = timed("mesh_decode", original_decode, self, result)
        neutral = FittingOutput(identity=result.identity, x=torch.zeros_like(result.x))
        neutral_vertices, _ = timed("neutral_decode", original_decode, self, neutral)
        with np.load("weights/flame/flame_assets.npz", allow_pickle=False) as support:
            faces = array(self.faces)
            if not np.array_equal(faces, support["faces"]):
                raise ValueError("FLAME connectivity differs from the UV support asset")
            np.savez_compressed(output / "mesh_native.npz", vertices=array(vertices),
                                faces=faces, uvs=support["uvs"], uv_faces=support["uv_faces"],
                                neutral_vertices=array(neutral_vertices), joints=array(joints))
        return vertices, joints

    def fit_frame(self, *args, **kwargs):
        nonlocal optimizer
        optimizer = self
        return timed("camera_search_and_solver", original_step, self, *args, **kwargs)

    def save_fit(path, result, width, height, config):
        original_save(path, result, width, height, config)
        decode(optimizer, result)

    def build_camera(self, w, h, fov):
        search.append({"fov_y_deg": float(fov)})
        return original_camera(self, w, h, fov)

    def evaluate_energy(self, *args, **kwargs):
        energy = original_energy(self, *args, **kwargs)
        if search and "mean_energy" not in search[-1]:
            search[-1]["mean_energy"] = float(energy.mean().item())
        return energy

    def detect(self, image):
        bbox = timed("facebox_detection", original_detector, self, image)
        capture["detected_bbox"] = np.full(4, np.nan) if bbox is None else bbox.copy()
        return bbox

    def run_core(self, tensor):
        return timed("network_and_uv_sampling", original_core, self, tensor)

    inference_module.crop_and_resize_image = crop
    native.RealDenseFaceInferencer.infer_image = infer
    native.RealDenseFaceInferencer._detect_face_bbox = detect
    native.RealDenseFaceInferencer._run = run_core
    native.GNFlameOptimizer.fit_frame = fit_frame
    native.GNFlameOptimizer._build_config_camera = build_camera
    native.GNFlameOptimizer._evaluate_energy = evaluate_energy
    native.save_flame_npz = save_fit
    try:
        native.main()
    finally:
        if cuda:
            runtime.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        (output / "timing.json").write_text(json.dumps({"runtime": runtime, "seconds": timings}, indent=2),
                                            encoding="utf-8")
        (output / "camera_search.json").write_text(json.dumps(search, indent=2), encoding="utf-8")



def regenerate(input_dir, output_path, coefficient=None, value=None):
    import torch
    import fit_single_image as native
    from common.types import FittingOutput

    import hashlib
    import json

    manifest = json.loads((input_dir / "scene.json").read_text(encoding="utf-8"))
    for name, filename in (("flame", "flame2023.pkl"), ("support", "flame_assets.npz")):
        with open(Path("weights/flame") / filename, "rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != manifest["assets"][name]["sha256"]:
            raise ValueError(f"{name} differs from the asset used for the source fit")
    with np.load(input_dir / "face.npz", allow_pickle=False) as saved:
        expression = saved["expression"].copy()
        if coefficient is not None:
            if not 0 <= coefficient < len(expression) or not np.isfinite(value):
                raise ValueError("Expression index or value is invalid")
            expression[coefficient] = value
        packed = np.concatenate((expression, saved["pose"], saved["translation"]))
        flame = native.FLAMEConfig(flame_model_path="weights/flame/flame2023.pkl",
                                   flame_assets_path="weights/flame/flame_assets.npz",
                                   num_expressions=len(expression))
        config = native.load_fitting_config("configs/fitting/single_image.yaml",
                                           native.GNFlameOptimizerConfig, flame=flame, device="cuda")
        optimizer = native.GNFlameOptimizer(config)
        fit = FittingOutput(identity=torch.from_numpy(saved["identity"]),
                            x=torch.from_numpy(packed))
        vertices, _ = optimizer.decode(fit)
        decoded = array(vertices)
        if coefficient is None:
            np.testing.assert_allclose(decoded, saved["vertices"], rtol=0, atol=1e-5)
        np.savez_compressed(output_path, vertices=decoded, faces=saved["faces"],
                            uvs=saved["uvs"], uv_faces=saved["uv_faces"],
                            identity=saved["identity"], expression=expression,
                            pose=saved["pose"], translation=saved["translation"])


def regenerate_command():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient", type=int)
    parser.add_argument("--value", type=float)
    args = parser.parse_args()
    if (args.coefficient is None) != (args.value is None):
        parser.error("Provide both --coefficient and --value for an expression edit")
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("Output must be a new .npz file")
    sys.path.insert(0, str(Path.cwd()))
    regenerate(args.regenerate, args.output, args.coefficient, args.value)


def preview_command():
    import cv2
    import torch

    sys.path.insert(0, str(Path.cwd()))
    from camera import IntrinsicsCamera, PerspectiveCamera
    from visualization import Visualizer, save_reconstruction_image

    output, input_path, device = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    torch.set_grad_enabled(False)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    scene = json.loads((output / "scene.json").read_text(encoding="utf-8"))
    intrinsics = np.asarray(scene["intrinsics"])
    extrinsics = np.asarray(scene["world_to_camera"])
    camera = scene["camera"]
    w, h = camera["image_width"], camera["image_height"]
    with np.load(output / "face.npz", allow_pickle=False) as face:
        vertices = face["vertices"]
        if "neutral_vertices" in face.files:
            save_neutral_landmarks(output, face["neutral_vertices"], vertices, face["faces"])

    def make_camera(K, width, height):
        return IntrinsicsCamera(K=K, R=extrinsics[:3, :3], T=extrinsics[:3, 3], width=width, height=height)

    native_camera = PerspectiveCamera(fov_y=np.deg2rad(camera["camera_fov_y"]),
                                      rot=np.asarray(camera["camera_rot"]), pos=np.asarray(camera["camera_pos"]),
                                      width=w, height=h)
    native_xy, native_valid = native_camera.project_points(torch.from_numpy(vertices), in_viewport=True)
    with np.load(output / "projection.npz", allow_pickle=False) as projected:
        np.testing.assert_array_equal(array(native_valid), projected["camera_clip_visible"])
        valid = projected["camera_clip_visible"]
        projection_error = float(np.abs(array(native_xy)[valid, :2] - projected["xy"][valid]).max())
        np.testing.assert_allclose(array(native_xy)[valid, :2], projected["xy"][valid], atol=.003, rtol=0)

    scale = min(1.0, 960 / max(w, h))
    pw, ph = round(w * scale), round(h * scale)
    # Preserve projection under the OpenCV pixel-center transform, even after rounding.
    sx, sy = pw / w, ph / h
    transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    preview_camera = make_camera(transform @ intrinsics, pw, ph)
    bgr = cv2.imread(input_path)
    if bgr is None or bgr.shape[:2] != (h, w):
        raise ValueError("Preview image could not decode at the original fitted dimensions")
    rgb = cv2.cvtColor(cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    visualizer = Visualizer(device=device)
    overlay, shading = visualizer.vis_flame_shading_overlay(preview_camera, rgb, vertices, return_shading=True)
    pivot = vertices.mean(axis=0)
    rot = extrinsics[:3, :3]
    side_yaw = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    side_vertices = (vertices - pivot) @ rot.T @ side_yaw @ rot + pivot
    _, side = visualizer.vis_flame_shading_overlay(preview_camera, rgb, side_vertices, return_shading=True)
    for name, pixels in (("overlay", overlay), ("render", shading), ("side", side)):
        save_reconstruction_image(output / f"{name}.png", pixels)
    result = {"preview_size_wh": [pw, ph], "input_to_preview": transform.tolist(),
              "native_projection_max_abs_px": projection_error, "visible_vertices": int(valid.sum())}
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    result["seconds"] = time.perf_counter() - start
    (output / "preview_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def save_neutral_landmarks(output, neutral, posed, faces):
    import hashlib
    import pickle
    import subprocess

    project = Path(__file__).resolve().parents[2]
    pixel = project / "third_party/Pixel3DMM"
    assets = pixel / "src/pixel3dmm/preprocessing/MICA/data/FLAME2020"
    template = assets / "generic_model.pkl"
    embedding_path = assets / "landmark_embedding.npy"
    if not template.is_file() or not embedding_path.is_file():
        return
    sys.path.insert(0, str(project))
    import common.fix_chumpy  # noqa: F401
    from src.geometry.landmarks import NOW_LANDMARK_INDICES, transfer_now_landmarks

    with template.open("rb") as handle:
        source_faces = pickle.load(handle, encoding="latin1")["f"]
    embedding = np.load(embedding_path, allow_pickle=True, encoding="latin1").item()
    for label, vertices in (("neutral", neutral), ("posed", posed)):
        points = transfer_now_landmarks(vertices, faces, source_faces, embedding)
        np.save(output / f"{label}_landmarks.npy", points)
    metadata = {"source": "MICA FLAME2020 full barycentric embedding staged with Pixel3DMM",
                "pixel3dmm_revision": subprocess.check_output(["git", "-C", str(pixel), "rev-parse", "HEAD"], text=True).strip(),
                "mica_demo": "https://github.com/Zielon/MICA/blob/master/demo.py#L678-L695",
                "topology_check": "all 9976 indexed triangles and corner order match",
                "indices_68_zero_based": NOW_LANDMARK_INDICES, "units": "native FLAME coordinate units"}
    for label, path in (("template", template), ("embedding", embedding_path)):
        with path.open("rb") as handle:
            metadata[f"{label}_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "landmark_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def cpu_parity_command():
    """Replay the captured normalized 512 crop through the same checkpoint on CPU."""
    import torch

    upstream = Path.cwd()
    sys.path.insert(0, str(upstream))
    from common.config import load_model_config
    from models.realdenseface import RealDenseFace

    output = Path(sys.argv[2])
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    model_id = run["model"]
    config = load_model_config(upstream / f"configs/model/{model_id}.yaml")
    torch.set_num_threads(2)
    names = ("raw_uv_coord", "raw_uv_coord_log_var", "raw_uv_depth", "raw_uv_depth_log_var")
    with torch.inference_mode(), np.load(output / "priors.npz", allow_pickle=False) as priors:
        model = RealDenseFace(**config).eval()
        model.load_state_dict(torch.load(upstream / f"weights/realdenseface/{model_id}.pth", map_location="cpu"))
        model.build_inference_cache()
        maps = [array(result) for result in model(torch.from_numpy(priors["network_input"]))]
        comparison = {}
        for name, cpu in zip(names, maps, strict=True):
            error = np.abs(cpu - priors[name])
            comparison[name] = {"shape": list(cpu.shape), "max_abs": float(error.max()),
                                "mean_abs": float(error.mean())}
        np.savez_compressed(output / "same_crop_cpu_maps.npz", **dict(zip(names, maps, strict=True)),
                            crop_bbox=priors["crop_bbox"])
    (output / "cpu_parity.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate_command()
    elif "--preview" in sys.argv:
        preview_command()
    elif "--cpu-parity" in sys.argv:
        cpu_parity_command()
    else:
        main()
