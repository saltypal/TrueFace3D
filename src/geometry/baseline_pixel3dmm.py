"""Run the upstream single-image pipeline with explicit output checks per stage."""

import argparse
from functools import partial
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from time import perf_counter, time
from types import ModuleType

import numpy as np

from .baselines import DEADLINE_ENV, MICA, PIXEL, ROOT
from .baseline_worker import NOW_INDICES, export_neutral, fit_mica, stage


CASE = "single"
TORCH_CACHE = ROOT / "checkpoints/pixel3dmm/torch"
SEGMENTATION_WEIGHTS = (
    "mobilenet0.25_Final.pth",
    "face_parsing.farl.celebm.main_ema_181500_jit.pt",
)
CUDA_STAGES = ("uv_map", "normals", "tracking")


def _run_stage(name, native):
    env = dict(os.environ)
    env.update(PIXEL3DMM_CODE_BASE=PIXEL.as_posix(),
               PIXEL3DMM_PREPROCESSED_DATA=(native / "preprocessed").as_posix(),
               PIXEL3DMM_TRACKING_OUTPUT=(native / "tracking").as_posix(),
               TORCH_HOME=os.environ.get("TORCH_HOME", str(TORCH_CACHE)),
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               PYTHONUNBUFFERED="1")
    if name in {"crop", "mica", "segmentation"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if name == "mica":
        env["TRUEFACE3D_COMPARATOR_DEVICE"] = "cpu"
    if name == "tracking":
        # The tracker compiles projection at import time. Pascal needs eager CUDA.
        env["TORCHDYNAMO_DISABLE"] = "1"
    command = [sys.executable, "-m", "src.geometry.baseline_pixel3dmm", name, str(native)]
    timeout = 1800.0
    if deadline := env.get(DEADLINE_ENV):
        # Give the parent process time to collect the failed stage receipt.
        timeout = min(timeout, max(0.0, float(deadline) - time() - 3))
    with (native / f"{name}.stdout.log").open("wb") as stdout, (
        native / f"{name}.stderr.log"
    ).open("wb") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   timeout=timeout, check=False)
    if completed.returncode:
        lines = (native / f"{name}.stderr.log").read_text(encoding="utf-8",
                                                        errors="replace").splitlines()
        detail = next((line for line in reversed(lines) if line.strip()), "No error text")
        raise RuntimeError(f"{name} exited with {completed.returncode}: {detail}")
    if name in CUDA_STAGES:
        return json.loads((native / f"{name}.cuda.json").read_text(encoding="utf-8"))
    return None


def _one(directory, suffix):
    files = sorted(part for part in directory.glob(suffix) if part.is_file() and part.stat().st_size)
    if len(files) != 1:
        raise ValueError(f"Expected one {suffix} file in {directory}, found {len(files)}")
    return files[0]


def fit_pixel3dmm(image_path, run_dir, metadata):
    native = run_dir / "native"
    prepared = native / "preprocessed" / CASE
    inputs = native / "input" / CASE
    metadata.update(mica_initialization=True, tracker_iters=800, tracker_compile=False,
                    predictor_eval_mode=True, requested_device="cuda", device="cpu",
                    actual_fit_device=None, detector_device="cpu", stage_cuda_metrics={})
    with stage("pixel3dmm_crop"):
        inputs.mkdir(parents=True)
        source = inputs / f"00000{image_path.suffix.lower()}"
        shutil.copyfile(image_path, source)
        rgb = prepared / "rgb"
        rgb.mkdir(parents=True)
        shutil.copyfile(source, rgb / source.name)
        _run_stage("crop", native)
        crop = _one(prepared / "cropped", "00000.*")
        _one(prepared / "PIPnet_landmarks", "00000.npy")
        metadata["crop"] = str(crop.relative_to(run_dir))
    with stage("pixel3dmm_mica_initialization"):
        _run_stage("mica", native)
        _one(prepared / "mica" / crop.stem, "identity.npy")
    with stage("pixel3dmm_segmentation"):
        _run_stage("segmentation", native)
        _one(prepared / "seg_og", "00000.*")
    for prediction in ("uv_map", "normals"):
        with stage(f"pixel3dmm_{prediction}"):
            metadata["stage_cuda_metrics"][prediction] = _run_stage(prediction, native)
            metadata["device"] = "cpu+cuda"
            _one(prepared / "p3dmm" / prediction, "00000.*")
    with stage("pixel3dmm_tracking"):
        metadata["stage_cuda_metrics"]["tracking"] = _run_stage("tracking", native)
        metrics = metadata["stage_cuda_metrics"].values()
        metadata.update(
            actual_fit_device="cuda", fit_device="cuda",
            peak_allocated_bytes=max(entry["peak_allocated_bytes"] for entry in metrics),
            peak_reserved_bytes=max(entry["peak_reserved_bytes"] for entry in metrics),
            memory_scope="maximum_pytorch_cuda_stage_peak_across_isolated_children",
        )
    with stage("pixel3dmm_export"):
        metadata.update(export_pixel_geometry(run_dir))


def export_pixel_geometry(run_dir):
    import trimesh

    native = run_dir / "native"
    canonical = _one(native / "tracking", "*/mesh/canonical.ply")
    native_now = _one(native / "tracking", "*/mesh/now_0.ply.npy")
    mesh = trimesh.load_mesh(canonical, process=False)
    landmarks = np.load(native_now, allow_pickle=False)
    # The tracker emits static face landmarks for these seven indices from
    # its zero-expression, zero-pose decode. Validate in the canonical frame.
    try:
        landmarks, provenance = _validate_pixel_landmarks(mesh.vertices, mesh.faces, landmarks)
    except ValueError as error:
        landmarks, provenance = None, {"landmark_status": str(error)}
    export_neutral(run_dir, mesh.vertices, mesh.faces, landmarks)
    return {**provenance, "mesh_unit_to_mm": 1000.0, "mesh_units": "m",
            "mesh_vertices": len(mesh.vertices), "mesh_faces": len(mesh.faces),
            "camera": "native tracking checkpoint"}


def _validate_pixel_landmarks(vertices, faces, points):
    import pickle

    from .baseline_worker import load_chumpy

    load_chumpy()
    basis = MICA / "data/FLAME2020/generic_model.pkl"
    embedding = MICA / "data/FLAME2020/landmark_embedding.npy"
    with basis.open("rb") as stream:
        basis_faces = pickle.load(stream, encoding="latin1")["f"]
    if not np.array_equal(basis_faces, faces):
        raise ValueError("Pixel3DMM canonical topology differs from the FLAME2020 basis")
    stored = np.load(embedding, allow_pickle=True, encoding="latin1").item()
    # The tracker concatenates 17 dynamic outline points with the 51 static points.
    selection = np.asarray(NOW_INDICES) - 17
    indices = np.asarray(stored["static_lmk_faces_idx"])[selection]
    barycentric = np.asarray(stored["static_lmk_bary_coords"])[selection]
    if (np.any(indices < 0) or np.any(indices >= len(faces))
        or not np.isfinite(barycentric).all()
        or not np.allclose(barycentric.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid Pixel3DMM static landmark map")
    from_surface = np.einsum("lk,lkc->lc", barycentric, vertices[faces[indices]])
    metadata = {
        "landmark_embedding_sha256": sha256(embedding.read_bytes()).hexdigest(),
        "flame_basis_sha256": sha256(basis.read_bytes()).hexdigest(),
    }
    if points.shape != (7, 3) or not np.allclose(points, from_surface, rtol=0, atol=2e-6):
        return None, {**metadata, "landmark_status": "native landmarks differ from canonical mesh"}
    return points, {**metadata, "landmark_indices_ibug68": list(NOW_INDICES)}


def _pipnet_nms(detections, threshold):
    """CPU NMS with the inclusive boxes used by the legacy FaceBoxes Cython extension."""
    boxes = detections[:, :4]
    areas = np.prod(boxes[:, 2:] - boxes[:, :2] + 1, axis=1)
    remaining = detections[:, 4].argsort()[::-1]
    kept = []
    while len(remaining):
        current, remaining = remaining[0], remaining[1:]
        kept.append(current)
        upper = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        lower = np.maximum(boxes[current, :2], boxes[remaining, :2])
        overlap = np.prod(np.maximum(0.0, upper - lower + 1), axis=1)
        union = areas[current] + areas[remaining] - overlap
        remaining = remaining[overlap < threshold * union]
    return kept


def _offline_weights(url, *args, **kwargs):
    raise FileNotFoundError(f"Required pretrained weight is absent from the local cache: {url}")


def _missing_inference_weights(model, loaded):
    inactive = ("net.dir_norm_uv.", "net.uv_enc_mlp.") if not model.cfg.model.use_uv_enc else ()
    return [key for key in loaded.missing_keys if key.startswith("net.") and not key.startswith(inactive)]


def _load_facer_replacements():
    """Apply the two modules installed by Pixel3DMM's upstream setup script."""
    import facer.face_parsing

    replacements = PIXEL / "src/pixel3dmm/preprocessing/replacement_code"
    for name, file in (("facer.transform", "facer_transform.py"),
                       ("facer.face_parsing.farl", "farl.py")):
        spec = spec_from_file_location(name, replacements / file)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    facer.face_parsing.FaRLFaceParser = sys.modules["facer.face_parsing.farl"].FaRLFaceParser


def _tracking_flame_assets(native):
    source = MICA / "data/FLAME2020"
    mask_path = Path("FLAME_masks/FLAME_masks.pkl")
    original = (source / mask_path).read_bytes()
    if not original.startswith(b"(dp0\r\n"):
        return source.parent
    # CRLF checkout filters corrupt protocol-0 pickle string delimiters.
    # Restore LF in a run-local copy and leave the vendor asset untouched.
    local = native / "flame-assets/FLAME2020"
    (local / mask_path.parent).mkdir(parents=True)
    (local / mask_path).write_bytes(original.replace(b"\r\n", b"\n"))
    for filename in ("generic_model.pkl", "landmark_embedding.npy"):
        try:
            os.link(source / filename, local / filename)
        except OSError:
            shutil.copyfile(source / filename, local / filename)
    print(f"Restored LF delimiters in FLAME mask at {local / mask_path}", flush=True)
    return local.parent


def _run_upstream(name, native):
    import torch

    torch.set_num_threads(4)
    torch.hub.download_url_to_file = _offline_weights
    if name in {"crop", "mica", "segmentation"}:
        torch.cuda.is_available = lambda: False
    scripts = PIXEL / "scripts"
    prepared = native / "preprocessed" / CASE
    if name == "crop":
        import scipy.integrate as integrate
        import torchvision.models

        # PIPNet immediately loads its full landmark network state dict. Avoid
        # downloading the transient ImageNet initializer for that backbone.
        resnet18 = torchvision.models.resnet18
        torchvision.models.resnet18 = lambda **_: resnet18(weights=None)
        if not hasattr(integrate, "simps"):
            integrate.simps = integrate.simpson
        torch.load = partial(torch.load, map_location="cpu")
        nms_wrapper = ModuleType("utils.nms_wrapper")
        nms_wrapper.nms = _pipnet_nms
        sys.modules[nms_wrapper.__name__] = nms_wrapper
        runner = runpy.run_path(str(scripts / "run_cropping.py"))
        runner["main"]((native / "input" / CASE).as_posix())
    elif name == "mica":
        crop = _one(prepared / "cropped", "00000.*")
        scratch = native / "mica-init"
        (scratch / "native").mkdir(parents=True)
        fit_mica(crop, scratch, {})
        output = prepared / "mica" / crop.stem
        shutil.copytree(scratch / "native", output)
    elif name == "segmentation":
        cache = Path(torch.hub.get_dir()) / "checkpoints"
        missing = [str(cache / name) for name in SEGMENTATION_WEIGHTS if not (cache / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing local segmentation weights {missing}")
        _load_facer_replacements()
        runner = runpy.run_path(str(scripts / "run_facer_segmentation.py"))
        runner["main"](video_name=CASE)
    elif name in {"uv_map", "normals"}:
        import timm
        from pixel3dmm.lightning.p3dmm_system import system

        # The fine-tuned DINO backbone is included in each release checkpoint.
        # Skip only the transient initializer and fail if the final state
        # would leave any of the main inference network's weights uninitialized.
        create_model = timm.create_model
        load_state_dict = system.load_state_dict

        def from_staged_checkpoint(model_name, **kwargs):
            return create_model(model_name, **{**kwargs, "pretrained": False})

        def validate_loaded_weights(model, state_dict, *args, **kwargs):
            loaded = load_state_dict(model, state_dict, *args, **kwargs)
            missing = _missing_inference_weights(model, loaded)
            if missing:
                raise RuntimeError(f"Incomplete Pixel3DMM inference state: {missing}")
            return loaded

        def load_release_checkpoint(cls, path, *, strict=False):
            # The official training checkpoint also holds optimizer and old
            # camera network states. Keep that storage file-backed on CPU.
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            model = cls(checkpoint["hyper_parameters"]["cfg"])
            model.load_state_dict(checkpoint["state_dict"], strict=strict)
            return model.eval()

        timm.create_model = from_staged_checkpoint
        system.load_state_dict = validate_loaded_weights
        system.load_from_checkpoint = classmethod(load_release_checkpoint)
        try:
            import pytorch3d._C
        except ImportError:
            # The release script imports this optional visualization utility
            # eagerly. Dense predictions do not use it with viz_uv_mesh=False.
            visualization = ModuleType("pixel3dmm.utils.uv")

            def unavailable_visualization(*args, **kwargs):
                raise RuntimeError("Pixel3DMM UV visualization requires PyTorch3D")

            visualization.uv_pred_to_mesh = unavailable_visualization
            sys.modules[visualization.__name__] = visualization
        sys.argv = ["network_inference.py", f"video_name={CASE}", f"model.prediction_type={name}",
                    "viz_uv_mesh=False"]
        runpy.run_path(str(scripts / "network_inference.py"), run_name="__main__")
    elif name == "tracking":
        import mediapy
        from omegaconf import OmegaConf
        from pixel3dmm import env_paths

        if shutil.which("ffmpeg") is None:
            import imageio_ffmpeg
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        from pixel3dmm.tracking import tracker
        from .baseline_worker import load_chumpy

        load_chumpy()
        env_paths.FLAME_ASSETS = str(_tracking_flame_assets(native))
        tracker.COMPILE = False
        config = OmegaConf.merge(
            OmegaConf.load(PIXEL / "configs/tracking.yaml"),
            OmegaConf.from_dotlist([f"video_name={CASE}", "iters=800", "save_landmarks=True",
                                    "use_flame2023=False", "ignore_mica=False"]),
        )
        tracker.Tracker(config).run()


def _run_cuda_stage(name, native):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    _run_upstream(name, native)
    torch.cuda.synchronize()
    metrics = {
        "scope": "synchronized_child_stage_wall_including_load_and_native_export",
        "seconds": perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (native / f"{name}.cuda.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("native", type=Path)
    args = parser.parse_args()
    if args.stage in CUDA_STAGES:
        _run_cuda_stage(args.stage, args.native)
    else:
        _run_upstream(args.stage, args.native)
