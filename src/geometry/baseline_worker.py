"""Geometry-only comparator workers. Upstream imports stay in this process."""

import argparse
import builtins
import contextlib
from hashlib import sha256
import importlib
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

from .baselines import DECA, EMOCA_CKPT, MICA, SCRFD, SPECS, _model_metadata, requested_device


NOW_INDICES = (36, 39, 42, 45, 33, 48, 54)


class StageError(RuntimeError):
    def __init__(self, stage, message):
        self.stage = stage
        super().__init__(message)


@contextlib.contextmanager
def stage(name):
    print(f"[{name}] started", flush=True)
    try:
        yield
    except StageError:
        raise
    except Exception as error:
        raise StageError(name, str(error)) from error
    print(f"[{name}] complete", flush=True)


def load_chumpy():
    """Load the FLAME2020 pickle dependency on current Python and NumPy."""
    aliases = {name: getattr(builtins, name) for name in
               ("bool", "int", "float", "complex", "object", "str")}
    aliases["unicode"] = str
    missing = {key: value for key, value in aliases.items() if key not in vars(np)}
    needs_getargspec = not hasattr(inspect, "getargspec")
    if needs_getargspec:
        inspect.getargspec = inspect.getfullargspec
    vars(np).update(missing)
    try:
        return importlib.import_module("chumpy")
    finally:
        for key in missing:
            delattr(np, key)
        if needs_getargspec:
            del inspect.getargspec


def probe_runtime(model_id):
    _, _, _, dependencies, _ = SPECS[model_id]
    errors = {}
    for module in dependencies:
        try:
            load_chumpy() if module == "chumpy" else importlib.import_module(module)
        except Exception as error:
            errors[module] = f"{type(error).__name__}: {error}"
    if requested_device(model_id) == "cuda" and "torch" not in errors:
        import torch
        if not torch.cuda.is_available():
            errors["cuda"] = "PyTorch CUDA is unavailable"
    return {"runtime_ready": not errors, "dependency_errors": errors}


def read_bgr(path):
    import cv2
    from PIL import Image

    with Image.open(path) as image:
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def detect_face(bgr, input_size):
    """SCRFD on the photo; if no face is found, retry with a black border (tight face crops make the face
    too large for SCRFD at 640 px, e.g. FaceScape-C 768 px crops) and shift the result back."""
    import cv2
    try:
        return _detect_face(bgr, input_size)
    except ValueError as first:
        for fraction in (0.25, 0.5, 1.0):
            pad = int(round(fraction * max(bgr.shape[:2])))
            padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            try:
                box, keypoints, score = _detect_face(padded, input_size)
            except ValueError:
                continue
            return box - pad, keypoints - pad, score
        raise first


def _detect_face(bgr, input_size):
    """Decode the staged antelopev2 SCRFD boxes and five keypoints on CPU."""
    import cv2
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # Exported anchors advertise 640-static output shapes.
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(SCRFD), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    height, width = bgr.shape[:2]
    resize = input_size / max(height, width)
    new_width, new_height = max(1, int(width * resize)), max(1, int(height * resize))
    resize = new_height / height  # InsightFace uses the height-derived scale.
    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded[:new_height, :new_width] = cv2.resize(bgr, (new_width, new_height))
    blob = cv2.dnn.blobFromImage(padded, 1 / 128, (input_size, input_size),
                                 (127.5,) * 3, swapRB=True)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    if len(outputs) != 9:
        raise ValueError("Unexpected SCRFD output count")
    boxes, scores, landmarks = [], [], []
    for index, stride in enumerate((8, 16, 32)):
        grid_size = input_size // stride
        grid = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), axis=-1)
        anchors = np.repeat(grid.reshape(-1, 2) * stride, 2, axis=0)
        confidences = outputs[index].reshape(-1)
        distances = outputs[index + 3] * stride
        keypoints = outputs[index + 6].reshape(-1, 5, 2) * stride
        if len(confidences) != len(anchors):
            raise ValueError("Unexpected SCRFD anchor count")
        valid = confidences >= 0.5
        boxes.append(np.column_stack((anchors - distances[:, :2],
                                      anchors + distances[:, 2:]))[valid] / resize)
        scores.append(confidences[valid])
        landmarks.append((keypoints + anchors[:, None, :])[valid] / resize)
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    landmarks = np.concatenate(landmarks)
    if not len(boxes):
        raise ValueError(f"No face detected at SCRFD input size {input_size}")
    xywh = np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2] + 1))
    selected = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.499999, 0.4)).ravel()
    if not len(selected):
        raise ValueError("No SCRFD box survived suppression")
    centers = (boxes[selected, :2] + boxes[selected, 2:]) / 2
    closest = selected[np.argmin(np.sum((centers - (width // 2, height // 2)) ** 2, axis=1))]
    return boxes[closest], landmarks[closest], float(scores[closest])


def write_obj(path, vertices, faces):
    with path.open("w", encoding="ascii") as stream:
        np.savetxt(stream, vertices, fmt="v %.9g %.9g %.9g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")


def checked_landmarks(vertices, faces, indices, barycentric):
    """Interpolate landmarks using the indexed FLAME face array."""
    if indices.shape == (1, 68) and barycentric.shape == (1, 68, 3):
        indices, barycentric = indices[0], barycentric[0]
    if (indices.shape != (68,) or barycentric.shape != (68, 3)
        or np.any(indices < 0) or np.any(indices >= len(faces))
        or not np.isfinite(barycentric).all()
        or np.any(barycentric < -1e-6) or np.any(barycentric > 1 + 1e-6)
        or not np.allclose(barycentric.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Unverified FLAME landmark embedding")
    return np.einsum("lk,lkc->lc", barycentric, vertices[faces[indices]])


def export_neutral(run_dir, vertices, faces, landmarks=None):
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
        or not np.isfinite(vertices).all() or np.any(np.ptp(vertices, axis=0) <= 0)):
        raise ValueError("Invalid neutral vertices")
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer) or np.min(faces) < 0
        or np.max(faces) >= len(vertices)):
        raise ValueError("Invalid neutral faces")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("Degenerate neutral mesh triangles")
    if landmarks is not None:
        if (landmarks.shape != (7, 3) or not np.isfinite(landmarks).all()
            or len(np.unique(landmarks, axis=0)) != 7):
            raise ValueError("Invalid NoW landmarks")
        np.save(run_dir / "neutral_landmarks.npy", landmarks, allow_pickle=False)
    temporary = run_dir / "neutral.obj.tmp"
    write_obj(temporary, vertices, faces)
    temporary.replace(run_dir / "neutral.obj")


def _flame_outputs(flame, vertices, native_landmarks, basis, embedding, native):
    vertices = vertices[0].cpu().numpy()
    faces = flame.faces_tensor.cpu().numpy()
    landmarks = checked_landmarks(vertices, faces, flame.full_lmk_faces_idx.cpu().numpy(),
                                 flame.full_lmk_bary_coords.cpu().numpy())
    if not np.allclose(native_landmarks[0].cpu().numpy(), landmarks, atol=2e-6, rtol=0):
        raise ValueError("FLAME native landmarks disagree with topology interpolation")
    np.save(native / "landmarks68.npy", landmarks, allow_pickle=False)
    return vertices, faces, landmarks, {
        "landmark_indices_ibug68": list(NOW_INDICES),
        "landmark_embedding_sha256": sha256(embedding.read_bytes()).hexdigest(),
        "flame_basis_sha256": sha256(basis.read_bytes()).hexdigest(),
        "mesh_faces_sha256": sha256(faces.astype("<i8").tobytes()).hexdigest(),
    }


@contextlib.contextmanager
def timed_forward(device, metadata):
    import torch

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    with stage("inference"), torch.inference_mode():
        yield
        if cuda:
            torch.cuda.synchronize(device)
    metadata["inference_seconds"] = perf_counter() - started
    if cuda:
        metadata.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        memory_scope="torch_inference_with_resident_weights")


def fit_deca(image_path, run_dir, metadata, variant="deca"):
    """DECA, or EMOCA v2 (variant="emoca"): EMOCA's E_flame plus its expression encoder replaces exp."""
    import cv2
    import torch
    from skimage.transform import estimate_transform, warp
    from types import SimpleNamespace

    native = run_dir / "native"
    device = torch.device(requested_device(variant))
    with stage("preprocess"):
        bgr = read_bgr(image_path)
        box, keypoints, confidence = detect_face(bgr, 640)
        left, top, right, bottom = box
        old_size = ((right - left) + (bottom - top)) / 2
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2 + old_size * 0.12  # DECA TestData bbox mode.
        half = int(old_size * 1.25) / 2
        src = np.array([[center_x - half, center_y - half],
                        [center_x - half, center_y + half],
                        [center_x + half, center_y - half]])
        tform = estimate_transform("similarity", src, [[0, 0], [0, 223], [223, 0]])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = warp(rgb, tform.inverse, output_shape=(224, 224))
        cv2.imwrite(str(native / "crop.png"), (crop[:, :, ::-1] * 255).round().astype(np.uint8))
        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float()[None]
        metadata.update(detector="antelopev2 SCRFD 640 CPU; DECA bbox crop fallback",
                        detector_confidence=confidence, input_size=[224, 224],
                        native_detector_parity=False)
    basis = DECA / "data/generic_model.pkl"
    embedding = DECA / "data/landmark_embedding.npy"
    with stage("checkpoint_load"):
        sys.path.insert(0, str(DECA))
        from decalib.models.encoders import ResnetEncoder
        from decalib.models.FLAME import FLAME

        load_chumpy()
        config = SimpleNamespace(n_shape=100, n_exp=50, flame_model_path=basis,
                                 flame_lmk_embedding_path=embedding)
        widths = {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}
        encoder = ResnetEncoder(sum(widths.values())).eval()
        expression_encoder = None
        if variant == "emoca":
            from ..models.safe_ckpt import load_state_dict
            state = load_state_dict(EMOCA_CKPT)
            part = lambda prefix: {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}  # noqa: E731
            encoder.load_state_dict(part("deca.E_flame."), strict=True)
            expression_encoder = ResnetEncoder(50).eval()
            expression_encoder.load_state_dict(part("deca.E_expression."), strict=True)
            expression_encoder.to(device)
            del state
            metadata.update(emoca_checkpoint=str(EMOCA_CKPT.name), emoca_expression_encoder=True)
        else:
            checkpoint = torch.load(DECA / "data/deca_model.tar", map_location="cpu", weights_only=True)
            encoder.load_state_dict(checkpoint["E_flame"], strict=True)
            del checkpoint
        encoder.to(device)
        flame = FLAME(config).eval().to(device)
    with timed_forward(device, metadata):
        output = dict(zip(widths, encoder(tensor.to(device)).split(tuple(widths.values()), dim=1)))
        if expression_encoder is not None:
            output["exp"] = expression_encoder(tensor.to(device))
        shaped = output["shape"]
        expression, pose = output["exp"], output["pose"]
        posed, _, _ = flame(shaped, expression, pose)
        neutral, _, landmarks = flame(shaped, torch.zeros_like(expression), torch.zeros_like(pose))
    with stage("export"):
        vertices, faces, points, provenance = _flame_outputs(
            flame, neutral, landmarks, basis, embedding, native)
        np.savez_compressed(native / "native_fit.npz",
                            **{key: value[0].cpu().numpy() for key, value in output.items()},
                            image_to_crop=tform.params, detected_box=box,
                            detected_landmarks=keypoints, camera_model="scaled_orthographic")
        write_obj(native / "posed.obj", posed[0].cpu().numpy(), faces)
        # The original NoW exporter writes native FLAME coordinates in metres.
        export_neutral(run_dir, vertices, faces, points[list(NOW_INDICES)])
        metadata.update(provenance, mesh_unit_to_mm=1000.0, mesh_units="m",
                        mesh_vertices=len(vertices), mesh_faces=len(faces),
                        camera="scaled orthographic in aligned crop coordinates")


def fit_mica(image_path, run_dir, metadata):
    import cv2
    import torch
    from skimage.transform import SimilarityTransform
    from types import SimpleNamespace

    native = run_dir / "native"
    device = torch.device(requested_device("mica"))
    with stage("preprocess"):
        bgr = read_bgr(image_path)
        box, keypoints, confidence = detect_face(bgr, 224)
        template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        transform = SimilarityTransform()
        if not transform.estimate(keypoints, template):
            raise ValueError("Could not align ArcFace's five keypoints")
        crop = cv2.warpAffine(bgr, transform.params[:2], (112, 112))
        cv2.imwrite(str(native / "crop.png"), crop)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        metadata.update(detector="antelopev2 SCRFD 224 ONNX Runtime CPU",
                        detector_confidence=confidence, input_size=[112, 112])
    basis = MICA / "data/FLAME2020/generic_model.pkl"
    embedding = MICA / "data/FLAME2020/landmark_embedding.npy"
    with stage("checkpoint_load"):
        sys.path.insert(0, str(MICA))
        from models.arcface import Arcface
        from models.generator import Generator

        load_chumpy()
        config = SimpleNamespace(n_shape=300, flame_model_path=basis,
                                 flame_lmk_embedding_path=embedding)
        arcface = Arcface().eval()
        generator = Generator(512, 300, 300, 3, config, "cpu").eval()
        # The staged official release contains validation metadata with NumPy arrays.
        checkpoint = torch.load(MICA / "data/pretrained/mica.tar", map_location="cpu",
                                weights_only=False)
        arcface.load_state_dict(checkpoint["arcface"], strict=True)
        flame = generator.generator
        weights = checkpoint["flameModel"]
        for buffer in ("faces_tensor", "full_lmk_faces_idx", "full_lmk_bary_coords"):
            torch.testing.assert_close(getattr(flame, buffer), weights[f"generator.{buffer}"],
                                       rtol=0, atol=0)
        generator.load_state_dict(weights, strict=True)
        del checkpoint, weights
        arcface.to(device)
        generator.to(device)
    with timed_forward(device, metadata):
        identity = torch.nn.functional.normalize(arcface(torch.from_numpy(blob).to(device)))
        neutral, shape = generator(identity)
        landmarks = flame.compute_landmarks(neutral)
    with stage("export"):
        vertices, faces, points, provenance = _flame_outputs(
            flame, neutral, landmarks, basis, embedding, native)
        np.save(native / "identity.npy", shape[0].cpu().numpy(), allow_pickle=False)
        np.savez_compressed(native / "native_fit.npz", arcface=identity[0].cpu().numpy(),
                            arcface_input=blob[0], image_to_crop=transform.params,
                            detected_box=box, detected_landmarks=keypoints)
        np.save(native / "kpt68.npy", points * 1000, allow_pickle=False)
        np.save(native / "kpt7.npy", points[list(NOW_INDICES)] * 1000, allow_pickle=False)
        export_neutral(run_dir, vertices * 1000, faces, points[list(NOW_INDICES)] * 1000)
        metadata.update(provenance, mesh_unit_to_mm=1.0, mesh_units="mm",
                        mesh_vertices=len(vertices), mesh_faces=len(faces), camera=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=SPECS)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    if args.qualify:
        with contextlib.redirect_stdout(sys.stderr):
            result = probe_runtime(args.model_id)
        print(json.dumps(result))
        return
    if args.image is None or args.run_dir is None:
        parser.error("Specify an image and run directory")
    started = perf_counter()
    result = {"fit_status": "failed", "error_stage": None, "message": None, "seconds": None}
    try:
        with stage("preflight"):
            source = _model_metadata(args.model_id)
            if source["revision"] != source["expected_revision"]:
                raise ValueError(f"Unexpected source revision {source['revision']}")
            if source["missing_assets"]:
                raise ValueError(f"Missing assets: {source['missing_assets']}")
            if not args.image.is_file():
                raise ValueError(f"Missing input image: {args.image}")
            runtime = probe_runtime(args.model_id)
            result.update(runtime)
            if args.model_id != "pixel3dmm" and not runtime["runtime_ready"]:
                raise ValueError(json.dumps(runtime["dependency_errors"]))
            (args.run_dir / "native").mkdir()
            import torch
            torch.set_num_threads(4)
        if args.model_id == "pixel3dmm":
            from .baseline_pixel3dmm import fit_pixel3dmm
            fit_pixel3dmm(args.image, args.run_dir, result)
        else:
            function = {"deca": fit_deca, "mica": fit_mica,
                        "emoca": lambda *a: fit_deca(*a, variant="emoca")}[args.model_id]
            result.update(fit_device=requested_device(args.model_id), detector_device="cpu")
            function(args.image, args.run_dir, result)
        result.update(fit_status="success", seconds=perf_counter() - started)
    except Exception as error:
        traceback.print_exc()
        result.update(error_stage=getattr(error, "stage", "worker"), message=str(error),
                      failure_elapsed_seconds=perf_counter() - started)
    path = args.run_dir / "baseline_result.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    main()
