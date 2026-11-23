"""NoW geometry evaluation and source-camera reprojection diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_now import load_scan_landmarks

LOCAL_SIMILARITY = "local_now_mask_similarity_icp"
LOCAL_METRICAL = "local_now_mask_fixed_scale_icp"


@dataclass(frozen=True)
class NoWScore:
    protocol: str
    reference_count: int
    distances_mm: np.ndarray
    mean_mm: float
    median_mm: float
    std_mm: float
    p95_mm: float
    within_2_5_mm: float
    alignment: np.ndarray | None = None


def distance_summary(distances_mm, protocol, alignment=None):
    values = np.asarray(distances_mm, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.min(values) < 0:
        raise ValueError("Expected a nonempty vector of finite unsigned distances")
    return NoWScore(protocol, len(values), values, float(values.mean()), float(np.median(values)),
                    float(values.std()), float(np.percentile(values, 95)),
                    float(np.mean(values <= 2.5)), alignment)


def _points(values, n=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or (n is not None and len(values) != n)
            or len(values) == 0 or not np.isfinite(values).all()):
        raise ValueError(f"Expected {'N' if n is None else n} finite XYZ points")
    return values


def now_face_mask(vertices, landmarks):
    vertices = _points(vertices)
    points = _points(landmarks, 7)
    bridge = (points[1] + points[2]) / 2
    nose = points[4]
    center = nose + 0.3 * (bridge - nose)
    radius = 0.7 * (np.linalg.norm(points[0] - points[3]) + np.linalg.norm(bridge - nose))
    if radius <= 0:
        raise ValueError("NoW scan annotations have a zero crop radius")
    return np.linalg.norm(vertices - center, axis=1) <= radius


def fit_similarity(source, target, scale=True):
    """Return a column-vector 4×4 similarity with a proper rotation."""
    source, target = _points(source), _points(target)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Expected at least three corresponding XYZ points")
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    a, b = source - src_mean, target - dst_mean
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Alignment landmarks are collinear")
    left, singular, right_t = np.linalg.svd(a.T @ b)
    orientation = np.ones(3)
    orientation[-1] = np.linalg.det(right_t.T @ left.T)
    rotation = (right_t.T * orientation) @ left.T
    gain = float(singular @ orientation / np.sum(a * a)) if scale else 1.0
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Invalid alignment scale")
    transform = np.eye(4)
    transform[:3, :3] = gain * rotation
    transform[:3, 3] = dst_mean - transform[:3, :3] @ src_mean
    return transform


def transform_points(points, transform):
    return _points(points) @ transform[:3, :3].T + transform[:3, 3]


def load_surface(path):
    import trimesh
    mesh = trimesh.load_mesh(Path(path), process=False)
    vertices = _points(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError(f"Invalid surface triangles: {path}")
    valid_faces = mesh.nondegenerate_faces(height=0)
    if not np.any(valid_faces):
        raise ValueError(f"Mesh has no positive-area triangles: {path}")
    if not np.all(valid_faces):
        mesh.update_faces(valid_faces)
    return mesh


def closest_surface(mesh, points, chunk_size=1024):
    import trimesh
    points = _points(points)
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    results = [trimesh.proximity.closest_point(mesh, points[i:i + chunk_size])
               for i in range(0, len(points), chunk_size)]
    return tuple(np.concatenate(arrays) for arrays in zip(*results))


def _load_prediction_landmarks(path):
    path = Path(path)
    values = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path)
    return _points(values, 7)


class Evaluator:
    def __init__(self, official_runner=None, max_iterations=12):
        if max_iterations < 0:
            raise ValueError("ICP iteration count cannot be negative")
        self.official_runner = official_runner
        self.max_iterations = max_iterations

    def reprojection(self, predicted, target, valid=None):
        predicted, target = np.asarray(predicted), np.asarray(target)
        if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
            raise ValueError("Expected matched N-by-2 points")
        mask = np.isfinite(predicted).all(axis=1) & np.isfinite(target).all(axis=1)
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != mask.shape:
                raise ValueError("Expected one validity flag per point")
            mask &= valid
        distances = np.linalg.norm(predicted[mask] - target[mask], axis=1)
        return {"valid_count": len(distances), "unit": "px",
                "mean": float(distances.mean()) if len(distances) else None,
                "median": float(np.median(distances)) if len(distances) else None}

    def score_now(self, scan, scan_lmk, pred_mesh, pred_lmk, protocol=LOCAL_SIMILARITY, unit_to_mm=None):
        if scan_lmk is None or pred_lmk is None:
            raise ValueError("NoW scoring requires seven scan points and seven predicted points")
        if protocol in {"official_now_nometrical", "official_now_metrical"}:
            if self.official_runner is None:
                raise RuntimeError("A qualified official NoW evaluator must be configured")
            return self.official_runner(scan, scan_lmk, pred_mesh, pred_lmk, protocol, unit_to_mm)
        if protocol not in {LOCAL_SIMILARITY, LOCAL_METRICAL}:
            raise ValueError(f"Unknown NoW protocol: {protocol}")

        scale = protocol == LOCAL_SIMILARITY
        if not scale and (unit_to_mm is None or not np.isfinite(unit_to_mm) or unit_to_mm <= 0):
            raise ValueError("Metrical scoring requires a verified mesh-unit-to-mm factor")
        factor = 1.0 if scale else unit_to_mm
        reference = load_surface(scan)
        reference_lmk = load_scan_landmarks(scan_lmk)
        samples = reference.vertices[now_face_mask(reference.vertices, reference_lmk)]
        if len(samples) < 3:
            raise ValueError("NoW face mask has too few reference vertices")
        mesh = load_surface(pred_mesh)
        mesh.vertices *= factor
        landmarks = _load_prediction_landmarks(pred_lmk) * factor
        transform = fit_similarity(landmarks, reference_lmk, scale=scale)
        mesh.apply_transform(transform)
        closest, distances, _ = closest_surface(mesh, samples)
        for _ in range(self.max_iterations):
            step = fit_similarity(closest, samples, scale=scale)
            candidate = mesh.copy()
            candidate.apply_transform(step)
            new_closest, new_distances, _ = closest_surface(candidate, samples)
            improvement = float(np.mean(distances ** 2) - np.mean(new_distances ** 2))
            if improvement < 0:
                break
            mesh, closest, distances = candidate, new_closest, new_distances
            transform = step @ transform
            if improvement < 1e-6:
                break
        transform[:3, :3] *= factor
        return distance_summary(distances, protocol, transform)
