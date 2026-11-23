"""Display and verify IMG_0041 scan comparisons from retained geometry fits."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from PIL import Image, ImageOps

from src.benchmark import MODEL_IDS, _score_values, file_digest, pair_variants, read_records, result_records
from src.data_now import installed_cases, load_face_box, load_scan_landmarks, pilot_cases
from src.evaluation import LOCAL_SIMILARITY, load_surface, now_face_mask, transform_points
from src.rendering.gallery import View, rasterize


CONDITIONS = {
    "original": "Original", "face-128": "128 px", "face-64": "64 px",
    "face-32": "32 px", "blur-mild": "Mild blur", "blur-heavy": "Heavy blur",
}
MODELS = {
    "vits": "RealDenseFace S", "vitb": "RealDenseFace B", "deca": "DECA",
    "pixel3dmm": "Pixel3DMM", "mica": "MICA",
}
METRICS = ("mean_mm", "median_mm", "p95_mm", "std_mm", "within_2_5_mm")
RUN = "one-photo-geometry-fresh-01"


@dataclass
class Comparison:
    case: object
    reference: object
    scan_points: np.ndarray
    scan_mask: np.ndarray
    landmarks: np.ndarray
    photos: dict
    meshes: dict
    errors: dict
    metrics: dict
    statuses: dict
    conditions: tuple


def load_comparison(root, *, partial=False):
    """Verify one fresh run against its source photo, scan, and score protocol."""
    root = Path(root)
    run = root / "experiments/geometry/runs" / RUN
    case, = pilot_cases(installed_cases(root / "Data"), count=1)
    rows = result_records(run) or (read_records(run) if partial else [])
    records = {(row.model_id, row.variant_id): row for row in rows}
    conditions = tuple(condition for condition in CONDITIONS
                       if all((model, condition) in records for model in MODEL_IDS))
    if not partial and (conditions != tuple(CONDITIONS) or len(records) != 30):
        raise ValueError("The fresh matrix must finish all 30 requested attempts")
    if "original" not in conditions:
        raise ValueError("Waiting for the five fresh original-input attempts")
    if any(row.image_key != case.image_key or row.run_id != RUN for row in rows):
        raise ValueError("The report expects fresh results for the selected portrait only")
    rows = [row for row in records.values() if row.variant_id in conditions]
    pairs = pair_variants([row for row in rows if row.variant_id != "original"], rows)
    if any(row["pair_status"] not in {"scored", "fit_unavailable", "score_unavailable"} for row in pairs):
        raise ValueError("Fresh pairs have incompatible score inputs")
    photos = {"original": case.image_path}
    for condition in conditions[1:]:
        photos[condition], = (run / "inputs").glob(f"*/{condition}.png")
    hashes = {condition: file_digest(path) for condition, path in photos.items()}
    reference = load_surface(case.scan_path)
    landmarks = load_scan_landmarks(case.scan_landmark_path)
    mask = now_face_mask(reference.vertices, landmarks)
    errors, meshes, metrics, statuses = {}, {}, {}, {}
    for row in rows:
        key = row.model_id, row.variant_id
        if row.input_sha256 != hashes[key[1]]:
            raise ValueError(f"Photo fingerprint mismatch for {key}")
        statuses[key] = (row.score_status, row.error_stage, row.message)
        if row.score_status != "success":
            continue
        folder = Path(row.case_dir)
        paths = (folder / "neutral.obj", folder / "fit/neutral.obj")
        mesh = load_surface(next(path for path in paths if path.is_file()))
        with np.load(Path(row.score_file).with_name("distances.npz"), allow_pickle=False) as data:
            matrix, distances = data["alignment"], data["distances_mm"]
        score = _score_values(row)
        if (distances.shape != (int(mask.sum()),) or matrix.shape != (4, 4)
                or not np.isfinite(matrix).all() or not np.isfinite(distances).all()
                or score["protocol"] != LOCAL_SIMILARITY
                or not np.isclose(distances.mean(), score["mean_mm"], atol=1e-9)):
            raise ValueError(f"Saved score inputs and summary disagree for {key}")
        mesh.apply_transform(matrix)
        meshes[key], errors[key], metrics[key] = mesh, distances, score
    return Comparison(case, reference, reference.vertices[mask], mask, landmarks,
                      photos, meshes, errors, metrics, statuses, conditions)


def metric_rows(data, condition):
    """Return absolute local scan scores, differences and aligned vertex movement."""
    rows = []
    for model in MODEL_IDS:
        current = data.metrics.get((model, condition))
        original = data.metrics.get((model, "original"))
        row = {"Model": MODELS[model], "Status": data.statuses[model, condition][0]}
        for key, label in zip(METRICS, ("Mean", "Median", "P95", "Std. dev.", "Within 2.5 mm")):
            factor, units = (100, "%") if key == "within_2_5_mm" else (1, "mm")
            value = current[key] * factor if current is not None else None
            delta = (current[key] - original[key]) * factor if current and original else None
            row[f"{label} ({units})"] = (
                "—" if value is None else f"{value:.3f}" if condition == "original" or delta is None
                else f"{value:.3f} ({delta:+.3f})"
            )
        if condition != "original":
            movement = mesh_movement(data, model, condition)
            row["Mean vertex movement (mm)"] = "—" if movement is None else f"{movement:.3f}"
        rows.append(row)
    return rows


def mesh_movement(data, model, condition):
    """Mean corresponding-vertex displacement after each mesh's scan alignment."""
    mesh, clean = data.meshes.get((model, condition)), data.meshes.get((model, "original"))
    if mesh is None or clean is None or not np.array_equal(clean.faces, mesh.faces):
        return None
    face = now_face_mask(clean.vertices, data.landmarks)
    inside = face & (np.asarray(clean.vertex_normals) @ face_frame(data.landmarks)[1][2] < 0)
    if not inside.any():
        return None
    return float(np.linalg.norm(mesh.vertices[inside] - clean.vertices[inside], axis=1).mean())


def face_frame(landmarks, yaw=0):
    """Camera axes from the scan landmarks. Rotate a shared view around vertical."""
    center = landmarks.mean(axis=0)
    right = landmarks[3] - landmarks[0]
    right /= np.linalg.norm(right)
    up = landmarks[:4].mean(axis=0) - landmarks[5:7].mean(axis=0)
    up -= (up @ right) * right
    up /= np.linalg.norm(up)
    forward = np.cross(right, up)
    if (landmarks[4] - landmarks[:4].mean(axis=0)) @ forward < 0:
        right, forward = -right, -forward
    axes = np.stack([right, -up, -forward])
    angle = np.deg2rad(yaw)
    c, s = np.cos(angle), np.sin(angle)
    return center, np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ axes


def _surface_view(mesh, vertices, normals, view, size):
    _, triangles, bary = rasterize(view, vertices, mesh.faces, size, surface=True)
    visible = triangles >= 0
    faces = mesh.faces[triangles[visible]]
    weights = bary[visible]
    n = (normals[faces] * weights[..., None]).sum(axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    strength = np.maximum(n @ [-0.35, -0.42, -0.837], 0)
    shade = (0.46 + 0.49 * strength[:, None]) * np.array([0.80, 0.85, 0.91])
    clay = np.full((size, size, 3), [0.10, 0.12, 0.15])
    clay[visible] = shade
    return clay, visible, faces, weights


def _scan_colors(values, data, surface, cmap, norm):
    clay, visible, faces, weights = surface
    per_vertex = np.full(len(data.reference.vertices), np.nan)
    per_vertex[data.scan_mask] = values
    pixels = (per_vertex[faces] * weights).sum(axis=1)
    colors = clay[visible]
    valid = np.isfinite(pixels)
    colors[valid] = plt.get_cmap(cmap)(norm(pixels[valid]))[:, :3]
    out = clay.copy()
    out[visible] = colors
    return out


class ScanView:
    """Use one crop, viewpoint, and error color scale for all conditions."""

    def __init__(self, data, size=230, yaw=20):
        self.data = data
        self.size = size
        self.center, self.axes = face_frame(data.landmarks, yaw)
        camera_xyz = self.camera_points(data.reference.vertices)
        self.view = View(np.ones(2), np.zeros(2)).framed(camera_xyz, size, margin=1.13)
        self.reference = self.surface(data.reference)
        self.error_norm = Normalize(0, 5, clip=True)
        self.change_norm = TwoSlopeNorm(0, vmin=-2, vmax=2)

    def camera_points(self, xyz):
        return (xyz - self.center) @ self.axes.T + [0, 0, 600]

    def surface(self, mesh):
        return _surface_view(mesh, self.camera_points(mesh.vertices),
                             mesh.vertex_normals @ self.axes.T, self.view, self.size)

    def figure(self, condition):
        rows = 2 if condition == "original" else 3
        fig, axes = plt.subplots(rows, 7, figsize=(15, 2.4 * rows + .6), layout="constrained")
        fig.suptitle(f"{CONDITIONS[condition]} · one NoW portrait and its reference scan", fontsize=16)
        for ax in axes.flat:
            ax.set_axis_off()
        with Image.open(self.data.photos["original"]) as original:
            box = np.asarray(load_face_box(self.data.case.box_path, original.size))
            original_size = original.size
        with Image.open(self.data.photos[condition]) as source:
            sx, sy = np.asarray(source.size) / original_size
            crop = source.crop(tuple(box * [sx, sy, sx, sy]))
            axes[0, 0].imshow(ImageOps.pad(crop, (self.size, self.size), color=(25, 29, 35)))
        axes[0, 0].set_title("Input crop")
        axes[0, 1].set_title("Actual NoW scan")
        axes[0, 1].imshow(self.reference[0])
        axes[1, 0].text(.01, .74, "Scan error\n(mm)", transform=axes[1, 0].transAxes, va="top")
        for index, model in enumerate(MODEL_IDS, start=2):
            key = model, condition
            if key not in self.data.metrics:
                axes[0, index].set_title(MODELS[model], fontsize=11)
                status, stage, _ = self.data.statuses[key]
                axes[0, index].text(.5, .5, f"{status}\n{stage or ''}", ha="center")
                continue
            score = self.data.metrics[key]
            axes[0, index].set_title(MODELS[model], fontsize=11)
            axes[0, index].imshow(self.surface(self.data.meshes[key])[0])
            axes[1, index].imshow(_scan_colors(
                self.data.errors[key], self.data, self.reference, "magma", self.error_norm))
            axes[1, index].set_title(f'Mean {score["mean_mm"]:.3f} mm', fontsize=10)
            if rows == 3 and (model, "original") in self.data.errors:
                difference = self.data.errors[key] - self.data.errors[model, "original"]
                axes[2, index].imshow(_scan_colors(
                    difference, self.data, self.reference, "coolwarm", self.change_norm))
                axes[2, index].set_title(f'Mean change {difference.mean():+.3f} mm', fontsize=10)
        self._colorbar(fig, axes[1, 1], "magma", self.error_norm, "0 to 5+ mm")
        if rows == 3:
            axes[2, 0].text(.01, .74, "Degraded minus\noriginal (mm)", transform=axes[2, 0].transAxes, va="top")
            self._colorbar(fig, axes[2, 1], "coolwarm", self.change_norm, "Blue improves · red worsens")
        return fig

    @staticmethod
    def _colorbar(fig, ax, cmap, norm, label):
        bar = ax.inset_axes([.03, .47, .94, .10])
        extent = "max" if cmap == "magma" else "both"
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                     orientation="horizontal", extend=extent)
        ax.text(.5, .29, label, transform=ax.transAxes, ha="center", fontsize=8)


def degradation_figure(data):
    """Separate resolution from blur and align the vertical error scale."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
    for ax, conditions in zip(axes, (("original", "face-128", "face-64", "face-32"),
                                     ("original", "blur-mild", "blur-heavy"))):
        for model in MODEL_IDS:
            absolute = data.metrics.get((model, "original"), {}).get("mean_mm", np.nan)
            ax.plot([CONDITIONS[key] for key in conditions],
                    [data.metrics.get((model, key), {}).get("mean_mm", np.nan) - absolute
                     for key in conditions],
                    marker="o", linewidth=1.8, label=MODELS[model])
        ax.axhline(0, color="#777", linewidth=1)
        ax.grid(axis="y", alpha=.25)
    axes[0].set(title="Resolution reduction", ylabel="Change in mean scan error (mm)")
    axes[1].set(title="Gaussian blur")
    axes[1].legend(fontsize=8, loc="upper left")
    return fig
