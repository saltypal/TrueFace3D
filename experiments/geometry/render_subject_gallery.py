"""Render all retained meshes for one NoW photograph and its five variants."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from src.benchmark import MODEL_IDS, file_digest, result_records, write_table
from src.data_now import installed_cases, load_face_box, pilot_cases
from src.evaluation import fit_similarity, load_surface, transform_points
from src.rendering.gallery import View, render_pair


ROOT = Path("experiments/geometry/runs")
OUTPUT = ROOT / "chapter0-one-photo-gallery"
CONDITIONS = {
    "original": "Original",
    "face-128": "128 px face width",
    "face-64": "64 px face width",
    "face-32": "32 px face width",
    "blur-mild": "Mild blur",
    "blur-heavy": "Heavy blur",
}
DISPLAY_NAMES = {
    "vits": "RealDenseFace ViT-S", "vitb": "RealDenseFace ViT-B",
    "deca": "DECA coarse", "pixel3dmm": "Pixel3DMM", "mica": "MICA",
}


def geometry(record):
    folder = Path(record.case_dir)
    if (folder / "fit/neutral.obj").is_file():
        folder /= "fit"
    mesh = load_surface(folder / "neutral.obj")
    landmarks = np.load(folder / "neutral_landmarks.npy", allow_pickle=False)
    return mesh, landmarks


def render_condition(case, condition, title, records):
    source_path = case.image_path
    if condition != "original":
        source_path, = (ROOT / "chapter1-degraded-01/inputs").glob(f"*/{condition}.png")
    digest = file_digest(source_path)
    if any(row.input_sha256 != digest or row.fit_status != "success" for row in records.values()):
        raise ValueError(f"Invalid saved fit or source fingerprint for {condition}")

    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    source_rgb = np.asarray(image)
    with Image.open(case.image_path) as original:
        box = np.asarray(load_face_box(case.box_path, original.size))
        sx, sy = np.asarray(image.size) / original.size
    box = box * [sx, sy, sx, sy]
    box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height)
    thumb = ImageOps.pad(image.crop(tuple(box)), (320, 320), color=(25, 29, 35))

    reference_folder = Path(records["vits"].case_dir) / "fit"
    scene = json.loads((reference_folder / "scene.json").read_text(encoding="utf-8"))
    intr = np.asarray(scene["intrinsics"])
    pose = np.asarray(scene["world_to_camera"])
    camera = View(intr.diagonal()[:2], intr[:2, 2])
    if (scene["camera"]["image_width"], scene["camera"]["image_height"]) != image.size:
        raise ValueError("Source image and diagnostic camera dimensions disagree")
    posed_landmarks = np.load(reference_folder / "posed_landmarks.npy", allow_pickle=False)
    with np.load(reference_folder / "face.npz", allow_pickle=False) as face:
        reference = transform_points(face["vertices"], pose)
    landmarks_camera = transform_points(posed_landmarks, pose)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), layout="constrained")
    fig.suptitle(f"NoW IMG_0041  |  {title}", fontsize=18)
    for ax in axes.flat:
        ax.set_axis_off()
    axes[0, 0].imshow(thumb)
    axes[0, 0].set_title("Input crop")
    axes[1, 0].text(
        .03, .82, "Top  ·  geometry only\n\nBottom  ·  photo color\n\n"
        "Gray areas have no\nvisible source color.",
        transform=axes[1, 0].transAxes, va="top", fontsize=12,
    )
    details = []
    for column, model in enumerate(MODEL_IDS, 1):
        record = records[model]
        mesh, landmarks = geometry(record)
        alignment = pose @ fit_similarity(landmarks, posed_landmarks)
        vertices = transform_points(mesh.vertices, alignment)
        normals = np.asarray(mesh.vertex_normals) @ alignment[:3, :3].T
        clay, skin, visible = render_pair(
            source_rgb, vertices, mesh.faces, normals, camera, reference, landmarks_camera)
        axes[0, column].imshow(clay)
        axes[0, column].set_title(DISPLAY_NAMES[model], fontsize=11)
        axes[1, column].imshow(skin)
        details.append({
            "condition": condition, "model": model, "fit_id": record.attempt_id,
            "source_sha256": digest, "visible_color_fraction": visible,
            "fit_directory": record.case_dir,
        })
        print(f"{condition} {model}: {visible:.1%} observed color", flush=True)
    target = OUTPUT / f"{condition}.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return details


def main():
    case, = pilot_cases(installed_cases(), count=1)
    rows = result_records(ROOT / "chapter1-original-scan-local-01")
    rows += result_records(ROOT / "chapter1-degraded-scan-local-01")
    rows = [row for row in rows if row.image_key == case.image_key]
    keyed = {(row.variant_id, row.model_id): row for row in rows}
    if len(rows) != 30 or len(keyed) != 30:
        raise ValueError("Expected 30 unique saved fits for the teaching portrait")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details = []
    for condition, title in CONDITIONS.items():
        details.extend(render_condition(
            case, condition, title, {model: keyed[condition, model] for model in MODEL_IDS}))
    write_table(OUTPUT / "gallery-manifest.csv", details)


if __name__ == "__main__":
    main()
