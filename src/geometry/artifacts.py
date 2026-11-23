"""Assemble geometry before launching the separate preview process."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from ..rendering.preview import export_glb
from .camera import camera_matrices, crop_transform, project_points


def save_bundle(output, input_path=None):
    start = time.perf_counter()
    with np.load(output / "mesh_native.npz", allow_pickle=False) as mesh:
        geometry = {name: mesh[name] for name in mesh.files}
    with np.load(output / "native_fit.npz", allow_pickle=False) as fit:
        camera = {name: fit[name].tolist() for name in (
            "image_width", "image_height", "camera_fov_y", "camera_rot", "camera_pos"
        )}
        camera["principal_point"] = [(camera[key] - 1) / 2 for key in ("image_width", "image_height")]
        count = int(fit["num_expressions"])
        x = fit["x"]
        if x.shape != (count + 18,) or not np.isfinite(x).all():
            raise ValueError("Unexpected packed FLAME parameters")
        geometry.update(identity=fit["identity"], expression=x[:count],
                        pose=x[count:count + 15], translation=x[count + 15:])

    files = {"face": "face.npz", "preview": "face.glb", "priors": "priors.npz",
             "fit": "native_fit.npz", "projection": "projection.npz"}
    if "neutral_vertices" in geometry:
        with (output / "neutral.obj").open("w", encoding="utf-8") as obj:
            np.savetxt(obj, geometry["neutral_vertices"], fmt="v %.9g %.9g %.9g")
            np.savetxt(obj, geometry["faces"] + 1, fmt="f %d %d %d")
        files["neutral"] = "neutral.obj"
    export_glb(geometry["vertices"], geometry["faces"], output / "face.glb",
               geometry["uvs"], geometry["uv_faces"])
    np.savez_compressed(output / "face.npz", **geometry)

    with np.load(output / "priors.npz", allow_pickle=False) as priors:
        transform = crop_transform(priors["crop_bbox"], priors["crop_size_hw"])
    intrinsics, extrinsics = camera_matrices(camera)
    projected, visible = project_points(geometry["vertices"], camera)
    np.savez_compressed(output / "projection.npz", xy=projected, camera_clip_visible=visible)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1, "camera": camera, "calibration": "estimated",
        "intrinsics": intrinsics.tolist(), "world_to_camera": extrinsics.tolist(),
        "input_to_crop": transform.tolist(), "crop_to_input": np.linalg.inv(transform).tolist(),
        "coordinate_system": "native FLAME world", "metric_scale_verified": False,
        "assets": run["assets"], "upstream_revision": run["revision"],
        "files": files, "landmarks": "unavailable: no verified topology-specific seven-point embedding",
        "host_export_seconds": time.perf_counter() - start,
    }
    publish = output / "scene.json"
    _write_manifest(publish, manifest)

    panels = output / "native_panels.png"
    if panels.is_file():
        with Image.open(panels) as source:
            w, h = camera["image_width"], camera["image_height"]
            if source.size != (2 * w, 2 * h):
                raise ValueError("Unexpected native panel dimensions")
            source.crop((0, h, w, 2 * h)).save(output / "overlay.png")
            source.crop((w, h, 2 * w, 2 * h)).save(output / "render.png")
    else:
        command = run["command"]
        image = Path(input_path) if input_path else Path(command[command.index("--input") + 1])
        upstream = Path(run["assets"]["flame"]["path"]).parents[2]
        started = time.perf_counter()
        with (output / "preview.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run([command[0], command[1], "--preview", str(output), str(image),
                                        command[command.index("--device") + 1]],
                                       cwd=upstream, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest["preview_status"] = "complete" if completed.returncode == 0 else "failed; see preview.log"
        manifest["preview_worker_wall_seconds"] = time.perf_counter() - started
    for name in ("overlay", "render", "side"):
        if (output / f"{name}.png").is_file():
            files[name] = f"{name}.png"
    if (output / "landmark_provenance.json").is_file():
        for name in ("neutral_landmarks", "posed_landmarks"):
            files[name] = f"{name}.npy"
        manifest["landmarks"] = json.loads((output / "landmark_provenance.json").read_text(encoding="utf-8"))
    _write_manifest(publish, manifest)


def _write_manifest(path, manifest):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
