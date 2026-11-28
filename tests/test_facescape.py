"""FaceScape-C geometry, crop and builder tests. Data-dependent tests skip when the zips are absent."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.facescape import constants as C
from src.facescape.crop import crop_photo
from src.facescape.geometry import (camera_center_tu, project, select_view, transform, tu_to_camera,
                                    view_angles)
from src.facescape.mesh import read_obj


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_tu_to_camera_matches_two_step_chain():
    rng = np.random.default_rng(0)
    x_tu = rng.normal(size=(50, 3)) * 80
    s, rt_tu = 123.4, np.hstack([_rot("y", 20) @ _rot("x", -5), [[3.0], [-7.0], [11.0]]])
    rt_c = np.hstack([_rot("x", 10), [[0.1], [0.2], [9.0]]])
    x_mv = (rt_tu[:, :3].T @ (x_tu - rt_tu[:, 3]).T).T / s
    expected_mm = ((rt_c[:, :3] @ x_mv.T).T + rt_c[:, 3]) * s
    r, t = tu_to_camera(rt_c, s, rt_tu)
    np.testing.assert_allclose(transform(x_tu, r, t), expected_mm, atol=1e-8)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_view_angles_sign_convention():
    # camera on the subject's left (+X) looking at the face -> positive yaw
    yaw, pitch, dist = view_angles(np.array([1000.0, 0.0, 1000.0]))
    assert yaw == pytest.approx(45) and pitch == pytest.approx(0) and dist == pytest.approx(1414.2, abs=0.1)
    r, t = np.eye(3), np.array([0.0, 0.0, 1500.0])
    np.testing.assert_allclose(camera_center_tu(r, t), [0, 0, -1500])


def test_select_view_prefers_valid_level_close_camera():
    rows = [{"view": 0, "valid": False, "yaw": 44, "pitch": 0, "width": 2896, "height": 4344},
            {"view": 1, "valid": True, "yaw": 40, "pitch": 30, "width": 2896, "height": 4344},
            {"view": 2, "valid": True, "yaw": 50, "pitch": 3, "width": 1728, "height": 2592},
            {"view": 3, "valid": True, "yaw": 90, "pitch": 0, "width": 2896, "height": 4344}]
    assert select_view(rows, 45)["view"] == 2
    assert select_view(rows, -45) is None


def test_read_obj_keeps_topology_and_triangulates_quads():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n"
    m = read_obj(obj)
    assert m.vertices.shape == (4, 3) and m.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert m.face_uvs.tolist() == m.faces.tolist() and m.uvs.shape == (4, 2)


def test_crop_camera_is_consistent_with_source_projection():
    k = np.array([[3000.0, 0, 1448], [0, 3000.0, 2172], [0, 0, 1]])
    photo = np.zeros((4344, 2896, 3), np.uint8)
    r, t = np.eye(3), np.array([0.0, 0.0, 1600.0])
    centre, width = np.array([0.0, 0.0, 0.0]), 140.0
    crop = crop_photo(photo, k, np.zeros(5), r, t, centre, width)
    pts = np.array([[0, 0, 0], [50, -30, 10], [-60, 40, -5]], dtype=float)
    src = project(transform(pts, r, t), k)
    x0, y0, side = crop.box
    expected = (src - [x0, y0] + 0.5) * (C.CROP_PX / side) - 0.5
    np.testing.assert_allclose(project(transform(pts, r, t), crop.k), expected, atol=1e-6)
    assert crop.image.shape == (C.CROP_PX, C.CROP_PX, 3)
    assert crop.face_px_source == pytest.approx(3000 * 140 / 1600)
    assert not crop.padded


ROOT = Path("/mnt/DataBunker/facescape_c")


@pytest.mark.skipif(not (ROOT / "manifest_clean.csv").exists(), reason="FaceScape-C not built here")
def test_built_dataset_landmarks_project_inside_crops():
    from src.facescape.dataset import FaceScapeC

    ds = FaceScapeC(ROOT)
    lm = ds.topology["landmarks68"]
    cases = ds.cases(split="test")[:9]
    assert cases, "no test cases"
    for case in cases:
        uv = case.project(case.gt_vertices_camera())[lm]
        assert (uv > 0).all() and (uv < C.CROP_PX).all(), case.image_id
        assert abs(case.camera["yaw"] - C.VIEWS[case.view]) <= C.VIEW_MAX_YAW_ERROR


def test_crop_fits_inside_narrow_photo_without_padding():
    # close face in a portrait frame: 1.6 x jaw width would exceed the photo width
    k = np.array([[4000.0, 0, 1448], [0, 4000.0, 2172], [0, 0, 1]])
    photo = np.zeros((4344, 2896, 3), np.uint8)
    r, t = np.eye(3), np.array([0.0, 0.0, 350.0])
    crop = crop_photo(photo, k, np.zeros(5), r, t, np.zeros(3), 160.0)
    assert crop.face_px_source * C.CROP_PX / C.FACE_PX > 2896
    assert crop.box[2] == 2896 and not crop.padded
    assert crop.face_px == pytest.approx(crop.face_px_source * C.CROP_PX / 2896)


def test_degraded_camera_uses_pixel_centre_convention(tmp_path):
    from src.facescape.dataset import Case

    k = np.array([[1000.0, 0, 383.5], [0, 1000.0, 383.5], [0, 0, 1]])
    (tmp_path / "cam.json").write_text(json.dumps({"K": k.tolist(), "R": np.eye(3).tolist(), "t": [0, 0, 1000]}))
    s = 96 / 768
    meta = {"clean_to_degraded": [[s, 0, 0], [0, s, 0], [0, 0, 1]]}
    case = Case(tmp_path, {"camera": "cam.json", "meta": json.dumps(meta)})
    p = np.array([[10.0, -20.0, 1000.0]])
    u_clean = project(p, k)
    np.testing.assert_allclose(case.project(p), (u_clean + 0.5) * s - 0.5, atol=1e-9)
