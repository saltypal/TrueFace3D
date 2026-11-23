import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.benchmark import ModelSpec, ORIGINAL, Variant, rescore_run, run_matrix, pair_variants
from src.data_now import NoWCase, load_cases, load_scan_landmarks, pilot_cases
from src.evaluation import Evaluator, LOCAL_METRICAL, closest_surface, fit_similarity, load_surface, transform_points
from src.robustness.degrade import make_face_width_variant


class DatasetTests(unittest.TestCase):
    def test_scan_mapping_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            key = "FaMoS_01/multiview_neutral/IMG.jpg"
            for name in ["photo/" + key, "box/" + key.replace(".jpg", ".npy"), "scan/FaMoS_01/head.obj"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            index = root / "index.txt"
            index.write_text(key + "\n", encoding="utf-8")
            case = pilot_cases(load_cases(root / "photo", root / "box", index, root / "scan"), 1)[0]
            self.assertEqual(case.subject_id, "FaMoS_01")
            self.assertIsNone(case.scan_landmark_path)
            index.write_text("FaMoS_01/../IMG.jpg\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(root / "photo", root / "box", index, root / "scan")

    def test_face_width_preserves_transform(self):
        image = Image.new("RGB", (3024, 4032), (125, 80, 30))
        variant, metadata = make_face_width_variant(image, (100, 90, 560, 750), 64)
        self.assertEqual(variant.size, (421, 561))
        self.assertAlmostEqual(metadata["achieved_face_width_px"], 64.041005, places=5)
        np.testing.assert_allclose(np.array(metadata["original_to_variant"])
                                   @ np.array(metadata["variant_to_original"]), np.eye(3))


class SurfaceTests(unittest.TestCase):
    def test_similarity_and_fixed_scale(self):
        src = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 3.]])
        target = 2 * src @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]) + 5
        recovered = fit_similarity(src, target)
        np.testing.assert_allclose(transform_points(src, recovered), target)
        fixed = fit_similarity(src, target, scale=False)
        self.assertAlmostEqual(np.linalg.det(fixed[:3, :3]), 1)
        self.assertGreater(np.mean(np.abs(transform_points(src, fixed) - target)), 0.4)

    def test_nearest_triangle_and_registered_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root / "patch.obj"
            mesh.write_text("v -4 -3 0\nv 4 -3 0\nv 0 5 0\nf 1 2 3\n", encoding="ascii")
            _, distances, _ = closest_surface(load_surface(mesh), np.array([[0., 0., 0.], [0., 0., 2.]]))
            np.testing.assert_allclose(distances, [0, 2])
            points = np.array([[-4, -3, 0], [-1, 0, 0], [1, 0, 0], [4, -3, 0],
                               [0, -1, 0], [-2, -2, 0], [2, -2, 0]])
            lmk = root / "scan.pp"
            entries = "".join(f'<point x="{x}" y="{y}" z="{z}" active="1"/>' for x,y,z in points)
            lmk.write_text(f"<PickedPoints>{entries}</PickedPoints>", encoding="ascii")
            pred = root / "pred.npy"
            np.save(pred, points)
            np.testing.assert_array_equal(load_scan_landmarks(lmk), points)
            score = Evaluator().score_now(mesh, lmk, mesh, pred, LOCAL_METRICAL, unit_to_mm=1)
            self.assertLess(score.mean_mm, 1e-10)
            self.assertEqual(score.reference_count, 3)


class MatrixTests(unittest.TestCase):
    def test_replay_attempts_preserve_failure_and_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, box, scan = [root / name for name in ("image.jpg", "box.npy", "scan.obj")]
            Image.new("RGB", (40, 40)).save(image)
            np.save(box, dict(left=2, top=2, right=38, bottom=38))
            scan.write_text("synthetic scanning fixture", encoding="ascii")
            case = NoWCase("FaMoS_01/multiview_neutral/image.jpg", "FaMoS_01", "multiview_neutral",
                           image, box, scan, None)
            specs = [ModelSpec(name, "rev", "asset", "config") for name in ("vits", "deca")]
            calls = []

            def fit(spec, image_path, run, timeout):
                calls.append((spec.model_id, image_path.suffix))
                if spec.model_id == "vits":
                    (run / "neutral.obj").touch()
                    return {"fit_status": "success"}
                return {"fit_status": "failed"}

            original = run_matrix([case], specs, [ORIGINAL], root / "original", runner=fit)
            replay = run_matrix([case], specs, [ORIGINAL], root / "original", runner=fit)
            degraded = run_matrix([case], specs, [Variant("face-16", width_px=16)], root / "degraded", runner=fit)
            self.assertEqual(calls.count(("vits", ".jpg")), 1)
            self.assertEqual(calls.count(("deca", ".jpg")), 2)
            self.assertEqual(original[0].attempt_id, replay[0].attempt_id)
            self.assertEqual(original[0].score_status, "missing_scan_landmarks")
            self.assertIsNone(pair_variants(degraded, original)[0]["delta_mm"])
            Image.new("RGB", (40, 40), "white").save(image)
            changed = run_matrix([case], specs, [Variant("face-16", width_px=16)], root / "changed", runner=fit)
            self.assertEqual(pair_variants(changed, original)[0]["pair_status"], "source_request_mismatch")

    def test_annotations_can_rescore_saved_fit_without_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, box, scan = [root / name for name in ("image.jpg", "box.npy", "scan.obj")]
            Image.new("RGB", (40, 40)).save(image)
            np.save(box, dict(left=2, top=2, right=38, bottom=38))
            scan.write_text("v -4 -3 0\nv 4 -3 0\nv 0 5 0\nf 1 2 3\n", encoding="ascii")
            photo_key = "FaMoS_01/multiview_neutral/image.jpg"
            case = NoWCase(photo_key, "FaMoS_01", "multiview_neutral", image, box, scan, None)
            points = np.array([[-4, -3, 0], [-1, 0, 0], [1, 0, 0], [4, -3, 0],
                               [0, -1, 0], [-2, -2, 0], [2, -2, 0]])

            def fit(spec, image_path, run, timeout):
                (run / "neutral.obj").write_bytes(scan.read_bytes())
                np.save(run / "neutral_landmarks.npy", points)
                return {"fit_status": "success", "mesh_unit_to_mm": 1}

            specs = [ModelSpec("fixture", "rev", "asset", "config")]
            original = run_matrix([case], specs, [ORIGINAL], root / "original", runner=fit)[0]
            lmk = root / "scan.pp"
            entries = "".join(f'<point x="{x}" y="{y}" z="{z}" active="1"/>' for x, y, z in points)
            lmk.write_text(f"<PickedPoints>{entries}</PickedPoints>", encoding="ascii")
            with_reference = NoWCase(photo_key, "FaMoS_01", "multiview_neutral", image, box, scan, lmk)
            scored = rescore_run(root / "original", root / "new-scores", [with_reference],
                                 protocol=LOCAL_METRICAL)[0]
            self.assertEqual(original.score_status, "missing_scan_landmarks")
            self.assertEqual(scored.score_status, "success")
            self.assertEqual(original.attempt_id, scored.attempt_id)
            self.assertEqual(original.case_dir, scored.case_dir)
            self.assertNotEqual(original.reference_sha256, scored.reference_sha256)
