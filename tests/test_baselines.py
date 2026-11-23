"""Comparator artifact gates and failure receipt behavior."""

import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.geometry.baseline_pixel3dmm import _pipnet_nms, _tracking_flame_assets
from src.geometry.baseline_worker import checked_landmarks, export_neutral
from src.geometry.baselines import _wsl_path, run_baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    def test_native_embedding_batch_axis_and_face_reference(self):
        indices = np.full((1, 68), 3)
        weights = np.tile([0.25, 0.25, 0.5], (1, 68, 1))
        points = checked_landmarks(self.vertices, self.faces, indices, weights)
        np.testing.assert_allclose(points, np.tile([0.25, 0.25, 0.5], (68, 1)))
        indices[0, -1] = len(self.faces)
        with self.assertRaisesRegex(ValueError, "embedding"):
            checked_landmarks(self.vertices, self.faces, indices, weights)

    def test_neutral_export_retains_mesh_when_map_unavailable(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            export_neutral(run, self.vertices, self.faces)
            obj = (run / "neutral.obj").read_text(encoding="ascii")
            self.assertIn("f 2 3 4\n", obj)
            self.assertFalse((run / "neutral_landmarks.npy").exists())

    def test_export_rejects_degenerate_triangle_before_artifacts(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaisesRegex(ValueError, "Degenerate"):
                export_neutral(run, self.vertices, np.array([[0, 0, 2]]))
            self.assertFalse((run / "neutral.obj").exists())

    def test_failed_worker_preserves_attempt_status_and_logs(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            with patch("src.geometry.baselines._model_metadata", return_value={"device": "cpu"}):
                with patch("src.geometry.baselines._invoke", return_value=CompletedProcess([], 7)) as invoke:
                    result = run_baseline("deca", run / "missing.jpg", run, timeout=15)
            self.assertEqual(invoke.call_args.kwargs["timeout"], 15)
            self.assertEqual(result["fit_status"], "failed")
            self.assertEqual(result["error_stage"], "worker")
            self.assertIsNone(result["seconds"])
            self.assertEqual(json.loads((run / "baseline_result.json").read_text()), result)
            self.assertTrue((run / "comparator.stdout.log").exists())
            self.assertTrue((run / "comparator.stderr.log").exists())
            with self.assertRaises(FileExistsError):
                run_baseline("deca", run / "missing.jpg", run)

    def test_pixel_faceboxes_preserves_highest_scoring_detection(self):
        candidates = np.array([[0., 0., 10., 10., 0.7], [0., 0., 10., 10., 0.9],
                               [20., 20., 30., 30., 0.8]], dtype=np.float32)
        self.assertEqual(list(_pipnet_nms(candidates, 0.3)), [1, 2])

    def test_wsl_mount_preserves_spaces_and_rejects_unmapped_paths(self):
        self.assertEqual(_wsl_path(r"D:\A directory\input.png"), "/mnt/d/A directory/input.png")
        with self.assertRaises(ValueError):
            _wsl_path("relative.png")

    def test_pixel_repairs_only_a_run_local_copy_of_crlf_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mica = root / "upstream-mica"
            flame = mica / "data/FLAME2020"
            mask = Path("FLAME_masks/FLAME_masks.pkl")
            (flame / mask.parent).mkdir(parents=True)
            original = b"(dp0\r\nS'face'\r\np1\r\nI1\r\ns."
            (flame / mask).write_bytes(original)
            (flame / "generic_model.pkl").write_bytes(b"basis")
            (flame / "landmark_embedding.npy").write_bytes(b"embedding")
            with patch("src.geometry.baseline_pixel3dmm.MICA", mica):
                local = _tracking_flame_assets(root / "run")
            self.assertEqual((flame / mask).read_bytes(), original)
            self.assertEqual((local / "FLAME2020" / mask).read_bytes(),
                             original.replace(b"\r\n", b"\n"))
            self.assertEqual((local / "FLAME2020/generic_model.pkl").read_bytes(), b"basis")


if __name__ == "__main__":
    unittest.main()
