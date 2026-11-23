from dataclasses import asdict
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from src import benchmark
from src.data_now import NoWCase, real_world_cases
from src.geometry.oracle import MODES, oracle_settings, substitute_priors
from src.kaggle import paths, runner
from src.robustness.degrade import make_compound_variant, make_face_width_variant


def noisy_image(size=(400, 300)):
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


class VariantTests(unittest.TestCase):
    def test_existing_names_unchanged_and_new_added(self):
        self.assertEqual([v.name for v in benchmark.DEGRADED],
                         ["face-64", "blur-mild", "blur-heavy", "face-128", "face-32"])
        self.assertEqual([v.name for v in benchmark.KAGGLE_VARIANTS],
                         ["original", "face-64", "face-32", "face-16", "compound"])
        self.assertEqual(benchmark.FACE_16.width_px, 16)

    def test_face_16_width(self):
        image, meta = make_face_width_variant(noisy_image(), (100, 50, 300, 250), 16)
        self.assertAlmostEqual(meta["achieved_face_width_px"], 16.0, places=3)
        self.assertEqual(image.size, (32, 24))

    def test_compound_is_deterministic_and_sized(self):
        image, box = noisy_image((920, 600)), (100, 50, 560, 500)  # 460 px face
        a, meta = make_compound_variant(image, box)
        b, _ = make_compound_variant(image, box)
        c, _ = make_compound_variant(image, box, seed=1)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertAlmostEqual(meta["achieved_face_width_px"], 48.0, places=3)
        self.assertAlmostEqual(meta["blur_sigma_px"], 1.5)
        self.assertEqual(a.size, (96, 63))

    def test_blur_scales_with_box_width(self):
        _, meta = make_compound_variant(noisy_image(), (0, 0, 230, 200))
        self.assertAlmostEqual(meta["blur_sigma_px"], 0.75)

    def test_identity_keeps_old_variant_shape(self):
        old = {k: v for k, v in asdict(benchmark.DEGRADED[0]).items() if not (k == "compound" and not v)}
        self.assertNotIn("compound", old)


class RealWorldTests(unittest.TestCase):
    def test_selects_three_categories(self):
        p = Path("x")
        cases = [NoWCase(f"S/{c}/a.jpg", "S", c, p, p, p, None) for c in
                 ("multiview_neutral", "selfie", "multiview_expressions", "multiview_occlusions", "other")]
        picked = real_world_cases(cases)
        self.assertEqual({c.category for c in picked}, {"selfie", "multiview_expressions", "multiview_occlusions"})


class FakeResult:
    def __init__(self):
        self.vertex_coord = np.zeros((4, 2), np.float32)
        self.vertex_coord_log_var = np.zeros((4, 2), np.float32)
        self.vertex_depth = np.zeros((4, 1), np.float32)
        self.vertex_depth_log_var = np.zeros((4, 1), np.float32)


class OracleTests(unittest.TestCase):
    def clean(self):
        return {k: np.ones(getattr(FakeResult(), k).shape, np.float32) for k in MODES["both"]}

    def test_modes(self):
        for mode, names in MODES.items():
            result = FakeResult()
            replaced = substitute_priors(result, self.clean(), mode)
            self.assertEqual(tuple(replaced), names)
            for key in MODES["both"]:
                self.assertEqual(getattr(result, key).max(), 1.0 if key in names else 0.0)

    def test_shape_mismatch_and_argv(self):
        bad = {k: np.ones((3, 2), np.float32) for k in MODES["both"]}
        with self.assertRaises(ValueError):
            substitute_priors(FakeResult(), bad, "coord")
        argv = ["w", "--priors-in", "a.npz", "--priors-mode", "depth", "--x"]
        self.assertEqual(oracle_settings(argv), ("a.npz", "depth"))
        self.assertEqual(argv, ["w", "--x"])

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRUEFACE3D_PRIORS_IN", None)
            self.assertIsNone(oracle_settings(["w"])[0])


class RuntimeTests(unittest.TestCase):
    def test_split_jobs_round_robin(self):
        self.assertEqual(runner.split_jobs(list(range(5)), 2), [[0, 2, 4], [1, 3]])

    def test_env_paths(self):
        with mock.patch.dict(os.environ, {"TRUEFACE3D_DATA": "/d", "TRUEFACE3D_WEIGHTS": "/w",
                                          "TRUEFACE3D_THIRD_PARTY": "/t"}):
            self.assertEqual((paths.data_root(), paths.weights_root(), paths.third_party_root()),
                             (Path("/d"), Path("/w"), Path("/t")))

    def test_local_defaults(self):
        if os.name == "nt":
            self.assertEqual(paths.data_root(), paths.PROJECT / "Data")


if __name__ == "__main__":
    unittest.main()
