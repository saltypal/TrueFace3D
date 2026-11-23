import tempfile
import unittest

import numpy as np

from src.appearance import metrics
from src.appearance.backends import NAMES, load
from src.appearance.projection import project_to_uv, render_textured


def quad():
    camera = {"image_width": 64, "image_height": 64, "camera_fov_y": 60.0,
              "camera_rot": np.eye(3).tolist(), "camera_pos": [0, 0, 0], "principal_point": [31.5, 31.5]}
    return {"vertices": np.array([[-.5, -.5, 2], [.5, -.5, 2], [.5, .5, 2], [-.5, .5, 2]], float),
            "faces": np.array([[0, 2, 1], [0, 3, 2]]), "uvs": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
            "uv_faces": np.array([[0, 2, 1], [0, 3, 2]]), "camera": camera}


class ProjectionTests(unittest.TestCase):
    def test_quad_colours_land_in_uv(self):
        photo = np.zeros((64, 64, 3), np.uint8)
        photo[:, :32] = (255, 0, 0)
        photo[:, 32:] = (0, 0, 255)
        uv, vis, info = project_to_uv(photo, quad(), tex_size=32, depth_size=128)
        self.assertGreater(info["uv_coverage"], 0.9)
        self.assertGreater(uv[16, 3, 0], 200)
        self.assertGreater(uv[16, 28, 2], 200)
        render, mask, valid = render_textured(uv, vis, quad(), (64, 64), max_side=64)
        self.assertGreater(mask.sum(), 300)
        self.assertGreater(render[32, 20, 0], 200)
        self.assertGreater(render[32, 44, 2], 200)

    def test_back_facing_texels_are_not_visible(self):
        mesh = quad()
        mesh["faces"] = np.array([[0, 1, 2], [0, 2, 3]])
        _, _, info = project_to_uv(np.zeros((64, 64, 3), np.uint8), mesh, tex_size=32, depth_size=64)
        self.assertEqual(info["visible_texels"], 0)


class MetricTests(unittest.TestCase):
    def test_identity_and_difference(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), bool)
        self.assertEqual(metrics.masked_psnr(a, a, mask), float("inf"))
        b = np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8)
        self.assertGreater(metrics.masked_psnr(a, b, mask), 20)
        self.assertLess(metrics.masked_psnr(a, b, mask), 30)
        self.assertEqual(metrics.all_metrics(a, a, mask)["mask_pixels"], 1024)

    def test_mask_excludes_pixels(self):
        a = np.zeros((8, 8, 3), np.uint8)
        b = a.copy()
        b[:4] = 255
        mask = np.zeros((8, 8), bool)
        mask[4:] = True
        self.assertEqual(metrics.masked_psnr(a, b, mask), float("inf"))

    def test_pair_metrics_common_mask(self):
        uv = np.zeros((8, 8, 3), np.uint8)
        va, vb = np.zeros((8, 8), bool), np.zeros((8, 8), bool)
        va[:4], vb[2:6] = True, True
        out = metrics.pair_metrics(uv, va, uv, vb, uv, uv, va)
        self.assertEqual(out["uv"]["mask_pixels"], 16)


class BackendTests(unittest.TestCase):
    def test_interface_and_graceful_not_run(self):
        for name in NAMES:
            ok, reason = load(name).available()
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(reason, str)
        self.assertNotIn("hrn", NAMES)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("opendelight", "trust"):
                result = load(name).run(__file__, tmp)
                self.assertIn(result["status"], {"not_run", "failed"})


if __name__ == "__main__":
    unittest.main()
