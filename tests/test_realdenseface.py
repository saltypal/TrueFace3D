from contextlib import nullcontext
import unittest

import numpy as np

from src.geometry.camera import camera_matrices, crop_transform, project_points
from src.geometry.landmarks import NOW_LANDMARK_INDICES, transfer_now_landmarks
from src.geometry.plot_priors import sample_uv
from src.geometry.worker import prior_autocast


class RealDenseFaceTests(unittest.TestCase):
    def test_default_autocast_uses_original_context(self):
        calls = []
        sentinel = object()

        def original(**options):
            calls.append(options)
            return sentinel

        options = dict(device_type="cuda", dtype="torch.bfloat16", enabled=True)
        self.assertIs(prior_autocast(original, precision="default", **options), sentinel)
        self.assertIsInstance(prior_autocast(original, precision="fp32-compat", **options), nullcontext)
        self.assertIs(prior_autocast(original, precision="fp32-compat", **(options | {"device_type": "cpu"})), sentinel)
        self.assertIs(prior_autocast(original, precision="fp32-compat", **(options | {"dtype": "torch.float16"})), sentinel)
        self.assertEqual(len(calls), 3)

    def test_native_camera_explicit_pixel_center(self):
        camera = dict(camera_pos=[1, 0, 0], camera_rot=np.eye(3),
                      image_width=200, image_height=100, camera_fov_y=90,
                      principal_point=[99.5, 49.5])
        K, extrinsic = camera_matrices(camera)
        np.testing.assert_allclose(K, [[50, 0, 99.5], [0, 50, 49.5], [0, 0, 1]])
        np.testing.assert_allclose(extrinsic[:3, 3], [-1, 0, 0])
        xy, valid = project_points([[1, 0, 2], [2, -1, 2], [1, 0, -1]], camera)
        np.testing.assert_allclose(xy[:2], [[99.5, 49.5], [124.5, 24.5]])
        np.testing.assert_array_equal(valid, [True, True, False])

    def test_crop_centers_and_bilinear_weights(self):
        T = crop_transform([10, 20, 14, 22], [4, 4])
        np.testing.assert_allclose(T @ [10, 20, 1], [0, .5, 1])
        np.testing.assert_allclose(np.linalg.inv(T) @ [0, .5, 1], [10, 20, 1])
        image = np.array([[2, 6], [10, 14]])
        np.testing.assert_allclose(sample_uv(image, [[.25, .25], [.5, .5], [0, 0]]), [2, 8, .5])

    def test_landmark_transfer_requires_exact_corner_order(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 4., 0.]])
        faces = np.array([[0, 1, 2]])
        weights = np.tile([1., 0., 0.], (1, 68, 1))
        weights[0, NOW_LANDMARK_INDICES[4]] = [.25, .25, .5]
        mapping = {"full_lmk_faces_idx": np.zeros((1, 68), dtype=int), "full_lmk_bary_coords": weights}
        points = transfer_now_landmarks(vertices, faces, faces.copy(), mapping)
        self.assertEqual(points.shape, (7, 3))
        np.testing.assert_allclose(points[4], [.5, 2., 0.])
        with self.assertRaisesRegex(ValueError, "face tables differ"):
            transfer_now_landmarks(vertices, faces, faces[:, [0, 2, 1]], mapping)


if __name__ == "__main__":
    unittest.main()
