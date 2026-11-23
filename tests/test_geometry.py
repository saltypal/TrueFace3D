import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from src.evaluation import Evaluator
from src.geometry.camera import project_points
from src.rendering.preview import export_glb


class GeometryTests(unittest.TestCase):
    def test_export_uv_seam(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "face.glb"
            export_glb(vertices, [[0, 1, 2], [0, 2, 3]], output,
                       [[0, 0], [1, 0], [0, 1], [1, 1], [.5, .5]], [[0, 1, 2], [4, 2, 3]])
            data = output.read_bytes()
        self.assertEqual(struct.unpack_from("<4sII", data), (b"glTF", 2, len(data)))
        size, tag = struct.unpack_from("<I4s", data, 12)
        self.assertEqual(tag, b"JSON")
        contents = json.loads(data[20:20 + size])
        self.assertEqual(contents["accessors"][0]["count"], 5)
        self.assertEqual(contents["accessors"][3]["count"], 6)
        self.assertEqual(contents["meshes"][0]["primitives"][0]["material"], 0)
        self.assertEqual(contents["materials"][0]["pbrMetallicRoughness"]["metallicFactor"], 0)

    def test_camera_and_score(self):
        camera = dict(camera_pos=[0, 0, 0], camera_rot=np.eye(3),
                      image_width=200, image_height=100, camera_fov_y=90)
        xy, mask = project_points([[0, 0, 2], [1, -1, 2], [0, 0, -1]], camera)
        np.testing.assert_allclose(xy[:2], [[100, 50], [125, 25]])
        result = Evaluator().reprojection(xy, [[100, 50], [128, 29], [0, 0]], mask)
        self.assertEqual((result["unit"], result["valid_count"]), ("px", 2))
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["median"], 2.5)

    def test_invalid_indices(self):
        with self.assertRaises(ValueError):
            export_glb([[0, 0, 0]], [[0, 1, 0]], Path("unused.glb"))

