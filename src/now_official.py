"""Bridge to the pinned NoW evaluator in its isolated environment."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np


REVISION = "7bd149850a5cad431a488857f1dc774e1eb533c9"


class OfficialNoW:
    def __init__(self, python, checkout, scratch, timeout=180):
        self.python = str(python)
        self.checkout = Path(checkout).resolve()
        self.scratch = Path(scratch).resolve()
        self.timeout = timeout
        self.wsl_distro = os.environ.get("TRUEFACE3D_NOW_WSL_DISTRO") if os.name == "nt" else None
        actual = subprocess.check_output(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != REVISION:
            raise ValueError(f"Expected official NoW revision {REVISION}, got {actual}")

    def __call__(self, scan, scan_lmk, pred_mesh, pred_lmk, protocol, unit_to_mm):
        metrical = protocol == "official_now_metrical"
        unit = {1.0: "mm", 10.0: "cm", 1000.0: "m"}.get(unit_to_mm)
        if metrical and unit is None:
            raise ValueError("Official metrical scoring requires a verified m, cm or mm mesh unit")
        script = Path(__file__).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.scratch) as folder:
            output = Path(folder) / "distances.npy"
            paths = [script, self.checkout, Path(scan).resolve(), Path(scan_lmk).resolve(),
                     Path(pred_mesh).resolve(), Path(pred_lmk).resolve(), output]
            if self.wsl_distro:
                from .geometry.baselines import _wsl_path

                command = ["wsl.exe", "-d", self.wsl_distro, "--exec", self.python,
                           *(_wsl_path(path) for path in paths)]
            else:
                command = [self.python, *map(str, paths)]
            command.extend([str(int(metrical)), unit or "mm"])
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode:
                raise RuntimeError(f"Official NoW worker failed: {result.stderr.strip()[-1600:]}")
            from .evaluation import distance_summary
            return distance_summary(np.load(output, allow_pickle=False).reshape(-1), protocol)


def worker(checkout, scan, scan_lmk, mesh, landmarks, output, metrical, units):
    # Runs as a script under the evaluator's Python, independent of the notebook venv.
    sys.path.insert(0, checkout)
    from psbody.mesh import Mesh
    from compute_error import compute_error_metric

    points = np.load(landmarks, allow_pickle=False) if landmarks.endswith(".npy") else np.loadtxt(landmarks)
    if points.shape != (7, 3) or not np.isfinite(points).all():
        raise ValueError("Predicted NoW landmarks must be seven ordered finite 3D points")
    scores = compute_error_metric(scan, scan_lmk, Mesh(filename=mesh), points,
                                  predicted_mesh_unit=units, metrical_eval=metrical == "1")
    np.save(output, scores, allow_pickle=False)


if __name__ == "__main__":
    worker(*sys.argv[1:])
