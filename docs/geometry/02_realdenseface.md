# RealDenseFace Chapter 1 lab

## Run on the MX330

Use the Windows project venv for notebooks and the existing WSL worker for
PyTorch CUDA, the FLAME solver and rasterization. Run these from the project
root in PowerShell.

```powershell
.\.venv\Scripts\python.exe -m src.cli audit
.\.venv\Scripts\python.exe -m src.cli inventory-now
wsl -d Ubuntu-26.04 -- /home/Satya/.venvs/trueface3d-vit/bin/python /mnt/d/Bunker/BaseCamp/ComputerVision/TrueFace3D/experiments/vits-local-cuda/runs/verify_extensions.py
```

The worker uses FP32 compatibility on Pascal. The override applies to the
ViT network's released BF16 autocast region. The FaceBoxes compile option
also follows `--no_compile` on Pascal. The 512-square network input and
native fitting settings are preserved.

```powershell
$tfRoot = '/mnt/d/Bunker/BaseCamp/ComputerVision/TrueFace3D'
$tfWorker = '/home/Satya/.venvs/trueface3d-vit/bin/python'
wsl -d Ubuntu-26.04 --cd $tfRoot -- $tfWorker -m src.cli reconstruct --python $tfWorker --device cuda --precision fp32-compat --input "$tfRoot/manual_tests/photos/official-example-01.jpg" --output "$tfRoot/experiments/vits-local-cuda/runs/my-vits-01"
```

Use a fresh output directory for each fit. Open the GLB in the local viewer
and compare it with the saved overlay. The fit and export run separately from
the preview, so a visualization failure can leave valid geometry to inspect.

```powershell
.\.venv\Scripts\python.exe -m http.server --bind 127.0.0.1 --directory ui 8769
```

Open http://127.0.0.1:8769 and select `face.glb` or `overlay.png`.
The bundle's `face.npz` contains fitted FLAME coefficients and geometry.
`scene.json` records the original camera, crop transforms, and artifact paths.
`priors.npz` has raw network heads and processed per-vertex targets.
Read `timing.json` for timed CUDA stages and peak framework allocation.
The full command's wall time and the preview time have separate entries.

## How the fit works

A crop gives the model a consistent place to look. Each canonical UV location
represents a location on the face template. The network estimates where that
location appears in the photo and how deep it is relative to its depth anchor.
The solver samples those maps at FLAME vertices, projects its current guess,
and changes the FLAME coefficients to reduce the differences. The faces
table connects the regenerated vertices into triangles.

For a one-dimensional example, suppose projected x is `10 + 2p` and the
observed point is at x=14. From p=0 the residual is -4 and its derivative is 2.
The Gauss–Newton update is `-2*(-4)/(2*2) = 2`. The new projection is 14.
The real solver combines many such residuals and uses uncertainty and
regularization to limit the effect of unreliable estimates.

After a real run, open the GLB and orbit it. Then open the overlay. Explain
why moving the preview camera changes the view without changing the fit,
and use the saved original-image camera to investigate a misaligned overlay.
