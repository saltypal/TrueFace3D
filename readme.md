# True Face 

The [TrueFace3D master plan](CHAPTER_ROADMAP.md) governs the current project. Chapter 1 benchmarks geometry and evaluates one targeted prior correction. The texture study compares raw projection with one appearance method and delivers an editable textured GLB. The [journal](Journal.md) records measured results and remaining gates. See the [geometry pilot contracts](IMPLEMENTATION_PLAN.md) for the implemented baseline.

TrueFace3D estimates an editable 3D face and skin appearance from one portrait. It compares pretrained geometry models on clean and degraded inputs and tests correction of their geometric priors when the data supports it.

RealDenseFace: Real-time Monocular 3D Face Reconstruction from Dense UV-space Prior

The geometry pilot exports a GLB for inspection and a parametric bundle for editing. Textured GLBs are a planned skin deliverable. Surface detail, eyes and lighting follow the core studies.

Start with the [research pack](Papers/README.md),
[44-paper index](Papers/PAPER_INDEX.md), and
[archived component design](Papers/TRUEFACE3D_MASTER_PLAN.md).
The [selected module stack](Papers/MODULE_SELECTIONS.md) resolves the first
implementation choices and records their release/compatibility limits.
The [October execution plan](Papers/ExecutionPlan/README.md) records the earlier pretrained-first design and its component research. CPU ViT-S prior maps exist for three portraits. The MX330 now runs real-photo ViT-S and ViT-B fits in WSL. See the Chapter 1 notebooks and journal for the actual evaluation coverage.

## Executed geometry pilot notebooks

The [one-photo scan error notebook](notebooks/04_one_photo_scan_error.ipynb)
compares fresh original and degraded fits against the real NoW scan. It explains
the common-scale error maps, score differences and mesh movement. With the CUDA
environment settings below, reproduce or resume those fits from the project root.

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -u experiments/geometry/rerun_one_photo.py
```

The [Chapter 0 geometry benchmark](notebooks/00a_shape_benchmark.ipynb) shows
the original-input table, local NoW alignment, and degradation scores. It
loads the completed development runs and the held-out comparison as they
become available.

Its one-photo gallery shows clay and projected photo color on all five models
for the same original and five degraded inputs. Regenerate the six figures
from the saved fits with this command.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -u experiments/geometry/render_subject_gallery.py
```

1. [Understand one ViT-S reconstruction](notebooks/01_geometry_tutorial.ipynb)
2. [Compare original-input models](notebooks/02_geometry_benchmark.ipynb)
3. [Inspect paired degradation](notebooks/03_degradation_and_prior_correction.ipynb)

Open JupyterLab from the project root. The comparison notebooks replay
local evidence by default. For new runs use a fresh output path so original
and degraded manifests can be traced independently.

```powershell
.\.venv\Scripts\python.exe -m jupyterlab .\notebooks
```

Use the [ViT-S lab commands](docs/geometry/02_realdenseface.md) to regenerate
a face. To rerun the original and degraded pilot matrices from the terminal:

```powershell
$env:TRUEFACE3D_COMPARATOR_DEVICE = 'cuda'
$env:TRUEFACE3D_PIXEL3DMM_WSL_DISTRO = 'Ubuntu-26.04'
$env:TRUEFACE3D_PIXEL3DMM_WSL_PYTHON = '/home/Satya/.venvs/trueface3d-vit/bin/python'
$env:TRUEFACE3D_PIXEL3DMM_WSL_PYTHONPATH = '/home/Satya/.cache/pixel3dmm-build/pytorch3d'
```

Run the current five-model benchmark with the same local settings. The
runner reuses completed attempts when you rerun an identical request.

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -u experiments/geometry/benchmark_chapter0.py
```

The earlier pilot commands remain available for a one-subject reproduction.

```powershell
.\.venv\Scripts\python.exe -m src.cli benchmark --mode original --subjects 1 --output experiments/geometry/runs/my-original-01
.\.venv\Scripts\python.exe -m src.cli benchmark --mode degraded --subjects 1 --output experiments/geometry/runs/my-degraded-01
```

Set `TRUEFACE3D_ORIGINAL_RUN` and `TRUEFACE3D_DEGRADED_RUN` to those directory
names before launching Jupyter to replay them. Compare times only when their
device and timing boundaries agree. The benchmark stores attempts, quality
status, CSV tables and worker logs. A fit can complete while scan scoring
waits for the official NoW scan landmarks. The local score backend is a
diagnostic and its rows carry the protocol name. An isolated official
evaluator environment can be selected with `--protocol official_now_nometrical
--official-python <interpreter>`.

The development restart saved the two additional neutral cases in
`experiments/geometry/runs/chapter1-development-manifest/new-identity-pairs.txt`.
With the same CUDA environment above, run their original and 64 px conditions
with these commands.

```powershell
$pilotList = 'experiments/geometry/runs/chapter1-development-manifest/new-identity-pairs.txt'
.\.venv\Scripts\python.exe -m src.cli benchmark --mode original --subjects 2 --image-list $pilotList --output experiments/geometry/runs/chapter1-development-original-01
.\.venv\Scripts\python.exe -m src.cli benchmark --mode degraded --primary-only --subjects 2 --image-list $pilotList --output experiments/geometry/runs/chapter1-development-degraded-01
```

If the official validation list is unavailable, the notebooks can replay from
the saved `three-neutral-pairs.txt` pilot manifest. The full validation inventory
still needs the official `imagepathsvalidation.txt`.

Once the official scan landmarks are under `Data/scans_lmks_onlypp/`,
inventory the files and score the retained meshes in new result folders.

```powershell
.\.venv\Scripts\python.exe -m src.cli inventory-now
.\.venv\Scripts\python.exe -m src.cli rescore-now --input-run experiments/geometry/runs/chapter1-original-01 --retry-run experiments/geometry/runs/chapter1-pixel-original-03 --output experiments/geometry/runs/my-original-scores-01
.\.venv\Scripts\python.exe -m src.cli rescore-now --input-run experiments/geometry/runs/chapter1-degraded-01 --retry-run experiments/geometry/runs/chapter1-pixel-degraded-03 --output experiments/geometry/runs/my-degraded-scores-01
```

Replay the original pilot crops to count supported neural operations in a
separate FP32 profiling pass. The per-stage receipts include the operator
breakdown. Notebook 02 displays the consolidated table.

```powershell
.\experiments\geometry\profile_chapter1.ps1 -Output experiments/geometry/runs/my-chapter1-flops
$env:TRUEFACE3D_FLOP_RUN = 'my-chapter1-flops'
```
