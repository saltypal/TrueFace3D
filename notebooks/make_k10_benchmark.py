"""Generate the FaceScape-C mesh benchmark notebooks (K10), one per model, for either Kaggle account.

    .venv-linux/bin/python notebooks/make_k10_benchmark.py            # writes notebooks/kaggle/K10_*.ipynb
    .venv-linux/bin/python scripts/kaggle_push.py notebooks/kaggle/K10_deca.ipynb --account b --slug tf3d-k10-deca

Scope (test grid): test subjects x expressions (1, 2, 3, 19) x 3 views x 43 conditions = 1,548 images per model.
Each notebook: fresh clone -> run info -> GPUs -> CUDA builds (asserted) -> dataset scope -> image quality
of every input -> one-image gate -> full run on both GPUs with an 11 h deadline -> tables -> 344 preview -> zip.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src" / "facescape").is_dir())
sys.path.insert(0, str(REPO))

from src.kaggle.notebook_kit import clone_cell, code, md, run_info_cell, save, zip_cell  # noqa: E402

NEEDS = {"vits": ("nvdiffrast.torch", "flame_solver"), "vitb": ("nvdiffrast.torch", "flame_solver"),
         "deca": (), "mica": ()}
NAMES = {"vits": "RealDenseFace ViT-S", "vitb": "RealDenseFace ViT-B", "deca": "DECA", "mica": "MICA"}


def cells(model: str, expressions=(1, 2, 3, 19), tag=""):
    out = f"/kaggle/working/k10_{model}{tag}"
    n = 3 * 3 * 43 * len(expressions)
    return [
        md(f"""# K10 - FaceScape-C mesh benchmark: {NAMES[model]}

Real FaceScape photos (no renders). Test subjects 344, 350, 352 x expressions {list(expressions)} x views front / left45 / right45
x 43 conditions (clean + C01-C10 and R1-R4 at 3 severities) = **{n:,} images**.

Saved for every image: the full fit bundle (meshes, priors when the model has them, renders, logs),
`record.json` (status, timing, GPU), and `quality.csv` (PSNR, SSIM, sharpness, noise, brightness of each input).
Scoring against the TU scans runs afterwards on CPU (notebook 32). Progress prints every 2 minutes."""),
        code(clone_cell()),
        code(run_info_cell(out)),
        md("## 1. GPUs and environment"),
        code(f'''from src.kaggle import readiness
ENV = readiness.environment()
print(ENV["nvidia_smi"]); print(json.dumps(ENV["torch"], indent=1))
(OUT / "environment.json").write_text(json.dumps(ENV, indent=1, default=str))
assert ENV["torch"].get("available"), "no CUDA device"
MODEL = "{model}"
EXPRESSIONS = {list(expressions)!r}'''),
        md("## 2. Upstream code, weights and CUDA extensions"),
        code(f'''from src.kaggle import setup
t = time.time(); REPORT = setup.main(["--skip-appearance"])
(OUT / "setup_report.json").write_text(json.dumps(REPORT, indent=1, default=str))
print("[setup] builds:", REPORT.get("builds"), f"{{(time.time() - t) / 60:.1f}} min", flush=True)
print("[setup] missing weights:", REPORT.get("missing_weights"), flush=True)
import importlib
IMPORTS = {{}}
for m in ("nvdiffrast.torch", "flame_solver", "pytorch3d"):
    try:
        importlib.import_module(m); IMPORTS[m] = "ok"
    except Exception as e:  # noqa: BLE001
        IMPORTS[m] = f"{{type(e).__name__}}: {{e}}"[:200]
print("[import]", IMPORTS, flush=True)
(OUT / "imports.json").write_text(json.dumps(IMPORTS, indent=1))
for m in {NEEDS[model]!r}:
    assert IMPORTS[m] == "ok", f"{{MODEL}} needs {{m}}: {{IMPORTS[m]}}"'''),
        md("## 3. Dataset and benchmark scope"),
        code('''import pandas as pd
from src.facescape.dataset import FaceScapeC
from src.kaggle import facescape_jobs as fj
DS = FaceScapeC.find(); print("[data]", DS.root, flush=True)
ROWS = fj.benchmark_cases(DS, expressions=EXPRESSIONS)
scope = pd.DataFrame(ROWS)
print(f"[data] {len(ROWS)} images; subjects {sorted(scope.subject.unique())}; "
      f"expressions {sorted(scope.expression.unique())}; views {sorted(scope.view.unique())}; "
      f"conditions {scope.condition.nunique()}", flush=True)
assert len(ROWS) > 0
scope.to_csv(OUT / "scope.csv", index=False)
scope.groupby(["subject", "view"]).size().unstack()'''),
        md("## 4. Input quality of every image (CPU)"),
        code('''from src.facescape import quality
clean_by = {r["image_id"]: r for r in DS.clean_rows}
Q = []
for i, r in enumerate(ROWS, 1):
    Q.append(quality.measure(DS.root, r, clean_by[r["image_id"]]))
    if i % 300 == 0 or i == len(ROWS):
        print(f"[quality] {i}/{len(ROWS)}", flush=True)
Q = pd.DataFrame(Q); Q.to_csv(OUT / "quality.csv", index=False)
Q["profile"] = Q.condition.str.split("_").str[0]
Q.groupby("profile")[["psnr_db", "ssim", "sharpness_lapvar", "noise_sigma_mad", "brightness_mean", "width"]].mean().round(2)'''),
        md("## 5. Gate: one clean image on GPU 0 must fit before the full run"),
        code('''from src.kaggle import runner
GATE_OUT = OUT   # same folder: the full run skips this finished fit
first = next(r for r in ROWS if r["condition"] == "clean")
gate_job = [{"kind": "facescape", "model": MODEL, "variant": "clean",
             "items": [[first["image_id"], "clean", first["path"]]], "out": str(GATE_OUT)}]
runner.launch("src.kaggle.facescape_jobs:run_job", gate_job, OUT / "logs_gate", gpu_ids=[0], cwd=REPO, report_every=30)
GATE = fj.read_records(GATE_OUT)
print("[gate]", [(g["image_id"], g["fit_status"], g.get("wall_seconds"), g.get("message", "")[:300]) for g in GATE], flush=True)
print((OUT / "logs_gate" / "worker_gpu0.log").read_text()[-3000:])
assert GATE and GATE[0]["fit_status"] == "success", "gate fit failed; see the log above"'''),
        md("## 6. Full run on both GPUs (stops starting new fits after 11 h so outputs are saved)"),
        code('''DEADLINE = T0 + 11 * 3600
JOBS = fj.make_jobs(ROWS, [MODEL], OUT)
print(f"[run] {len(JOBS)} jobs ({sum(len(j['items']) for j in JOBS)} fits) on 2 GPUs; "
      f"deadline {time.strftime('%H:%M', time.localtime(DEADLINE))}", flush=True)
codes = runner.launch("src.kaggle.facescape_jobs:run_job", JOBS, OUT / "logs", gpu_ids=[0, 1], cwd=REPO,
                      extra_env={"TRUEFACE3D_DEADLINE": str(DEADLINE)}, report_every=120)
print("[run] worker exit codes", codes, flush=True)'''),
        md("## 7. Results: status, timing, priors"),
        code('''R = pd.DataFrame(fj.read_records(OUT))
R.drop(columns=["files"], errors="ignore").to_csv(OUT / "fits.csv", index=False)
print(f"[result] {len(R)} records; success {(R.fit_status == 'success').sum()}", flush=True)
R["profile"] = R.condition.str.split("_").str[0]
display(R.pivot_table(index="profile", columns="fit_status", values="image_id", aggfunc="count", fill_value=0))
display(R.groupby("fit_status").wall_seconds.describe().round(1))
if "error_stage" in R:
    display(R[R.fit_status != "success"].groupby(["error_stage"]).size().rename("failures"))
print("[result] fits with priors.npz:", int(R.get("has_priors", pd.Series(dtype=bool)).fillna(False).sum()), flush=True)'''),
        md("## 8. Preview: subject 344 (publishable), front view, neutral"),
        code('''import matplotlib.pyplot as plt, cv2
pick = ["clean", "C01_s3", "C04_s3", "C07_s3", "R2_s3"]
img_id = "344_01_front"
fig, ax = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.6))
for j, cond in enumerate(pick):
    row = next((r for r in ROWS if r["image_id"] == img_id and r["condition"] == cond), None)
    if row is not None:
        ax[0, j].imshow(cv2.cvtColor(cv2.imread(str(DS.root / row["path"])), cv2.COLOR_BGR2RGB))
    folder = OUT / "fits" / MODEL / cond / img_id
    shots = sorted(folder.rglob("overlay.png")) or sorted(folder.rglob("render.png")) or sorted(folder.rglob("*.png"))
    if shots:
        ax[1, j].imshow(cv2.cvtColor(cv2.imread(str(shots[0])), cv2.COLOR_BGR2RGB))
    ax[0, j].set_title(cond); [a.axis("off") for a in ax[:, j]]
plt.tight_layout(); plt.savefig(OUT / "preview_344.png", dpi=90); plt.show()'''),
        md("## 9. Zip every output (fit bundles, records, quality, logs)"),
        code(zip_cell(out, exclude=("*.whl", "__pycache__"), move=model.startswith("vit"))),
    ]


def main():
    written = []
    # RealDenseFace keeps ~12 MB per fit (9 MB priors), so it is split in two halves of 774 images (~9 GB each).
    for model in ("vits", "vitb"):
        for exprs, tag in (((1, 2), "_e1-2"), ((3, 19), "_e3-19")):
            written.append(save(REPO / "notebooks" / "kaggle" / f"K10_{model}{tag}.ipynb", cells(model, exprs, tag)))
    for model in ("deca", "mica"):
        written.append(save(REPO / "notebooks" / "kaggle" / f"K10_{model}.ipynb", cells(model)))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
