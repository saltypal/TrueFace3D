"""Generate notebooks/32_mesh_benchmark.ipynb: every FaceScape-C mesh metric, tables and figures.

    .venv-linux/bin/python notebooks/make_32_mesh_benchmark.py
    .venv-linux/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/32_mesh_benchmark.ipynb

Reads scores.csv / quality.csv / fits.csv from local runs (/mnt/DataBunker/benchmark_local/*) and from
downloaded Kaggle outputs (/mnt/DataBunker/kaggle_outputs/*, folders or result zips). Only measured rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src" / "facescape").is_dir())
sys.path.insert(0, str(REPO))

from src.kaggle.notebook_kit import code, md, save  # noqa: E402


def cells():
    return [
        md("""# 32 - FaceScape-C mesh benchmark results

**Question.** How much does each monocular face reconstruction model's 3D error grow when the photo is degraded?

**Data.** Real FaceScape photos (no renders): test subjects x expressions 1, 2, 3, 19 x views front / left45 / right45,
each clean and under 42 degradations (C01-C10 synthetic corruptions, R1-R4 real-world pipelines, 3 severities).

**Metric (scan error).** The predicted mesh is aligned to the TU ground-truth mesh (7-landmark similarity, then
similarity ICP). Error = distance from each TU face-region vertex to the predicted surface, in mm (lower is better).
`posed` = expression mesh vs that expression's GT (MICA predicts identity only, so it gives a neutral mesh);
`neutral` = neutral mesh, scored on neutral (expression 1) photos only, which is fair to every model.

Every number below is measured from stored fits; nothing is estimated."""),
        code('''import sys, json
from pathlib import Path
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "facescape").is_dir())
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.facescape import report
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
S = report.prefer_local(report.collect("scores.csv"))
F = report.prefer_local(report.collect("fits.csv"))
Q = report.collect("quality.csv").drop_duplicates(["image_id", "condition"])
print("score rows:", len(S), "| sources:", S.source.value_counts().to_dict() if len(S) else {})
print("models:", sorted(S.model.unique()) if len(S) else [])'''),
        md("## 1. Did every fit run? (fit success per model and degradation profile)"),
        code('''if len(F):
    t = F.assign(ok=F.fit_status.eq("success")).pivot_table(index="model_name", columns="profile", values="ok", aggfunc="mean")
    display((100 * t).round(1).style.background_gradient(cmap="RdYlGn", vmin=50, vmax=100).format("{:.1f}%"))
    fails = F[F.fit_status != "success"]
    if len(fails):
        display(fails.groupby(["model_name", "error_stage"]).size().rename("failed fits").to_frame())'''),
        md("## 2. Headline: clean error and mean error under degradation (posed protocol)"),
        code('''ok = S[S.score_status == "success"].copy()
P = ok[ok.protocol == "posed"]
head = pd.DataFrame({
    "clean mean mm": P[P.condition == "clean"].groupby("model_name").mean_mm.mean(),
    "clean median mm": P[P.condition == "clean"].groupby("model_name").median_mm.mean(),
    "degraded mean mm": P[P.condition != "clean"].groupby("model_name").mean_mm.mean(),
    "worst profile mm": P[P.condition != "clean"].groupby(["model_name", "profile"]).mean_mm.mean().groupby("model_name").max(),
    "images scored": P.groupby("model_name").size(),
})
head["degradation +mm"] = head["degraded mean mm"] - head["clean mean mm"]
head.sort_values("clean mean mm").round(3)'''),
        md("## 3. Error per profile and severity (mm, posed)"),
        code('''tab = P.pivot_table(index=["model_name", "severity"], columns="profile", values="mean_mm", aggfunc="mean")
display(tab.round(2).style.background_gradient(cmap="magma_r", axis=None).format("{:.2f}"))'''),
        code('''profiles = sorted(P[P.condition != "clean"].profile.unique())
fig, axes = plt.subplots(2, (len(profiles) + 1) // 2, figsize=(3.1 * ((len(profiles) + 1) // 2), 6.2), sharey=True)
for ax, prof in zip(axes.ravel(), profiles):
    for name, g in P[(P.profile == prof) | (P.condition == "clean")].groupby("model_name"):
        y = g.groupby("severity").mean_mm.mean()
        ax.plot(y.index, y.values, marker="o", label=name)
    ax.set_title(prof); ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=.3)
axes.ravel()[0].set_ylabel("scan error (mm)"); axes.ravel()[0].legend(fontsize=7)
for ax in axes.ravel()[len(profiles):]: ax.axis("off")
plt.suptitle("Scan error vs severity (0 = clean)"); plt.tight_layout(); plt.show()'''),
        md("""## 4. Robustness: paired increase over the same image's clean error
`delta` = degraded error - clean error of the same image (same model). `mCE vs DECA` averages, over profiles, each model's
corruption error divided by DECA's (ImageNet-C convention; < 1 means more robust than DECA)."""),
        code('''D = report.paired_degradation(P)
rob = D.groupby("model_name")[["delta_mean_mm", "ratio_mean_mm"]].mean()
mc = report.mce(D.assign(model=D.model)).rename(index=report.MODEL_NAMES)
rob["mCE vs DECA"] = mc
display(rob.sort_values("delta_mean_mm").round(3))
display(D.pivot_table(index="model_name", columns="profile", values="delta_mean_mm", aggfunc="mean").round(2)
        .style.background_gradient(cmap="Reds", axis=None).format("{:+.2f}"))'''),
        md("## 5. View and expression effects (clean photos)"),
        code('''C = P[P.condition == "clean"]
display(C.pivot_table(index="model_name", columns="view", values="mean_mm", aggfunc="mean").round(3))
display(C.pivot_table(index="model_name", columns="expression", values="mean_mm", aggfunc="mean").round(3))'''),
        md("## 6. Neutral protocol (neutral photos only; every model predicts a neutral mesh)"),
        code('''N = ok[ok.protocol == "neutral"]
display(N.pivot_table(index="model_name", columns="profile", values="mean_mm", aggfunc="mean").round(2))'''),
        md("## 7. Does input quality predict the error? (Spearman correlation per model)"),
        code('''if len(Q):
    M = P.merge(Q[["image_id", "condition", "psnr_db", "ssim", "sharpness_lapvar", "noise_sigma_mad", "width"]],
                on=["image_id", "condition"], how="left")
    corr = M.groupby("model_name").apply(lambda g: g[["mean_mm", "psnr_db", "ssim", "sharpness_lapvar", "width"]]
                                         .corr(method="spearman")["mean_mm"].drop("mean_mm"))
    display(corr.round(3))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in M.groupby("model_name"):
        b = g.groupby(pd.cut(g.psnr_db, 8)).mean_mm.mean()
        ax.plot([i.mid for i in b.index], b.values, marker="o", label=name)
    ax.set_xlabel("input PSNR vs clean (dB)"); ax.set_ylabel("scan error (mm)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.show()'''),
        md("## 8. Where on the face is the error? Subject 344 (publishable), neutral, front view"),
        code('''from src.facescape.dataset import FaceScapeC
from src.facescape import score as sc
ds = FaceScapeC.find()
case = ds.cases(subject=344, expression=1, view="front")[0]
gt = case.gt_vertices_tu(); mask = sc.face_mask_vertices(str(ds.root)); ref = gt[mask]
ref_lmk = gt[ds.topology["landmarks68"][sc.IBUG7]]
conds = ["clean", "C01_s3", "C07_s3", "R2_s3"]
roots = {}
for src in sorted(set(S.source)):
    kind, name = src.split(":", 1)
    roots[src] = (report.LOCAL if kind == "local" else report.KAGGLE) / name
models = sorted(P.model.unique())
fig, axes = plt.subplots(len(models), len(conds), figsize=(3 * len(conds), 3.2 * len(models)), squeeze=False)
for i, m in enumerate(models):
    for j, cond in enumerate(conds):
        ax = axes[i, j]; ax.axis("off")
        folder = next((r / "fits" / m / cond / "344_01_front" for r in roots.values()
                       if (r / "fits" / m / cond / "344_01_front").is_dir()), None)
        if folder is None:
            continue
        try:
            v, f, lmk, _ = sc.prediction(m, folder, "posed")
            dist, _, _ = sc.align_and_measure(v, f, lmk, ref, ref_lmk)
        except Exception as e:  # noqa: BLE001
            ax.set_title(f"{m} {cond}: {type(e).__name__}", fontsize=7); continue
        im = ax.scatter(ref[:, 0], ref[:, 1], c=dist, s=0.3, cmap="jet", vmin=0, vmax=5)
        ax.set_aspect("equal"); ax.set_title(f"{report.MODEL_NAMES.get(m, m)} | {cond}\\nmean {dist.mean():.2f} mm", fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.5, label="error (mm)"); plt.show()'''),
        md("## 9. Save the tables"),
        code('''OUT = REPO / "experiments" / "facescape_c"; OUT.mkdir(parents=True, exist_ok=True)
head.round(4).to_csv(OUT / "headline_posed.csv"); tab.round(4).to_csv(OUT / "profile_severity_posed.csv")
rob.round(4).to_csv(OUT / "robustness.csv")
print("saved to", OUT)'''),
    ]


if __name__ == "__main__":
    print(save(REPO / "notebooks" / "32_mesh_benchmark.ipynb", cells()))
