"""GARD-inspired prior corrector for RealDenseFace ViT-S (demo on NoW validation, cross-validated by person).

GARD's idea: correct the network's intermediate geometric representation, conditioned on the degradation,
instead of restoring pixels. Here the representation is ViT-S's per-vertex priors (5,023 vertices x
[2D position, depth, two log-variances]). A small residual network maps degraded priors to the clean photo's
priors in one step. Coordinates are normalised by each photo's face box, so clean and degraded priors live in
the same frame; corrected priors are written back in the degraded photo's pixel frame and fed to the unchanged
FLAME solver through the oracle substitution hook (src/geometry/oracle.py).

This is not GARD itself (no multi-view, no diffusion denoiser); GARD is cited as the source of the idea.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

VARIANTS = ("original", "face-64", "face-32", "face-16", "compound")
KEYS = ("vertex_coord", "vertex_coord_log_var", "vertex_depth", "vertex_depth_log_var", "face_bbox")


# ----------------------------------------------------------------------------- data

def load_vits_priors(root):
    """{(subject, variant): {"priors": {...}, "path": priors.npz, "image_key": ...}} from runs/*vits*/ records."""
    out = {}
    for rec_path in Path(root).rglob("record.json"):
        rec = json.loads(rec_path.read_text())
        if rec.get("model_id") != "vits" or rec.get("fit_status") != "success":
            continue
        pri = rec_path.parent / "fit" / "priors.npz"
        if not pri.is_file():
            continue
        with np.load(pri) as z:
            if not all(k in z.files for k in KEYS):
                continue
            pri_dict = {k: np.asarray(z[k], np.float32) for k in KEYS}
        out[(rec["subject_id"], rec["variant_id"])] = {"priors": pri_dict, "path": str(pri), "image_key": rec["image_key"]}
    return out


def to_features(p):
    """Per-vertex features in a resolution-independent frame plus the face width (degradation level)."""
    box = p["face_bbox"].reshape(-1)[:4]
    size = np.maximum(box[2:] - box[:2], 1e-6)
    uv = (p["vertex_coord"].reshape(-1, 2) - box[:2]) / size
    feat = np.concatenate([uv, p["vertex_depth"].reshape(-1, 1), p["vertex_coord_log_var"].reshape(-1, 1),
                           p["vertex_depth_log_var"].reshape(-1, 1)], 1).astype(np.float32)
    return feat, float(np.log(size.mean()))


def from_features(feat, like):
    """Back to a priors dict in ``like``'s pixel frame."""
    box = like["face_bbox"].reshape(-1)[:4]
    size = np.maximum(box[2:] - box[:2], 1e-6)
    out = dict(like)
    out["vertex_coord"] = (feat[:, :2] * size + box[:2]).reshape(like["vertex_coord"].shape).astype(np.float32)
    out["vertex_depth"] = feat[:, 2:3].reshape(like["vertex_depth"].shape).astype(np.float32)
    out["vertex_coord_log_var"] = feat[:, 3:4].reshape(like["vertex_coord_log_var"].shape).astype(np.float32)
    out["vertex_depth_log_var"] = feat[:, 4:5].reshape(like["vertex_depth_log_var"].shape).astype(np.float32)
    return out


def make_pairs(data, subjects):
    """(degraded features, log face width, clean features) for every variant incl. original->original."""
    X, W, Y, tags = [], [], [], []
    for s in subjects:
        clean = data.get((s, "original"))
        if not clean:
            continue
        y, _ = to_features(clean["priors"])
        for v in VARIANTS:
            d = data.get((s, v))
            if d:
                x, w = to_features(d["priors"])
                X.append(x); W.append(w); Y.append(y); tags.append((s, v))
    return np.stack(X), np.asarray(W, np.float32), np.stack(Y), tags


# ----------------------------------------------------------------------------- model

def build_model(n_vertices=5023, width=128, embed=16):
    import torch
    from torch import nn

    class PriorCorrector(nn.Module):
        """Residual per-vertex network with a global shape summary and degradation conditioning."""

        def __init__(self):
            super().__init__()
            self.vertex_embed = nn.Embedding(n_vertices, embed)
            self.local = nn.Sequential(nn.Linear(5 + embed + 1, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.head = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                      nn.Linear(width, 5))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # starts as identity

        def forward(self, x, logw):
            b, n, _ = x.shape
            e = self.vertex_embed.weight[None].expand(b, n, -1)
            cond = logw[:, None, None].expand(b, n, 1)
            h = self.local(torch.cat([x, e, cond], -1))
            g = h.max(1, keepdim=True).values.expand(-1, n, -1)
            # Quality gate (degradation conditioning): fades the correction out for large, clean faces
            # (face width >> 96 px), so the corrector leaves good photos untouched.
            gate = torch.sigmoid(4.0 * (math.log(96.0) - logw))[:, None, None]
            return x + gate * self.head(torch.cat([h, g, cond], -1))

    return PriorCorrector()


def _loss(pred, target):
    import torch
    w = torch.exp(-target[..., 3:4]).clamp(max=1e4)  # trust confident clean vertices more
    w = w / w.mean()
    coord = (w * (pred[..., :2] - target[..., :2]).abs()).mean()
    depth = (w * (pred[..., 2:3] - target[..., 2:3]).abs()).mean()
    logv = (pred[..., 3:] - target[..., 3:]).abs().mean()
    return coord * 50 + depth * 20 + logv * 0.1


def train_fold(X, W, Y, epochs=600, lr=2e-3, device=None, seed=0):
    import torch
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(X.shape[1]).to(device)
    x, w, y = (torch.as_tensor(a, device=device) for a in (X, W, Y))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        idx = torch.randperm(len(x), device=device)[:16]
        noise = torch.randn_like(x[idx]) * torch.tensor([2e-3, 2e-3, 1e-3, 0, 0], device=device)  # light augmentation
        loss = _loss(model(x[idx] + noise, w[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


def cross_validate(data, folds=5, out_dir=None, epochs=600, seed=0):
    """Person-disjoint folds. Writes corrected priors per (subject, variant); returns prior-level error table."""
    import torch
    subjects = sorted({s for s, v in data if v == "original"})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subjects))
    groups = [order[i::folds] for i in range(folds)]
    rows = []
    out_dir = Path(out_dir) if out_dir else None
    for k, test in enumerate(groups):
        train = [s for s in subjects if s not in test]
        X, W, Y, _ = make_pairs(data, train)
        model = train_fold(X, W, Y, epochs=epochs, seed=seed + k)
        Xt, Wt, Yt, tags = make_pairs(data, test)
        dev = next(model.parameters()).device
        with torch.no_grad():
            P = model(torch.as_tensor(Xt, device=dev), torch.as_tensor(Wt, device=dev)).cpu().numpy()
        for (s, v), x, p, y in zip(tags, Xt, P, Yt):
            rows.append({"subject": s, "variant": v, "fold": k,
                         "coord_err_degraded": float(np.abs(x[:, :2] - y[:, :2]).mean()),
                         "coord_err_corrected": float(np.abs(p[:, :2] - y[:, :2]).mean()),
                         "depth_err_degraded": float(np.abs(x[:, 2] - y[:, 2]).mean()),
                         "depth_err_corrected": float(np.abs(p[:, 2] - y[:, 2]).mean())})
            if out_dir:
                like = data[(s, v)]["priors"]
                path = out_dir / s / v / "priors.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **from_features(p, like))
    return rows


# ----------------------------------------------------------------------------- refits (Kaggle GPU workers)

def refit_jobs(data, corrected_dir, out):
    """One job per (subject, variant, arm). Arms: corrected priors (all 5 variants), oracle clean priors (4)."""
    jobs = []
    for (s, v), d in sorted(data.items()):
        if (s, "original") not in data:
            continue
        cpath = Path(corrected_dir) / s / v / "priors.npz"
        if cpath.is_file():
            jobs.append({"arm": "corrected", "key": d["image_key"], "variant": v, "priors": str(cpath),
                         "out": str(Path(out) / "runs" / f"corrected-{s}")})
        if v != "original":
            jobs.append({"arm": "oracle", "key": d["image_key"], "variant": v, "priors": data[(s, "original")]["path"],
                         "out": str(Path(out) / "runs" / f"oracle-{s}")})
    return jobs


def run_refit_job(job):
    """Fit one degraded variant with substituted priors (mode both) and score it against the scan."""
    from ..benchmark import KAGGLE_VARIANTS, _fit, model_specs, run_matrix
    from ..data_now import installed_cases
    from ..kaggle import paths
    env = {"TRUEFACE3D_PRIORS_IN": job["priors"], "TRUEFACE3D_PRIORS_MODE": "both"}

    def runner(spec, input_path, folder, limit):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return _fit(spec, input_path, folder, limit)
        finally:
            for k, val in old.items():
                os.environ.pop(k, None) if val is None else os.environ.__setitem__(k, val)

    cases = [c for c in installed_cases(paths.data_root()) if c.image_key == job["key"]]
    variant = next(v for v in KAGGLE_VARIANTS if v.name == job["variant"])
    return run_matrix(cases, model_specs(("vits",)), [variant], job["out"], runner=runner, timeout=900)


def collect_scores(root, arm_from_run=False):
    """{(subject, variant): mean_mm} (or {(arm, subject, variant): mean_mm}) for ViT-S records under ``root``."""
    scores = {}
    for rec_path in Path(root).rglob("record.json"):
        rec = json.loads(rec_path.read_text())
        score = rec_path.parent / "score.json"
        if rec.get("model_id") != "vits" or rec.get("score_status") != "success" or not score.is_file():
            continue
        mm = json.loads(score.read_text())["mean_mm"]
        if arm_from_run:
            run = next((p.name for p in rec_path.parents if p.parent.name == "runs"), "")
            scores[(run.split("-", 1)[0], rec["subject_id"], rec["variant_id"])] = mm
        else:
            scores[(rec["subject_id"], rec["variant_id"])] = mm
    return scores


def gap_table(baseline_scores, arm_scores):
    """Per variant: mean scan error for degraded / corrected / oracle and % of the clean->degraded gap recovered.

    baseline_scores: {(subject, variant): mm} for the plain ViT-S fits; arm_scores: {(arm, subject, variant): mm}.
    """
    table = []
    for v in VARIANTS:
        subs = [s for (s, vv) in baseline_scores if vv == v and (s, "original") in baseline_scores]
        row = {"variant": v, "n": len(subs)}
        for arm in ("baseline", "corrected", "oracle"):
            vals = []
            for s in subs:
                if arm == "baseline":
                    vals.append(baseline_scores[(s, v)])
                elif (arm, s, v) in arm_scores:
                    vals.append(arm_scores[(arm, s, v)])
            row[arm] = float(np.mean(vals)) if vals else math.nan
            row[arm + "_n"] = len(vals)
        clean = float(np.mean([baseline_scores[(s, "original")] for s in subs])) if subs else math.nan
        row["clean"] = clean
        gap = row["baseline"] - clean
        row["corrected_gap_recovered_pct"] = 100 * (row["baseline"] - row["corrected"]) / gap if gap > 1e-6 else math.nan
        row["oracle_gap_recovered_pct"] = 100 * (row["baseline"] - row["oracle"]) / gap if gap > 1e-6 else math.nan
        table.append(row)
    return table
