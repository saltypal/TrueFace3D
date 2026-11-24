# TrueFace3D — Final Plan (2-day sprint)

**23 November 2026 · Satya Srinivas Paladugu · AID24063**
Status: **proposed, awaiting review.** The previous roadmap is archived in
[docs/archive/](docs/archive/CHAPTER_ROADMAP_2026-11-23_before-replan.md).

---

## 1. Where the project actually stands

| Area | State | Verdict |
|---|---|---|
| Data | NoW validation: 352 photos, 20 people, scans + 7 scan landmarks, all verified. NoW test split: **1,702 photos of 80 other people** on disk (no scans). | Ready |
| Models | RealDenseFace ViT-S / ViT-B, DECA coarse, Pixel3DMM, MICA all run on the MX330 (WSL, FP32). Real-ESRGAN, GFPGAN, OpenDelight weights staged. | Ready |
| Scorer | Local NoW-style scorer (7-landmark init + similarity ICP + scan→mesh distance). Official evaluator failed to build (CGAL/Boost). | Usable, **needs one sanity check** |
| Code | `src/benchmark.py`, `evaluation.py`, `data_now.py`, `robustness/degrade.py`, `geometry/*`, `reporting.py`, GLB viewer. Restartable run records. | Good base — reuse |
| Evidence | 50 + 30 meshes on **3 people**, 9 held-out originals. | **Too thin** |
| Notebooks | 00a, 01–04 executed. | Overlapping; consolidate |

### What the numbers say so far (local protocol, mean scan error change, mm)

| Model | 32 px (3 people) | 64 px (3 people) | Heavy blur (3 people) |
|---|---|---|---|
| ViT-S | +0.341, +0.114, +0.036 | +0.062, −0.135, −0.011 | +0.016, −0.086, −0.254 |
| ViT-B | +0.362, +0.195, −0.238 | +0.058, +0.069, −0.198 | +0.289, +0.212, −0.372 |
| DECA | +0.021, −0.025, −0.037 | +0.026, +0.001, −0.015 | +0.015, −0.044, −0.054 |
| MICA | −0.155, +0.181, +0.075 | −0.120, +0.143, −0.023 | −0.242, +0.129, +0.083 |

**Honest reading:** the degradation penalty is small (≤0.36 mm on a ~1.3 mm
baseline) and about the same size as the run-to-run scatter between people.
Only ViT-S at 32 px is consistently positive. A prior-correction paper
cannot be built on this until we know the gap is real.

### Gaps in the work done so far

1. **Too few people.** Every degradation conclusion rests on 1–3 subjects.
2. **Degradations too mild and too clean.** Single-factor downsampling/blur
   only. Real bad photos are *combined* (blur + downsample + noise + JPEG).
   No 16 px condition.
3. **No upper bound.** We never measured how much error disappears if the
   priors were perfect (clean-photo priors swapped in). Without this we
   don't know whether prior correction *can* help.
4. **The metric may hide the damage.** NoW scores *neutral* shape after a
   scale-allowed ICP. FLAME's shape prior keeps neutral faces plausible, so
   errors from bad input may mostly show up in pose/expression/camera,
   which NoW does not score.
5. **Scorer not anchored.** The local scorer has never been compared with a
   published number.
6. **Training data plan was weak** (folds over 20 scanned people). The 80
   test-split people need no scans to train a prior corrector — the target
   is the clean photo's own priors.
7. **Over-documentation.** 19 planning docs and 5 notebooks describe the
   same pilot. This plan replaces them.

### How far from a paper

- **Basic conference / workshop paper:** reachable in 2 days **if** the
  20-person benchmark shows a real gap and the corrector closes part of it
  better than Real-ESRGAN / GFPGAN. A clean negative result (“geometry is
  robust, the oracle bound is small, pixels/appearance are what break”)
  is also publishable at workshop level.
- **High-end venue (CVPR/ICCV/ECCV):** not in 2 days. It would need posed
  geometry with ground truth (NeRSemble/FaceScape), real low-quality
  photos, a second dataset, and a clear win over L2R (CVPR 2022).

---

## 2. The one research question

> **When the photo is bad, is it better to fix the pixels (restoration) or
> to fix the geometric evidence (priors) that the 3D fitter consumes?**

Everything else (texture, eyes, lighting) is downstream and optional.

---

## 3. Decision record

### ADR-001: Where to intervene for degraded-photo reconstruction

**Status:** Proposed · **Deciders:** Satya

**Context.** RealDenseFace splits reconstruction into a network that
predicts priors (where each face point lands in the photo, its depth, its
uncertainty) and a solver that fits FLAME to them. That split gives four
places to intervene. Budget: 2 days, MX330 2 GB, NoW data must stay local.

| Option | Acts on | Training | Cost | Risk |
|---|---|---|---|---|
| A. Restore pixels (Real-ESRGAN, GFPGAN) | Photo | None | Slow, ~1–3 s | Hallucinated identity |
| B. Feature corrector (GARD idea) | Encoder features | Adapter | Must cache big features | Needs hooks into ViT; heavy |
| **C. Prior corrector (ours)** | Priors in UV | Small net | Tiny inputs, trains on CPU | Capped by the oracle bound |
| D. Robust fitting | Solver weights | None | Cheap | Only downweights, never fixes |

**Decision.** Build **C** as the method. Run **A** and **D** as controls.
Keep **B** as a stretch ablation (same training code, different tensor).

**Why C.** Priors are small (5,023 vertices × 5 numbers, or a UV image),
interpretable, and plug into the unchanged solver. It is the only option we
can train, evaluate and explain inside 2 days. GARD itself needs ~10 views,
Depth Anything 3 and a diffusion denoiser; we borrow only its idea —
correct the intermediate representation, not the pixels.

**Go / no-go gate (decided before training).** Run the *oracle* (clean
priors swapped into the degraded fit) on all 20 validation people.

- If the oracle recovers **≥ 50 % of the degradation penalty** and the
  penalty is **> 0.10 mm median with a paired Wilcoxon p < 0.05** →
  train the corrector.
- Otherwise → stop correcting geometry. Write the benchmark paper
  (robustness of 5 models + oracle bound) and move effort to texture.

**Consequences.** Easier: one small network, honest upper bound, clean
story. Harder: a small gap caps the result. Revisit B and posed geometry
only after the paper draft exists.

---

## 4. Experiments

### E1 — Benchmark 1: clean accuracy (all 5 models)
- 20 validation people × 1 neutral photo (deterministic pick, already in
  manifest). Pixel3DMM on a fixed 6-person subset (≈5 min/fit).
- Report mean / median / p95 scan error, fit success, time, VRAM, GFLOPs.
- **Scorer anchor:** compare our DECA numbers to DECA's published NoW
  validation numbers. Must be in the same ballpark before we trust anything.

### E2 — Benchmark 2: degraded vs clean (all 5 models)
Same photos, same scans. Conditions (fixed now, no more changes):

| Condition | Recipe |
|---|---|
| `face-32` | Downsample whole photo so the face is 32 px wide |
| `face-16` | Same, 16 px (new, harsher) |
| `compound` | Real-ESRGAN-style chain: blur σ≈1.5 → downsample to 48 px face → Gaussian noise σ=8 → JPEG q=30 (fixed seed) |

Report per person: clean error, degraded error, **penalty = degraded − clean**,
fit/score coverage. Statistic: paired Wilcoxon over 20 people.

### E3 — Oracle bound (ViT-S only)
For each degraded photo, fit with: degraded priors / clean correspondence
swapped in / clean depth swapped in / both. Same crop, same solver. Tells us
the maximum a prior corrector could ever gain and *which* prior matters.

### E4 — Prior corrector (only if the gate passes)
- **Input:** degraded ViT-S priors as UV maps (correspondence x,y, depth,
  two log-variances) + face width in px.
- **Network:** small UV U-Net (< 2 M params). Two heads:
  (1) *analyse* — predicted degradation severity s ∈ [0,1];
  (2) *fix* — residual Δ for correspondence and depth.
  Output = degraded + s·Δ, so a clean photo passes through unchanged.
- **Target:** the clean photo's own priors. **No scans used in training.**
- **Training data:** the 80 NoW **test-split** people (1,702 photos) ×
  {clean, 32, 16, compound, plus mild random degradations}. Evaluation
  uses only the 20 validation people → zero identity overlap, no folds.
- **Runs locally** (CPU/MX330). NoW data stays on the laptop; Kaggle is
  not needed for a network this small.

**Methods compared on the 20 validation people:**

| # | Method | Type |
|---|---|---|
| 1 | Raw degraded | Baseline |
| 2 | Real-ESRGAN → ViT-S | Pixel fix |
| 3 | GFPGAN → ViT-S | Face-pixel fix |
| 4 | Robust fitting (Huber reweight, one refit) | Solver fix |
| 5 | **Prior corrector → ViT-S solver** | Ours |
| — | Oracle clean priors | Upper bound |
| stretch | Feature corrector (GARD idea); corrector + slightly-worse copy input (Noisier2Noise idea) | Ablations |

Headline number: **% of the clean→degraded gap recovered**, plus the
no-harm check on clean photos and added ms / MB.

### E5 — Texture (optional, only after E4)
Project the photo onto the chosen mesh (visible-only UV map). Show
clean vs degraded vs corrected-geometry texture side by side, UV coverage
and SSIM vs the clean texture. One figure, no new model.

---

## 5. Metrics (what each one means)

| Metric | Meaning | Used in |
|---|---|---|
| Scan→mesh mean (mm) ↓ | Average distance from each real scan point to the nearest point on our face after alignment. **Primary.** | E1–E4 |
| Median / p95 (mm) ↓ | Typical point / worst 5 % of points | E1–E4 |
| Penalty (mm) ↓ | degraded − clean, same person, same scan | E2–E4 |
| Gap recovered (%) ↑ | (raw − method) / (raw − clean) | E3, E4 |
| Fit coverage ↑ | meshes produced / attempts (failures count) | E1–E4 |
| Prior error (px, depth units) ↓ | Distance from degraded/corrected prior to clean prior | E3, E4 |
| Time (s), VRAM (MB), GFLOPs ↓ | Cost on the MX330, same scope for every model | E1, E4 |
| UV coverage, SSIM ↑ | Texture completeness / similarity to clean texture | E5 |

---

## 6. Architecture

```text
            photo (clean | degraded)
                    │
     ┌──────────────┼───────────────────────┐
     │ [A] restore  │                       │ other models
     │ Real-ESRGAN/ │                       │ (ViT-B, DECA,
     │ GFPGAN       ▼                       │  Pixel3DMM, MICA)
     │        ViT-S prior network           │
     │              │ priors (UV)           │
     │   [C] prior corrector ◄── trained on │
     │      (analyse s, fix Δ)  80 NoW-test │
     │              ▼            people     │
     │   [D] FLAME Gauss–Newton solver      │
     │              ▼                       ▼
     │        FLAME mesh + camera ──────► neutral OBJ + 7 landmarks
     │                                        │
     │                        NoW scan + landmarks
     │                                        ▼
     └──────────────────────────────►  Evaluator (scan→mesh mm)
                                              │
                              run records ─► notebooks ─► Journal.md
```

### Code: reuse vs add

| Keep as is | Extend | Add |
|---|---|---|
| `data_now.py`, `evaluation.py`, `benchmark.py`, `reporting.py`, `geometry/baselines.py`, `rendering/*`, `ui/` | `robustness/degrade.py` (+`face-16`, `compound`); `geometry/worker.py` (+ `--priors-in` to fit from a given priors file) | `src/correction/cache.py` (run ViT-S once per photo×condition, save priors) · `src/correction/model.py` (UV U-Net) · `src/correction/train.py` · `src/correction/apply.py` · `src/restoration.py` (Real-ESRGAN, GFPGAN wrapper) |

Rule: modules do the work, notebooks only call modules and show results.

---

## 7. Notebooks (final series)

| Notebook | Content | Reuses |
|---|---|---|
| `01_how_reconstruction_works` | One photo → priors → FLAME fit → mesh → GLB → scan error map | current 01 + 04 |
| `02_benchmark_clean` (E1) | 5 models, 20 people, tables, CED, cost | current 02 + 00a |
| `03_benchmark_degraded` (E2) | Conditions, penalty table, per-person plots, galleries | current 03 + 00a |
| `04_oracle_and_correction` (E3, E4) | Oracle bound, training curves, method table, gap recovered | new |
| `05_paper_figures` | Every table/figure used in Journal.md | new |

Old notebooks move to `notebooks/archive/`; nothing is deleted.

---

## 8. Two-day schedule and checklist

Compute estimates on the MX330: four fast models ≈ 90 s per photo-condition;
Pixel3DMM ≈ 5 min; ViT-S fit-from-priors ≈ 10 s; ViT-S prior caching ≈ 1–2 s/image.

**Day 1 (24 Nov) — measure**
- [ ] Scorer anchor: DECA on 20 originals vs published NoW numbers (≈1 h, mostly reuse).
- [ ] Add `face-16`, `compound` to `degrade.py`; unit test.
- [ ] E1 + E2 on 20 people, 4 fast models (80 inputs ≈ 2 h, background).
- [ ] Pixel3DMM on 6-person subset (24 inputs ≈ 2 h, background, after the above).
- [ ] Add `--priors-in` to the ViT-S worker; parity test (re-injecting unchanged priors reproduces the fit).
- [ ] E3 oracle: 20 × 3 × 4 ViT-S fits (≈ 40 min).
- [ ] **Gate decision** written into Journal.md.

**Day 1 night — cache (only if gate passes)**
- [ ] Cache ViT-S priors for NoW-test photos × 5 conditions (subsample to ~600 photos if slow).

**Day 2 (25 Nov) — fix and write**
- [ ] Train corrector (target < 1 h CPU/MX330). Save loss curves.
- [ ] Methods 1–5 + oracle on 20 validation people × 3 conditions.
- [ ] Clean-photo no-harm check, timing.
- [ ] Notebooks 02–05 executed and saved; old ones archived.
- [ ] Journal.md: method, tables, figures, limitations, conclusion.
- [ ] (Optional) E5 texture figure.

**If the gate fails:** Day 2 becomes: finish 20-person benchmark + oracle
analysis + texture/appearance-degradation figure, and write the benchmark
paper. No corrector training.

---

## 9. Out of scope for this sprint

Eyes, wrinkles, lighting, skin-colour correction networks, video, extra
datasets, Kaggle training, official NoW evaluator build. Listed as future work.
