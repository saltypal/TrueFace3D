# TrueFace3D staged plan (geometry → appearance → wrinkles)

Status date: 2026-11-25. Faculty approved the direction. This file is the canonical research plan; it
supersedes the structure of the local (untracked) `CHAPTER_ROADMAP.md`. Operating instructions for any new
machine or chat are in `docs/HANDOFF.md`.

Everything marked *candidate* or *hypothesis* is unverified. Do not report it as a result.

---

## 0. One-paragraph story

Modern dense face reconstructors (RealDenseFace ViT-S/ViT-B, Pixel3DMM) read the face at high resolution and
produce detailed meshes on good photos, but they lose accuracy quickly when the photo is small, noisy, blurred,
compressed or badly lit. Older FLAME regressors (DECA, MICA) shrink the face to 224 px (MICA's identity
encoder to 112 px), so degradation removes little that they use: they are stable but coarse. We measure this
trade-off on FaceScape with real ground truth, then correct the dense model's intermediate per-vertex priors
with a small degradation-conditioned network (GARD-inspired; GARD is cited for the idea, not reproduced),
and compare against pixel restoration (Real-ESRGAN, GFPGAN, CodeFormer). Stage 2 repeats the
benchmark-plus-correction for appearance (texture/albedo). Stage 3 targets wrinkle detail under low quality.

## 1. What is already measured (NoW, 20 people, mean scan error in mm)

| Model | Original | 64 px | 32 px | 16 px | Compound |
|---|---|---|---|---|---|
| ViT-S (RealDenseFace) | 1.430 | 1.445 | 1.599 | 2.302 | 2.149 |
| ViT-B (RealDenseFace) | 1.497 | 1.533 | 1.657 | 2.092 | 1.994 |
| DECA | 1.477 | 1.485 | 1.462 | 1.530 | 1.509 |
| MICA | 1.168 | 1.177 | 1.195 | 1.314 | 1.241 |
| Pixel3DMM (n = 15/18/15/1/12) | 1.159 | 1.154 | 1.176 | 1.125 | 1.304 |

- ViT-S at 16 px: 20/20 people worse, paired Wilcoxon p ≈ 2e-6. MICA at 16 px: +0.15 mm, p = 0.024.
- Pixel3DMM's own face crop fails on most small faces (1 of 20 at 16 px).
- Appearance on NoW (no ground truth, compared with the same method on the clean photo), at 16 px:
  ArcFace identity similarity TRUST 0.89, DECA texture 0.66, OpenDelight 0.59. TRUST is the most stable.
- Prior corrector v1 (NoW, 5 person-disjoint folds): local CPU dry run reduced prior coordinate error by
  21/18/5/10 % at 64/32/16/compound. **The Kaggle refit that measures scan-error gap recovered
  (`notebooks/kaggle/13_kaggle_prior_corrector.ipynb`, account A) has not returned yet.**

**Important caveat for the story.** On NoW, MICA (2022) is more accurate than ViT-S even on original photos,
and DECA is only 0.05 mm worse than ViT-S. So "older models always give poorer meshes" is **not** supported by
mean scan error on NoW. NoW scores a neutral, masked, ICP-aligned region and rewards overall shape, not detail.
FaceScape lets us test the claim properly with detail-sensitive metrics (normal error, per-region error,
displacement/wrinkle error, expressions). State the trade-off as hypothesis H1 below and let the FaceScape
numbers decide; if MICA stays best, the paper says so and the story becomes "detail vs robustness", not
"new vs old".

## 2. FaceScape: what to use and why

Drive folder contents: Information (16 KB), Parametric Model (15.4 GB), TU-Model (120 GB, 9 files),
Multi-View Data (689 GB, 36 files). Plus the public GitHub repo (toolkit, SVFR benchmark).

| Part | What it is | Use it for | Priority |
|---|---|---|---|
| **Information** (`info_list_v2.txt`) | gender / age of 847 subjects | stratify results by age (wrinkles!), balanced test split | **must, tiny** |
| **SVFR benchmark** (GitHub `benchmark/` + its data download) | standard single-view reconstruction benchmark with eval code (lab images with pose/focal variation and in-the-wild images; confirm details in its README) | comparability with published numbers; second geometry protocol beside our NoW-style ICP score | **must** |
| **Multi-View Data** | 359 ids × 20 expressions × ≈60 calibrated views (4–12 MP) + a 3D mesh per expression | **real photos with a ground-truth mesh and known camera**: the Stage 1 test inputs and the corrector's training images | **must, subset only** |
| **TU-Model** | 847 ids × 20 exp, topologically uniform meshes + 4K displacement map + 4K texture map | Stage 2 texture ground truth; Stage 3 **wrinkle ground truth** (displacement maps); optional synthetic photos by rendering | Stage 2/3, subset only |
| **Parametric (bilinear) model** | statistical base-shape model, 4 variants | optional extra baseline / fitting tool; it has no detail | low |
| Toolkit (GitHub `toolkit/`) | demos: multi-view projection, landmarks, facial mask, render, alignment, symmetry, rig | reuse the facial mask, landmark indices, projection and render code instead of writing our own | **must read** |

### Recommended subset (download on the second laptop, never the whole 689 GB)

1. Download Information, the License PDF and `facescape_google_drive.md` first.
2. Download **one** Multi-View file and **one** TU-Model file; list their contents and record the real
   folder layout in `docs/facescape_layout.md` before writing any loader. Do not guess paths.
3. Test set: **30 subjects × 3 expressions × 3 views**.
   - Expressions (confirm names in the files): 1 neutral, 2 smile, and a wrinkle-rich one
     (19 brow raiser for forehead wrinkles, or 16 grin). Neutral keeps NoW comparability; the others test
     expression and give Stage 3 something to measure.
   - Views: frontal, about ±30° yaw. Use the provided camera parameters to project the GT mesh into each
     view and to build the face box.
   - Balance age and gender from the info list; include enough subjects over ~45 for wrinkles.
   - For any figure in a paper, use only subjects the license allows to be shown (check the license and
     FaceScape docs for the publishable-ID list).
4. Training pool for the corrector: different subjects (person-disjoint), e.g. 100–200 ids, all 20
   expressions, many views. The corrector needs (degraded photo → clean-photo priors) pairs, so training
   images do not need GT meshes; they only need to be real faces.
5. Package as **private** Kaggle datasets in small parts (images + cameras + GT meshes for the test set; images
   only for training). Check the Kaggle dataset size limits before uploading. Never make them public and never
   share the Drive link (license terms).

## 3. Stage 1 — geometry: benchmark, trade-off, prior correction

### Research questions and hypotheses

- **H1 (trade-off).** High-resolution dense models are more accurate on clean photos in detail-sensitive
  metrics, and degrade faster than 224-px FLAME regressors. Measured by: clean accuracy vs degradation slope
  per model, per region, per metric.
- **H2 (where the damage is).** Degradation hurts the dense model's per-vertex priors (2D coordinates and
  depth), and swapping in clean priors (oracle) recovers most of the scan-error gap. Gate: oracle recovers
  ≥ 40 % → prior correction is the method; < 20 % → move the correction to features / crop.
- **H3 (correction).** A small degradation-conditioned prior corrector recovers a significant part of the gap,
  beats or complements pixel restoration, and does not hurt clean photos.
- **H4 (generality).** The same idea transfers to another dense-prior model (Pixel3DMM's predicted
  UV/normal maps) and across datasets (train FaceScape → test NoW).

### Degradations (extend `src/robustness/degrade.py`)

Keep the existing ones (face width 64/32/16, compound = 48 px + blur + noise + JPEG). Add, each at 3
severities, deterministic seeds, spec saved in `record.json`:

| Type | Severities (starting point, tune on 3 photos) |
|---|---|
| Gaussian blur | σ = 1.5, 3, 5 px at face width 460 px (scale with face size) |
| Motion blur | length 7, 15, 25 px, random angle (seeded) |
| Sensor noise | Gaussian σ = 10, 25, 45 (0–255) |
| JPEG | quality 30, 15, 5 |
| Low light | brightness × 0.35 / 0.2 / 0.1 with gamma, then shot + read noise (realistic dim photo) |
| Over-exposure / contrast | gain 1.8 / 2.5 with clipping; contrast 0.5 / 0.3 |

Training uses a random second-order pipeline (Real-ESRGAN style) so the corrector sees mixtures; evaluation
uses the fixed grid above so results are reproducible.

### Models

| Model | Status | Why |
|---|---|---|
| RealDenseFace ViT-S, ViT-B | have | main dense model; correction host (ViT-S default) |
| Pixel3DMM | have (crop fails on small faces) | second dense-prior model; H4 |
| DECA | have | old 224-px FLAME baseline |
| MICA | have | strongest stable baseline on NoW |
| EMOCA v2 | *candidate* | DECA successor, expression-aware, same dependencies |
| SMIRK (2024) | *candidate* | recent FLAME regressor, expression accuracy |
| TEASER (2025) | *candidate* | recent FLAME + detail tokens |
| HRN (2023) | *candidate* | hierarchical coarse + detail; useful again in Stage 3 |
| Deep3DFaceRecon | *candidate* | classic BFM regressor; topology-agnostic scoring covers BFM |
| 3DDFA-V2 | *candidate* | fast old baseline |
| FaceScape DFDN (riggable detail) | *candidate* | dataset authors' detail model; Stage 3 |

Before adding a candidate: confirm public code + weights + license, check whether it was trained on FaceScape
(contamination → report separately), and pass the gate in §6. Add one model at a time.

### Correction methods (all compared on the same images)

| ID | Method |
|---|---|
| B | baseline model on the degraded photo |
| R1–R3 | restoration first (Real-ESRGAN, GFPGAN, CodeFormer), then the model |
| O | oracle: clean-photo priors substituted (upper bound; coordinates mapped between face boxes) |
| C1 | prior corrector v1 (current NoW model, quality gate on face width) |
| C2 | prior corrector v2: conditioned on a predicted degradation embedding (small blind estimator on the crop), trained on FaceScape mixtures |
| C2+R | restoration + C2 |
| C2-P | C2 applied to Pixel3DMM's priors (H4) |

Ablations for C2: no conditioning; no gate; coordinates only vs depth only; training set size; per-degradation
training vs mixture.

### Metrics

- NoW-style masked ICP mean scan error (existing `local_now_mask_similarity_icp`) — continuity with NoW.
- FaceScape SVFR benchmark metrics with its own code — comparability.
- Per-region error (forehead, eyes, nose, mouth, cheeks) and normal error — detail sensitivity for H1.
- Gap recovered = (B − method) / (B − clean), with paired Wilcoxon tests and person-level bootstrap CIs.

### Job budget (estimate; measure on Kaggle before committing)

Test set 90 images (30 × 3 exp × 1 frontal view first; side views later). Conditions: original + 3 width +
compound + 6 types × 3 severities = 23. Full grid = 2,070 images per model.

- ViT-S and ViT-B: full grid. Other models: original + width 3 + compound + the middle severity of each new
  type = 11 conditions (990 images each).
- Kaggle gives two accounts, about 30 GPU-hours per week each (check the current quota), 2 GPU sessions
  per account. Read per-fit times from the NoW runs and size chunks so one notebook finishes in < 11 h.

### Stage 1 steps

1. **S1.0 Data intake (second laptop).** Download the subset, write `docs/facescape_layout.md`,
   `src/data_facescape.py` (cases with image, camera, face box, GT mesh, mask, expression, age, gender), and
   `scripts/facescape_prepare.py` (select subjects, crop/resize originals, write degraded variants, package
   Kaggle datasets). Unit test the loader on 1 subject.
2. **S1.1 Protocol port.** Score one subject × 5 current models on Kaggle; visually check GT mesh projected
   onto the photo and the predicted mesh after alignment. Gate: alignment and units correct (mm), errors in
   a sane range.
3. **S1.2 Degradation suite.** Add the new types with tests (determinism, severity monotonic in PSNR).
4. **S1.3 FaceScape benchmark run** with the current 5 models (Kaggle A + B in parallel).
5. **S1.4 Oracle on FaceScape** for ViT-S (H2 gate), and read the pending NoW corrector result.
6. **S1.5 Model expansion** (EMOCA v2, SMIRK first; others after).
7. **S1.6 Corrector v2** trained on FaceScape training pool (Kaggle), evaluated on FaceScape test + NoW.
8. **S1.7 Restoration controls and ablations; Stage 1 write-up** (figures: trade-off plot of clean accuracy vs
   degradation slope, per-region heatmaps, gap-recovered table, mesh galleries).

## 4. Stage 2 — appearance: benchmark and correction

- **Ground truth.** FaceScape TU texture maps (4K) and the multi-view photos. Two scoring routes:
  (a) render each method's textured mesh into a held-out real view and compare with the real photo (masked
  PSNR/SSIM/LPIPS/ΔE2000/ArcFace); (b) UV-space comparison after registering each method's topology to the TU
  topology (harder; do (a) first).
- **Models.** Have: DECA texture/albedo, TRUST, OpenDelight. *Candidates* (verify code/weights/license):
  FFHQ-UV, HRN texture, NextFace (optimisation, CPU-capable), Relightify, FitMe, AlbedoGAN.
- **Degradations.** Same suite as Stage 1; low light and colour cast matter most for albedo.
- **Correction.** Appearance prior corrector: correct the method's intermediate appearance representation
  (albedo coefficients / UV albedo map / lighting code) conditioned on degradation; compare with GFPGAN /
  CodeFormer restoration first. Also test whether corrected geometry from Stage 1 improves texture alone.
- **Questions.** Which methods keep identity and skin tone under each degradation? Is albedo vs lighting
  separation what breaks in low light? Does geometry correction carry over to appearance?

## 5. Stage 3 — wrinkles under low quality

- **Ground truth.** TU displacement maps (4K) per subject and expression. Wrinkle-rich expressions
  (brow raiser, smile/grin) and older subjects (info list) give the signal.
- **Models with detail.** DECA detail, EMOCA detail, HRN, FaceScape DFDN, TEASER (if it predicts detail),
  Pixel3DMM normals. Most others have none — that gap is the contribution space.
- **Metrics.** Displacement / normal error inside wrinkle masks (forehead, crow's feet, nasolabial,
  frown lines); high-frequency energy ratio vs GT; wrinkle-line precision/recall from ridge detection;
  age-stratified results. Guard against hallucination: a plausible wrinkle in the wrong place must score worse
  than no wrinkle, so use subject-specific GT, never only realism scores.
- **Method idea (hypothesis).** A degradation-conditioned detail enhancer that predicts a displacement residual
  from the degraded crop + coarse corrected mesh, trained with TU displacement GT on the training pool,
  evaluated on held-out subjects. Compare with running detail models on restored photos.

## 6. Gates and rules (apply to every stage)

- New model gate: runs on the Kaggle image, 3-photo smoke test scored, mesh visually checked, runtime/memory
  recorded, license and training-data exposure noted.
- Data gate: loader unit tested; GT projected onto a photo and checked visually; units in mm.
- Training gate: person-disjoint splits fixed in a committed file; seeds, configs, data hashes, Kaggle
  kernel version and git commit recorded; results persisted as a Kaggle output and downloaded.
- Notebook gate: every notebook asserts its key step worked (builds, data found, smoke fit scored, enough fits)
  before spending GPU hours on the rest.
- Reporting: only measured numbers; illustrative arithmetic is never evidence; failures and partial coverage
  are reported with counts.

## 7. Compute split

- **Kaggle (main):** all GPU inference, refits and training. The CUDA pieces — nvdiffrast, RealDenseFace's
  FLAME solver extension and PyTorch3D — are custom-compiled for the Kaggle image (torch 2.11 + cu130,
  Python 3.13, T4) and cached as wheels; see `docs/HANDOFF.md`.
- **Second laptop (1 TB, integrated graphics, no CUDA):** FaceScape download and subsetting, degradation
  generation, packaging Kaggle datasets, CPU scoring of downloaded meshes, figures and notebooks. It cannot
  run nvdiffrast, the FLAME solver CUDA extension or PyTorch3D GPU code.
- **Main laptop:** local checks and the faculty notebook; holds the NoW outputs.

## 8. Immediate next actions

1. Read the prior-corrector result when `out_PC.zip` from Kaggle A arrives (gap-recovered table).
2. Second laptop: clone the repo, read `docs/HANDOFF.md`, download FaceScape Information + 1 Multi-View file +
   1 TU file + SVFR benchmark, write `docs/facescape_layout.md`.
3. Write `src/data_facescape.py` and `scripts/facescape_prepare.py` against the real layout.
4. Extend the degradation suite with tests.
5. Run S1.1 (one subject, five models) on Kaggle.
