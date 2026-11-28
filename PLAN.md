# TrueFace3D: final plan (FaceScape-C benchmark on real photos → GARD-Face correction)

**Status:** for your review, 28 Nov 2026 (revision 2). Nothing below has been run yet. This file replaces
`docs/PROPOSAL_2026-11-27.md` (kept as a record). Items marked *hypothesis* are unverified.

**Title:** *Degradation-robust monocular 3D face reconstruction: a corruption benchmark on real FaceScape photos
(FaceScape-C) and geometry-aware representation denoising (GARD-Face).*

**Revision 2 (28 Nov, your decision): real photos only.** Every image that a model sees, for testing or for
training, is a **real camera photo** from FaceScape's Multi-View data. No face is rendered from a scan to make
an input. Rendering is used only to *display* meshes (clay renders, error heatmaps). Subjects 801–847 have no
photos, so they leave the benchmark. The official FS-Lab images are renders (checked: scan on a black
background), so they are dropped too.

**Scope (28 Nov, your decision):** the core is the **mesh + appearance** degradation benchmark and GARD-Face.
**Wrinkles and eyes are stretch goals**, done only if time remains (DECA/EMOCA/HRN detail meshes are still saved,
so wrinkle metrics can be added later without rerunning). The main grid is reduced (§7).

**Data status (checked 28 Nov):**
- **Have:**
  - TU-Model **301–400** (15.2 GB, complete; all of 321–359 have 20 meshes each);
  - TU-Model 801–847 (not used for inputs);
  - `info_list_v2.txt`;
  - `publishable_list_v1.txt` = **122, 212, 340, 344, 393, 395, 421, 527, 594, 610**;
  - `publishable_nomasaic_tex.zip`;
  - the toolkit, including `predef/Rt_scale_dict.json`, which covers all 847 subjects;
  - SVFR benchmark data.
- **To download:** Multi-View photos `fsmview_trainset_images_321-340.zip` and `…_341-359.zip`, about 28 GB each.
- **Optional:** the matching multi-view mesh archives, if the folder has them.

**Time:** 2.5 working days (Sat 28 → Mon 30 Nov), with Tue 1 – Wed 2 Dec as buffer.
**Fallback rule (yours):** if the FaceScape pilot is not producing correctly scored meshes by
**Sun 29 Nov, 14:00**, we switch to NoW (also real photos), which already works end to end.

---

## 0. Ground rules

1. **Nothing is lost.** No reset, clean, stash-drop or force-push. The untracked local results, the
   `ubuntu-handoff` branch on GitHub (uv.lock + stripped showcase notebook) and the CRLF-only diffs all stay
   untouched. New work goes on a new branch `facescape-c` cut from `master`.
2. **Reuse first.** Extend the existing modules (§10); write new files only where nothing exists.
3. **Pretrained, openly available models only.** No fine-tuning and no training from scratch, with two
   exceptions: GARD-Face and the prior corrector, which are our methods. Every model must have public code and
   weights (free academic registration, as for FLAME or EMOCA, is acceptable).
4. **Real photos only.** Model inputs are real camera photos. Degradations are applied to those photos.
5. Commits are made as you, with **no AI attribution** (`docs/HANDOFF.md`). No licensed data or weights go into
   git.
6. **Only measured results are reported.** Every reported number is printed by an executed notebook cell, with
   all of its metrics visible. No number lives only in a JSON file.

---

## 1. Your hypothesis, checked against the papers

**Your claim:** DECA and MICA barely react to degradation because they shrink the face to a small input at the
start. High-resolution models (RealDenseFace ViT-S/B) use a high-quality input to produce a mesh close to the
true one, so they have more to lose.

### 1.1 What each model actually reads

| Model | Input the network sees | Backbone | Output geometry | Wrinkle capacity |
|---|---|---|---|---|
| RealDenseFace (2026) | **512×512 crop**, 32×32 tokens | frozen **DINOv3** ViT-S+/16 or ViT-B/16 + trained decoder | FLAME (5,023 vertices) fitted to dense per-vertex 2D + depth priors | none (FLAME has no displacement) |
| DECA (2021) | **224×224 crop** | ResNet-50 | FLAME coarse + 256×256 UV **displacement** | yes (detail branch) |
| EMOCA v2 (2022) | 224×224 (DECA architecture) | ResNet-50 + emotion loss | FLAME + DECA detail | yes |
| MICA (2022) | **112×112 aligned crop** (ArcFace) | ArcFace ResNet-100, pretrained on Glint360K, last 3 blocks refined | FLAME **neutral identity only** | none |
| SMIRK (2024) | 224×224 (to verify in code) | 3 × MobileNetV3, 3.6 M parameters; identity distilled from MICA | FLAME expression + eyelid closure | none |
| HRN (2023) | 224×224 (Deep3D face analyzer) | ResNet-50 + 2 pix2pix | BFM coarse + **64×64 deformation map** + **256×256 displacement map** | yes (strongest open model) |

### 1.2 Evidence

- **Confirmed: higher-resolution dense models are more accurate on good photos, mainly for expression and
  pose.** RealDenseFace paper, NeRSemble official benchmark, L1 Chamfer in mm (lower is better):
  - *posed (expressive) faces:* RealDenseFace 1.610, Pixel3DMM 1.659, SMIRK 2.276, DECA 2.385, EMOCA 2.636;
  - *extended posed benchmark:* RealDenseFace 1.56 vs SMIRK 2.49 (≈37 % lower error);
  - *neutral identity:* the gap is small: RealDenseFace 1.637 vs MICA 1.680 (2.6 %).

  So your "X → close to X2" holds for **expression and pose**, and only marginally for neutral identity.
- **Confirmed: the low-resolution, identity-centred design is what keeps MICA stable.** MICA's own ablation on
  Stirling compares low-quality vs high-quality photos: median error 1.25 vs 1.18 mm with a frozen ArcFace (a
  6 % gap). ArcFace is trained to ignore illumination, expression, rotation and occlusion.
- **Correction 1: input size alone does not explain the stability.** On NoW, 64, 32 and 16 px faces are all
  smaller than DECA's 224 px input, yet DECA moved only 1.477 → 1.530 mm. The second reason is the
  **low-dimensional, strongly regularised output**: when the image carries little information, a regressor
  falls back towards the mean face. **Stable is not the same as accurate:** a model that always outputs the
  mean FLAME face is perfectly robust and always wrong. We therefore add the mean face as a floor baseline and
  measure how far each model's output moves away from it (identity specificity).
- **Correction 2: our NoW run was not the paper's configuration.** RealDenseFace's paper starts the
  single-image fit from **MICA's identity**. The released `fit_single_image.py` (pinned commit `7d855a2`) calls
  `optimizer.fit_frame(inference, camera)` without it, so identity starts at zero. The paper's own ablation:
  removing MICA initialisation raises neutral error from 1.86 to 2.09 mm (+12 %). This very likely explains why
  our ViT-S (1.430 mm) lost to MICA (1.168 mm) on clean NoW photos, while the paper reports RealDenseFace ahead of
  MICA on NoW (0.82 vs 0.90 median). The optimizer already accepts `init_identity`.
- **Correction 3: "higher-quality mesh" never includes wrinkles for RealDenseFace.** Its output is a FLAME mesh.
  Wrinkles come only from DECA/EMOCA displacement (256×256 UV) and HRN (deformation + displacement maps).

### 1.3 Consequences for the study

- **H1 (restated):** models that read the face at high resolution (RealDenseFace) are more accurate on clean
  photos, especially for expressions, and lose that advantage faster under degradation than 112–224 px
  regressors. The crossover point (the degradation level where the old model becomes better) is the key result.
- RealDenseFace runs in two configurations: **paper-faithful (MICA identity initialisation)** as the main one,
  and zero initialisation (what we ran on NoW) as an ablation. *Hypothesis:* MICA initialisation also makes
  RealDenseFace more robust, because the identity starts from a stable source.
- Gate: confirm that MICA's FLAME 2020 shape coefficients are valid for RealDenseFace's FLAME 2023 identity space
  (compare the shape bases) before using them.

---

## 2. What to download

The source is the FaceScape Google Drive share (`facescape_share5`). **Save on `/mnt/DataBunker`** (456 GB free),
not in `~/Downloads`: the home drive has 134 GB free, which is too little for the zips plus their extracted
contents.

| # | Item | Size | Why | Status |
|---|---|---|---|---|
| D1 | Information list, publishable list, license PDF | small | ages/genders; who may be shown; terms | have |
| D2 | GitHub repo `zhuhao-nju/facescape` | small | camera parser, undistortion, alignment table `Rt_scale_dict.json`, facial mask, landmark indices | have |
| D3 | **TU-Model 301–400** | 15.2 GB | **ground-truth 3D** (mesh in mm + 4K texture + displacement) for every photo subject 321–359 | have |
| D4 | **Multi-View `fsmview_trainset_images_341-359.zip`** | ≈28 GB | real photos: 19 people incl. **344 (publishable)** | **must** |
| D5 | **Multi-View `fsmview_trainset_images_321-340.zip`** | ≈28 GB | real photos: 20 people incl. **340 (publishable)** | **must** |
| D6 | Multi-View mesh archives for 321–359, if present lower in the same folder | unknown | meshes already in each photo's camera frame; checks the "minor misalignment" the toolkit warns about | recommended |
| D7 | Toolkit sample (`samples/download_sample.sh`): subject 212, anger, photos + `params.json` + `.ply` | small | 212 is publishable; lets me build and test the photo pipeline **before** D4/D5 finish | must (I fetch it) |
| D8 | SVFR benchmark data | 207 MB | FS-Wild: 400 real web photos with GT meshes. Used as an optional outside check, after confirming how its GT was made. **FS-Lab is dropped** (renders) | have |

**What the Multi-View photos are** (from `doc_mview_model.md`): for each subject × expression, about 60 real
photos from a calibrated camera rig (4–12 MP), with each camera's intrinsics `K`, extrinsics `Rt`, lens
distortion and a `valid` flag in `params.json`. Images are undistorted with the given parameters before use.
Views with `valid = False` are ignored.

**Kaggle upload** (private, owned by account A, B added as collaborator; never public):
- **Dataset:** `satyapaladugu/facescape-c`, built locally by `python -m src.facescape.build` (revision 3, 28 Nov).
  It contains every clean crop, every degraded image, cameras, GT and manifests, about 7 GB. Nothing is
  generated on Kaggle.
- **Per person × expression:** 3 real photos (front, left45, right45), each with `clean.png` and `camera.json`.
- **Degraded, test people:** the full grid (42 conditions) for the 4 test expressions; the pilot (344) gets the
  grid for all 20.
- **Degraded, training people:** 3 random (profile, severity) draws per photo, all views mixed.
- **GT:** TU vertices (mm) per person × expression; TU texture for test people.
- **Provenance:** every file is recorded with its SHA-256; `build_info.json` records the commit.

---

## 3. Data design

### 3.1 Subjects (FaceScape 321–359, real photos)

From `info_list_v2.txt`:
- 39 subjects, aged 19–66 (20 female, 19 male).
- **323 is flagged invalid** (texture and displacement missing), so **38 are usable**.
- Older subjects, useful for wrinkles: 350 (f, 66), 328 (f, 50), 331 (f, 46), 356 (f, 42), 345 (f, 38), 358 (f, 37).

| Set | Subjects | Expressions | Purpose |
|---|---|---|---|
| **Pilot** | **344** (f, 25, publishable: every image may be shown) | **all 20** | first end-to-end run; alignment and unit gates; ranks expressions by degradation sensitivity |
| **Test** | 6 people: **340** (f, 22), **344** (f, 25), **350** (f, 66), **325** (m, 23), **339** (m, 30), **352** (m, 32) | see §7 | the reported benchmark; 3 female / 3 male, ages 22–66, both publishable subjects |
| **Train pool** (GARD-Face, prior corrector) | the other **32** (321, 322, 324, 326–338, 341–343, 345–349, 351, 353–359; 323 excluded) | all 20, ~10 views each | training only; person-disjoint from every reported test number |
| Outside check (optional) | SVFR FS-Wild, 400 real photos | as provided | generalisation, official CD / MNE / CR |

**Only 340 and 344 may appear in the report or slides** (photos, textures, overlays). Other subjects appear only
as numbers or as grey clay renders of meshes, and only in local notebooks.

**Expressions and wrinkles.** This was measured on all displacement maps of 801–847 (46 subjects). It describes
how FaceScape expressions create detail, so it guides the choice of expressions. Rerun on 321–359 in
notebook 30. The table gives the median per-subject change in displacement magnitude relative to the same
person's neutral:

| Expression | forehead | glabella (frown) | crow's feet | under-eye | nasolabial |
|---|---|---|---|---|---|
| smile | +1 % | −1 % | +1 % | +2 % | **+11 %** |
| mouth_stretch | +3 % | +1 % | +3 % | +2 % | 0 % |
| anger | +1 % | **+8 %** | +1 % | +2 % | +5 % |
| dimpler | −2 % | −1 % | −2 % | +2 % | **+17 %** |
| eye_closed | −3 % | −2 % | −1 % | −1 % | −5 % |
| brow_raiser | **+10 %** | −3 % | −2 % | −4 % | −2 % |
| brow_lower | +5 % | **+21 %** | +1 % | +3 % | +5 % |

### 3.2 From a real photo to a scored test case

1. **Pick views.** The TU frame faces +Z. Each camera's viewing direction is mapped into that frame with
   `Rt_scale_dict`, which gives its yaw and pitch relative to the face.
   - **Three views per expression, for test and training alike:** the valid, level (|pitch| ≤ 15°) camera
     closest to 0°, +45° and −45° yaw, within ±15°. Measured on the rig: front −2 to −6°; right45 −41 to −46°;
     left45 37–52°.
   - GARD-Face trains on all three views mixed.
2. **Undistort** the photo with its `K` and distortion coefficients (OpenCV, as in `demo_mview_projection`).
3. **Ground truth in the photo's camera.** Load the TU mesh (millimetres, canonical frame) and invert the
   per-subject, per-expression scale `s` and transform `Rt_tu` to reach the multi-view frame:
   `X_mv = R_tuᵀ(X_tu − t_tu) / s`. Then apply the camera's `Rt`. The scale `s` keeps every error in mm.
4. **Crop.** A square 768 px crop around the projected face, with the jaw-to-jaw width at **480 px**. A
   tighter frame (1.6 × jaw width) fits the portrait 12 MP photos without black padding.
   - If the face is near the photo border, the window slides inside the photo rather than padding.
   - Every source face is 760–1,750 px wide, so crops are always downscales. This undistorted, resized real photo is the **clean reference**. Each case
   saves:
   - the crop transform and the GT mesh in the crop's camera;
   - the facial mask from the toolkit;
   - subject, expression, view, age and gender.
5. **Gates before any benchmark:**
   - (a) the GT mesh outline overlaid on the photo lines up, checked visually on the pilot and on 212's sample;
   - (b) the inter-pupil distance in mm is plausible (SVFR uses 62.85 mm as its reference);
   - (c) TU-to-photo alignment is measured against the multi-view scan. **Measured:** the toolkit placement is
     off by 1.6–2.9 mm and 0.5–2°. Rigid ICP to the scan (`refine` stage) leaves a 0.23–0.24 mm median
     residual, so `camera.json` stores the refined pose when the scan archive is present.
   - (d) the source face width in pixels is ≥ 480, so the clean reference is never upscaled.
6. Eyes and the inside of the mouth are real in the photos. The TU mesh has holes there, so the scoring mask
   excludes them (eye-region metrics use the eyelid landmarks instead).

---

## 4. Degradation profiles (FaceScape-C): 10 core + 4 restoration-aligned, 3 severities each

All profiles are applied to the **real clean photo crop**. Severities are set relative to face width
(fw = 512 px), so every profile means the same thing at every scale. Seeds are fixed; each spec is saved in the
run record. Synthetic degradation of real photos is the standard protocol (ImageNet-C, and the training data of
Real-ESRGAN, GFPGAN and CodeFormer). The faces themselves stay real.

| ID | Capture scenario | Operations | S1 / S2 / S3 |
|---|---|---|---|
| C01 | Distant subject (low resolution) | box-downsample the whole frame to a face width | fw 64 / 32 / 16 px (same as NoW) |
| C02 | Out of focus | disk (defocus) blur | radius 0.6 / 1.2 / 2.4 % of fw |
| C03 | Camera shake | linear motion blur, seeded angle | length 1.5 / 3 / 5 % of fw |
| C04 | High-ISO sensor noise | Poisson (shot) + Gaussian (read) + chroma noise | low / mid / high gain |
| C05 | Night / low light | exposure × 0.35 / 0.2 / 0.1, gamma, amplified shot + read noise, mild JPEG | — |
| C06 | Harsh flash / over-exposure | gain 1.6 / 2.2 / 3.0 with clipping, highlight bloom, contrast loss | — |
| C07 | Compression | JPEG | q 30 / 15 / 5 |
| C08 | Messaging-app chain | (downscale + JPEG) twice | ×0.5 q70 / ×0.35 q50 / ×0.25 q30 |
| C09 | Cheap webcam / old phone | downscale + chromatic aberration + vignetting + noise + over-sharpening + JPEG | mild / mid / severe |
| C10 | Beauty / skin-smoothing filter | edge-preserving smoothing + slight brightening | low / mid / high (**removes wrinkles**) |
| R1 | Classical blind-face-restoration model (GFPGAN/CodeFormer training) | Gaussian blur → downsample → Gaussian noise → JPEG | sampled from CodeFormer's ranges (σ 1–15, r 1–30, δ 0–20, q 30–90) at the 25th / 50th / 75th percentile |
| R2 | Real-ESRGAN second-order model | the exact `train_realesrgan_x4plus.yml` recipe (two rounds of blur/resize/noise/JPEG + sinc) | 3 fixed seeds ordered by severity |
| R3 | Haze (DA-CLIP type) | atmospheric scattering | 3 densities |
| R4 | Shadow (DA-CLIP type) | hard cast shadow across part of the face | 3 strengths |

That is 14 profiles × 3 severities + clean = **43 conditions**. Restoration-aligned profiles R1–R4 let us test
the restorers on the degradations they were trained for, and also on ones they were not (C05, C06, C10).
Input quality is recorded per image: PSNR/SSIM to the clean photo, plus no-reference NIQE and MUSIQ.

---

## 5. Models (all public; 6 mesh models; 5 appearance methods; 5 restorers)

### 5.1 Mesh: RealDenseFace + the five strongest from the last five years

| # | Model | Year | Why it is in | Access | Risk on the Kaggle image (py3.13, torch 2.11, T4) |
|---|---|---|---|---|---|
| M1 | **RealDenseFace ViT-S** (main) + ViT-B; counts as one model | 2026 | high-resolution dense priors; correction host | have | none (already runs) |
| M2 | **DECA** (coarse + detail) | 2021 | the classic; wrinkles via displacement | have | none |
| M3 | **MICA** | 2022 | most stable neutral identity; also the init for M1 | have | none |
| M4 | **EMOCA v2** (coarse + detail) | 2022 | emotion and expression; DECA detail | `EMOCA_v2_lr_mse_20.zip` (380 MB, after EMOCA registration). The `EMOCA-emorec` zips are emotion classifiers, not this model | medium (old PyTorch Lightning, PyTorch3D) |
| M5 | **SMIRK** | 2024 | best expression accuracy, eyelid closure | GitHub + Google Drive weights | medium (PyTorch3D, MediaPipe) |
| M6 | **HRN** | 2023 | strongest open wrinkle model; also gives texture | GitHub + Google Drive assets, which already include the processed BFM (`BFM_model_front.mat`); no separate Basel registration | medium (nvdiffrast is already built) |

Each new model gets a time-boxed setup (2 hours on Kaggle readiness). If one fails, its replacement is
**3DDFA-V3** (2024).

**Pixel3DMM is dropped.** It ranks second on NeRSemble (1.659 mm) but took 151 s per fit in our Kaggle runs and
failed on 78 of 200 images. It would replace HRN only on your request.

**Upstream exposure (stated in the report):** all test photos come from FaceScape subjects ≤ 359.
- **Exposed:** RealDenseFace (244K FaceScape samples) and HRN trained on that multi-view subset; the exact subject
  lists are not published.
- **To check:** MICA's training mix. SMIRK inherits MICA's identity.
- **Not exposed:** DECA and EMOCA (trained on 2D photo collections).

Clean-photo accuracy of exposed models may therefore be optimistic. The main quantity, **change under
degradation on the same photo**, is much less affected. Each table marks exposed models.

### 5.2 Appearance (top 5 that work on our frames)

| # | Method | Year | Output | Status |
|---|---|---|---|---|
| A1 | Raw projection onto the predicted mesh | — | photo texture in UV (floor; keeps the degradation) | have |
| A2 | DECA albedo + detail-shaded texture | 2021 | FLAME UV albedo | have |
| A3 | TRUST | 2022 | lighting-aware albedo, fair skin tone | have |
| A4 | OpenDelight | 2025 | high-detail delit albedo | have |
| A5 | HRN de-retouched texture | 2023 | BFM UV texture with its detailed mesh | comes with M6 |

EMOCA shares DECA's albedo and SMIRK and MICA predict no texture, so they are not separate appearance methods.
Appearance GT is the subject's TU texture (captured under the rig's lighting). The other reference is the clean
photo re-projected onto the GT mesh.

### 5.3 Pixel restorers (pretrained; the controls GARD-Face must beat)

| # | Restorer | Year | Paradigm | Trained on |
|---|---|---|---|---|
| P1 | Real-ESRGAN x4plus (your ESRGAN line) | 2021 | GAN SR, generic | second-order synthetic degradation (= R2) |
| P2 | GFPGAN v1.4 | 2021 | StyleGAN2 face prior + eye/mouth component losses | classical face degradation (≈ R1) |
| P3 | CodeFormer (w = 0.5) | 2022 | VQ codebook + transformer; w trades fidelity vs quality | blur σ 1–15, down 1–30×, noise 0–20, JPEG 30–90 |
| P4 | DiffBIR v2.1 (face mode) | 2024 | diffusion restoration (strong hallucination) | Real-ESRGAN/CodeFormer-style synthetic |
| P5 | DA-CLIP (universal) | 2024 | degradation-aware all-in-one | 10 real degradation sets (motion blur, haze, JPEG, low light, noise, raindrop, rain, shadow, snow, inpainting; to confirm in the paper) |

Diffusion restorers (P4, P5) are slow, so they run on a subset (§7).

**Download links** (checked 28 Nov):

| Model | Link |
|---|---|
| EMOCA v2 | `https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip` |
| HRN (code) | `https://github.com/youngLBW/HRN` |
| HRN (assets) | Google Drive folder `1qI3-5GKxEDbQBSUCwDWmH15uWU3Ka5LS` |
| GFPGAN | `https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth` |
| CodeFormer | `https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth` |
| DiffBIR | `https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt`. Its script fetches Stable Diffusion 2.1 by itself |

---

## 6. Metrics (final list, with sources)

All mesh metrics are computed after aligning the prediction to GT inside the facial mask (eyes and mouth
interior excluded). **Main protocol:** landmark similarity + masked ICP (NoW protocol; scores shape only).
**Secondary:** depth-only alignment in the real camera (FaceScape SVFR protocol; also scores pose and scale).
Units: mm.

| Group | Metric | Measures | From |
|---|---|---|---|
| Shape accuracy | scan-to-mesh distance: mean, median, std, p95 | overall shape | NoW (RingNet 2019) |
| | Chamfer distance L1/L2 (both directions) | overall shape | FaceScape SVFR; NeRSemble SVFR |
| | normal consistency (NC) / mean normal error (MNE) | mid- and fine-scale surface | NeRSemble; FaceScape |
| | recall @ 2.5 mm (R2.5), F-score @ 1 / 2.5 mm | share of the surface that is close | Pixel3DMM; RealDenseFace |
| | complete rate (CR) | coverage of the GT face | FaceScape SVFR |
| | region errors: forehead, eyes, nose, mouth, cheeks, chin | where the damage is | REALY (2022) region-wise idea |
| | max error | perceptually relevant extremes | SMIRK (MultiFace evaluation) |
| | success rate | fits that produced a mesh | FaceScape SVFR |
| | distance to mean face | identity specificity (stable ≠ accurate) | ours |
| Expression | per-expression accuracy (all of the above, per expression) | does degradation hurt expressive faces more | ours on FaceScape |
| | expression-induced error = error(expression) − error(neutral) | cost of the expression itself | ours |
| | expression measures vs GT: mouth opening, smile width, brow height, eyelid aperture (mm) | interpretable expression error | landmark measures (EMOCA/SMIRK eye-closure and lip motivation) |
| | 2D landmark NME on the input image | alignment | standard (68 points) |
| Wrinkles (stretch) | normal error inside wrinkle masks (forehead, glabella, crow's feet, nasolabial) | wrinkle geometry | ours, using GT TU mesh + displacement |
| | high-frequency energy ratio (prediction / GT) in masks | smoothing (< 1) or hallucination (> 1) | ours |
| | wrinkle-line precision / recall / F1 (ridge detection) | right wrinkles in the right place | ours, inspired by wrinkle-segmentation metrics |
| Eyes (stretch) | eyelid aperture error; open/closed accuracy on `eye_closed` | eyelid geometry | ours |
| | eye-crop PSNR / LPIPS against the clean photo | eye appearance | appearance metrics |
| Appearance | PSNR, SSIM, LPIPS, DISTS | pixel / perceptual fidelity | standard; CodeFormer/GFPGAN use PSNR, SSIM, LPIPS |
| | ΔE2000 | colour error | existing `metrics.py` |
| | ITA difference | skin-tone shift | TRUST |
| | ArcFace cosine | identity kept | CodeFormer (IDS); existing |
| | UV coverage | how much texture was recovered | existing |
| | stability: degraded vs the same method's clean result | robustness without GT | our NoW protocol |
| Input quality | PSNR, SSIM to clean; NIQE, MUSIQ (no reference) | how bad each profile is | restoration literature |
| Robustness | Δ = degraded − clean (paired); relative Δ % | damage | — |
| | slope over severity; area under the error–severity curve | how fast it breaks | — |
| | **Corruption Error** CE = Σₛ E(model,s) / Σₛ E(DECA,s); **mCE** = mean over profiles; **relative mCE** uses Δ instead of E | ImageNet-C style ranking with DECA as reference | Hendrycks & Dietterich, ICLR 2019 |
| | crossover severity (where an old model beats the new one) | tests H1 | ours |
| Correction | gap recovered = (E_deg − E_method) / (E_deg − E_clean) | how much of the damage is undone | existing `gap_table` (bug A5 fixed) |
| | clean-photo harm (method applied to clean images) | does it break good photos | ours |
| Statistics | paired Wilcoxon, person-level bootstrap 95 % CI, Holm correction, n always shown | significance | — |

---

## 7. Experiments, in order

**E0, one face end to end (your request).** Subject **344** × 20 expressions × 3 views:
1. Show the data: real photo, camera, GT mesh overlaid on the photo, texture, displacement (notebook 30).
2. Run all 6 mesh models on the clean photos → compare with GT (all metrics).
3. Apply the 14 profiles × 3 severities → rerun → change vs GT **and** vs each model's own clean result.

2,580 images (60 clean + 2,520 degraded), about 8 GPU-hours with persistent workers (estimate).
Option to save time: run degraded E0 on the front view only (860 images, about 3 GPU-hours). Gate: alignment, units and errors look sane, visually checked.

**E1, main benchmark (reduced).** 6 test subjects × 4 expressions × 3 views × 43 conditions = **3,096 images**
× 6 mesh models:
- expressions: neutral, smile, brow_raiser, mouth_stretch;
- conditions: clean + 10 core profiles × 3 severities.

Additions:
- Appearance methods on clean + S2.
- Oracle check for RealDenseFace on the same set.

About 10 GPU-hours for the 6 mesh models with persistent workers (estimate, about 12 s per image for all six).
The three views triple the earlier 1-view grid. If time runs short, the side views drop to the 10 core
profiles at S2.

**E2, decision gate (written down before training).** The correction target is the model × signal where:
(a) the degradation effect is large and significant; (b) the model has an intermediate representation we can
correct (RealDenseFace priors/features); (c) the oracle recovers ≥ 40 % of the gap. Expected:
RealDenseFace ViT-S. If (c) fails, report that clearly and correct features rather than priors.

**E3, corrections** (all evaluated on the E1 test set, person-disjoint from training):

| ID | Method | Acts on | Trained? |
|---|---|---|---|
| B | baseline (degraded input) | — | — |
| P1–P5 | restorer → mesh and appearance models | pixels | pretrained |
| **G** | **GARD-Face** | RealDenseFace encoder tokens | yes (ours) |
| C | prior corrector v2 | RealDenseFace per-vertex priors | yes (ours; v1 code reused) |
| G+P | best restorer + GARD-Face | both | — |
| O | oracle (clean features / clean priors) | upper bound, not a method | — |

Restorer reruns are limited to 5 subjects × 4 expressions × 14 profiles at S2 (280 images × 5 restorers).

---

## 8. GARD-Face (main method) and the prior corrector

**GARD** (Kim et al., KAIST/Samsung, arXiv 2605.26230, May 2026) keeps a feed-forward 3D model frozen and
denoises its **intermediate features** with a flow-matching transformer that starts from the degraded feature,
not from noise. The restored features feed the original geometry decoder.

**GARD-Face (single image, RealDenseFace only):**
- Host: RealDenseFace's frozen DINOv3 encoder (ViT-S+/16: 12 blocks, 384-d, 32×32 tokens at 512 px). The decoder
  reads blocks **2, 5, 8 and 11**, so the denoiser sits **after block 1**. All four tapped features are then
  downstream of the fix, as in GARD (K=18, taps 20–40).
- Denoiser: a small DiT-style transformer (about 4 + 4 blocks, 384-d, ≈15 M parameters) with a timestep
  embedding.
- Training: *interpolated flow matching* as in GARD: z_t = (1−t)(z_deg + αε) + t·z_clean, target velocity
  z_clean − (z_deg + αε), α = 0.3. Plus our addition: a **prior-consistency loss**, where restored tokens are
  passed through the frozen rest of RealDenseFace and the decoded priors must match the clean priors (replacing
  GARD's multi-view attention-alignment loss).
  - *Option, if time allows:* FaceScape gives several real views of the same expression at the same instant, so
    GARD's multi-view alignment idea becomes possible.
- Inference: 1–4 Euler steps (GARD uses 50) so it runs fast on a T4.
- **Data: real photos of the 32 train-pool people.**
  - Inputs: 20 expressions × about 10 valid views within ±45° yaw, about 6,400 clean photos.
  - Each step draws a random profile and severity.
  - Clean tokens come from the same frozen encoder on the clean photo. No 3D labels are needed for training;
    GT meshes are used only for scoring.
- Ablations: steps 1 vs 4; with/without the prior loss; denoiser position (after block 1 vs on the four taps);
  clean-photo harm.
- Hypothesis: GARD-Face beats every pixel restorer on mesh metrics because it fixes what the geometry decoder
  reads, without hallucinating pixels.

**Prior corrector v2:** reuse `src/correction/prior_corrector.py`. Changes:
- fix the gap-table pairing (A5);
- replace the face-width gate with a degradation-aware code;
- retrain on train-pool priors from real photos.

**Appearance correction** uses the pixel restorers (Real-ESRGAN line) before the appearance methods, as you asked.
UV-ESRGAN, my earlier idea, **is dropped**: it would have been a new network trained by us.

---

## 9. Compute and Kaggle

- Accounts: A `satyapaladugu`, B `satyapaladuguplus`; each runs up to 2 GPU sessions (T4 ×2) and has
  30 GPU-hours this week: **60 hours in total**. Check on the quota page whether a T4 ×2 session uses quota at 1×
  and when the weekly reset falls.
- CLI on this laptop: one `kaggle` install, two settings folders (`~/.config/kaggle/a`, `~/.config/kaggle/b`),
  and two commands `kaggle-a` / `kaggle-b` that set `KAGGLE_CONFIG_DIR`. You place the tokens; I never touch them.
- Throughput fix: a **persistent worker** per model per GPU (load once, fit many). Today every ViT fit starts a
  new process: 27.5 s wall time vs 2.9 s of work.
- Photo preparation (unzip, view selection, undistortion, GT transform, crops) runs on this laptop's CPU and
  uses no GPU quota.

**Measured per-image times from our earlier Kaggle runs (T4):**

| Model | Wall time per image | Model compute | Notes |
|---|---|---|---|
| RealDenseFace ViT-S | 27.5 s | 2.9 s | |
| RealDenseFace ViT-B | 32.4 s | 3.0 s | |
| DECA mesh | 6.7 s | 0.4 s | |
| MICA | 5.3 s | 0.5 s | |
| DECA texture | median 8.5 s | — | |
| TRUST | median 14–17 s | — | |
| OpenDelight | median 43 s | — | |

EMOCA, SMIRK and HRN are not measured yet; they are timed on the first 20 images.

| Job | Account | GPU-hours (estimate) |
|---|---|---|
| Readiness + new-model setup (EMOCA, SMIRK, HRN) | A + B | 3 |
| E0 pilot (3 views) | A + B | 8 |
| E1 mesh: RealDenseFace (+ oracle), 3 views | A | 6 |
| E1 mesh: DECA, MICA, EMOCA, SMIRK, HRN, 3 views | B | 6 |
| E1 appearance | B | 6 |
| Restorers + reruns on the subset | B | 9 |
| GARD-Face: token extraction for the train pool + train + eval + ablations | A | 9 |
| Prior corrector v2: collect + train + eval | A | 6 |
| **Total** | | **≈ 53 of 60** (tight: first cut is E0 degraded on side views) |

If time runs short, cut in this order: ViT-B beyond S2 → prior corrector v2 → DiffBIR → wrinkle-line F1.
Scoring runs on CPU (in parallel inside the sessions and on this laptop); it does not use GPU quota.

---

## 10. Code: reuse vs new

**Reused (extended, not rewritten):**
- `benchmark.py` (run records, matrix) and `evaluation.py` (similarity + ICP, closest-surface);
- `appearance/metrics.py`, `appearance/projection.py`, and the appearance backends (DECA, TRUST, OpenDelight);
- `geometry/worker.py`, `baselines.py`, `baseline_worker.py`, `oracle.py`;
- `correction/prior_corrector.py` and `robustness/degrade.py`;
- `kaggle/setup.py`, `runner.py`, `staging.py`, `export.py`;
- `analysis.py`, `showcase.py`, `rendering/gallery.py` (display only);
- the notebook-builder pattern (`notebooks/make_*.py`).

**New:**
- `src/data_facescape.py`: multi-view cases; `params.json` parsing, undistortion, view selection, the
  TU-to-camera GT transform, crops, masks, metadata. It reuses the toolkit's conventions.
- `src/robustness/profiles.py`: the 14 profiles.
- Adapters for EMOCA, SMIRK and HRN, and a persistent-worker mode.
- `src/metrics_mesh.py`: Chamfer, NC/MNE, R2.5, F-score, CR, regions, expression and eyelid measures.
- `src/metrics_detail.py`: wrinkle-mask normal error, HF ratio, ridge F1.
- `src/correction/gard_face.py` and `scripts/kaggle_fleet.py`.
- Unit tests for each. The GT-transform test uses the toolkit sample (subject 212).

**Store everything the corrections need during the benchmark:** every RealDenseFace fit saves two things:
- its priors (`fit/priors.npz`, already done by the existing worker);
- its encoder tokens after block 1 (≈0.8 MB per image in fp16).

The oracle, the prior corrector and GARD-Face evaluation then reuse them: only the cheap FLAME solve is rerun,
never the full benchmark.

Results format: every run writes per-image records (as now); a collector builds **one tidy table per benchmark**
(image × model × condition × metric) that all notebooks read.

---

## 11. Notebooks

**Kaggle notebooks (they produce data):**
- K00 readiness (A and B);
- K01 degrade;
- K10 mesh A, K11 mesh B, K12 appearance B;
- K20 restorers B;
- K30 GARD-Face train/eval A, K31 prior corrector A.

Each asserts its key step (builds, data found, smoke fit scored) before spending GPU hours.

**Local notebooks (presentation; executed, outputs saved):**

| Notebook | Shows |
|---|---|
| 30_facescape_dataset | what FaceScape contains and what we downloaded; real multi-view photos, cameras, undistortion; TU mesh, texture, displacement; GT overlaid on the photo (subject 344); the 20 expressions; wrinkle statistics on 321–359; subsets and the person-disjoint split |
| 31_degradation_profiles | every profile × severity on subject 344's photo; input-quality table; which restorer was trained on which profile |
| 32_mesh_benchmark | E0 + E1: faculty-showcase grids (input strip, clay renders, error heatmaps per model × condition), every metric table, mCE ranking, crossover plot (H1), per-expression heatmaps |
| 33_appearance_benchmark | textures and renders per method × condition vs GT; all appearance metrics |
| 34_wrinkles_and_eyes | (stretch) wrinkle masks, normal-error maps, HF ratio, eyelid measures, eye crops |
| 35_restoration_controls | P1–P5: restored inputs and their effect on mesh and appearance |
| 36_gard_face | the method: design, training curves, before/after meshes, gap recovered, ablations |
| 37_prior_corrector | v1 → v2, results |
| 38_corrections_all | all methods side by side in one place: the final tables |
| 39_final_showcase | the faculty-facing story, shortest path through the evidence |

Rules for every notebook:
- **Framing:** a plain-language "what you will learn / how to read this" at the top and "what this shows" at the
  bottom.
- **Metrics:** all of them together (mean, median, p95, CI, n, success rate), and a coverage table whenever
  something failed.
- **Figures:** in the style of `20_faculty_showcase.ipynb`, plus error-coloured 3D meshes (plotly), rank
  heatmaps, robustness curves with CIs, slope graphs and before/after sliders.
- **Privacy:** photos and textures shown only for 340 and 344. Notebooks showing anyone else's face stay local
  and are never committed.

---

## 12. Journal.md

`Journal.md` is already IEEE-style. After E0 I rewrite Abstract, Introduction, Related Work and Method for this
plan; after every experiment its tables and figures go into Results, each citing the notebook that printed it.
The dated log moves to an appendix.

---

## 13. Schedule

| When | Me | You |
|---|---|---|
| **Sat 28, now → night** | Linux env (uv), Kaggle wrappers, branch; `data_facescape.py` built and tested on the toolkit sample (subject 212) while D4/D5 download; `profiles.py` + tests; adapters for EMOCA, SMIRK, HRN; K00 readiness | Kaggle tokens; download D4, D5 (and D6 if present) to `/mnt/DataBunker`; EMOCA v2 zip; HRN assets folder |
| **Sun 29, morning** | unzip + prepare 321–359 crops; GT-overlay and unit gates on 344; notebooks 30, 31 executed; upload the private dataset; E0 pilot on Kaggle (A: RealDenseFace, B: the others); notebook 32 (pilot part) | review the pilot figures |
| **Sun 29, 14:00 checkpoint** | go on FaceScape, or switch to NoW (your rule) | decide |
| **Sun 29, afternoon → night** | E1 main benchmark on A + B; restorers; GARD-Face code; train-pool token extraction; oracle | — |
| **Mon 30, morning** | E2 decision gate; GARD-Face + prior corrector training; restorer reruns | approve the gate decision |
| **Mon 30, afternoon** | correction evaluation; notebooks 33–38 executed; Journal update | review |
| **Tue 1 – Wed 2** | buffer, reruns, notebook 39, report polish | presentation prep |

---

## 14. Main risks

| Risk | Mitigation |
|---|---|
| The 56 GB download is not done by Sunday morning | the pipeline is built on the 212 sample first; start with whichever zip finishes first (341–359 holds pilot 344 and two more test subjects) |
| TU-to-photo alignment is off ("minor misalignment" per the toolkit) | overlay gate on the pilot; measure against the multi-view mesh (D6); the NoW-style ICP protocol absorbs rigid offsets |
| Upstream exposure (RealDenseFace, HRN trained on FaceScape photos) | marked in every table; the headline is change under degradation; FS-Wild as an outside check; stated as a limitation |
| EMOCA/SMIRK/HRN do not build on the Kaggle image | 2-hour time-box each, then 3DDFA-V3 |
| Topologies differ (FLAME, BFM, TU) | all metrics are surface-to-surface or landmark-based; no vertex correspondence needed |
| GPU quota | persistent workers; budget table; cut list |
| GARD-Face does not beat restorers | reported honestly; the benchmark (FaceScape-C, mCE, crossover) stands as a contribution on its own |

---

## 15. What I need from you

1. Your review of this revision (especially §3 test subjects and §3.2 pipeline).
2. Downloads to `/mnt/DataBunker`:
   - `fsmview_trainset_images_341-359.zip` and `fsmview_trainset_images_321-340.zip`;
   - the matching multi-view mesh archives, if the folder lists them (tell me their names and sizes).
3. `EMOCA_v2_lr_mse_20.zip` (link in §5.3) and the HRN assets folder.
4. Kaggle: tokens placed as in §9.
