# TrueFace3D Master Plan

Nov 23, 2026 · @Satya Srinivas Paladugu

## Goal

TrueFace3D turns one face photo, good or poor, into a 3D face that is as accurate and realistic as one built from a clean photo.

**The one-line story of the paper:** fix the evidence the 3D model relies on, not the pixels, and prove it against real 3D scans.

**Scope for this round:** the 3D shape, skin color and skin texture layers. Wrinkles, eyes and lighting come later. All evaluation uses the **NoW validation set** (352 photos, 20 people, each with a 3D scan and 7 scan landmarks).

**This document is the single source of truth for Codex.** It replaces the earlier roadmap, implementation plan and the 12-hour report where they disagree.

## Project structure

The project runs in three stages, and each one starts only after the previous gate passes.

&#91;embedded content: project roadmap · 3 stages, 2 gates\]

**Every layer follows the same recipe:** clean photo vs. degraded photo, measure the gap against a reference, fix the evidence that layer depends on, and show the fix beats simply upscaling the photo. The user expects Chapter 1 alone to fill most of the project time.

## Chapter 0 — Benchmark every candidate model

Chapter 0 answers one question per layer: which model is best on clean photos, and how much worse does each get on poor photos? The answer is the **gap table** that Chapter 1 sets out to close.

### Input conditions

Every degraded photo is made from the **same clean NoW photo**, with the same face crop, so only quality changes.

| Condition | How it is made | Status |
| --- | --- | --- |
| Clean | Original NoW photo | Run |
| Low resolution: 128, 64, 32 px face width | Antialiased downsample, then resized back for the model | Run |
| Blur: mild, heavy | Gaussian blur at full resolution | Run |
| Noise: mild, heavy | Gaussian noise, proposed σ = 5 and 15 (0–255 scale) | New |
| JPEG: mild, heavy | Re-save at proposed quality 30 and 10 | New |

### Candidate models per layer

**Shape (3D mesh)**

| Model | What it is | Status |
| --- | --- | --- |
| RealDenseFace ViT-S | Dense priors + FLAME solver; our main host | Runs on MX330 |
| RealDenseFace ViT-B | Same, larger encoder | Runs |
| Pixel3DMM | UV + normal priors + optimisation | Runs, 4–6 min per fit → subset only |
| DECA (coarse) | Direct FLAME regressor | Runs |
| MICA | Neutral identity shape | Runs |
| SMIRK, EMOCA | Extra FLAME regressors | Optional, only if quick to install |

**Skin color (albedo)** — all methods use the **same fixed mesh and camera** (RealDenseFace ViT-S on the clean photo), so color differences come from the color method only.

| Model | What it is | Status |
| --- | --- | --- |
| Raw projection | Photo pixels projected onto the mesh UV map | Baseline, to build |
| DECA albedo | FLAME/BFM albedo model fitted by DECA | To set up |
| TRUST | Skin-tone-aware albedo estimation | Check release, install |
| OpenDelight | Removes lighting from skin color | Check release, install |
| MARCUS | Full skin materials | Optional, likely too heavy for 2 GB |

**Skin texture (pores, marks, detail)** — same fixed mesh and camera.

| Model | What it is | Status |
| --- | --- | --- |
| Raw projection | Photo pixels in UV | Baseline, to build |
| Real-ESRGAN → projection | Upscale photo first | Weights on disk |
| CodeFormer / GFPGAN → projection | Face restoration first | To install |
| OpenDelight / MARCUS texture | Learned texture output | Reuse from color layer where available |

### Metrics

NoW has 3D scans but **no skin-color or texture reference**, so the color and texture gaps are measured against **the same method's result on the clean photo**. That is exactly the gap we want to close.

| Layer | Metric | Meaning |
| --- | --- | --- |
| Shape | Scan-to-mesh error, mm (median / mean / std) | Distance from the real scan to our mesh, official NoW protocol |
| Shape | Gap, mm | Degraded error minus clean error, per person |
| Color | Skin-tone error, ITA degrees | Is the skin as light or dark as in the clean result? |
| Color | Color difference, ΔE | Average color difference in the skin region of the UV map |
| Texture | LPIPS and SSIM vs clean result | How close the detail is to the clean-photo texture |
| Texture | Hallucination score | Detail present in the degraded result but absent in the clean one |
| All | Identity similarity (ArcFace) | Does the render still look like the person? |
| All | Success rate, time, peak VRAM, GFLOPs | Does it work, and what does it cost? |

### Scale and statistics

- **People:** all 20 NoW validation subjects. Start with 2 photos per person (40 photos), expand if time allows.
- **Pixel3DMM:** 1 photo per person × 3 conditions (clean, 64 px, heavy blur), because of its runtime.
- **Statistics:** average per person first, then report the median with a 95% bootstrap interval over people. Compare two methods with a paired Wilcoxon test over the 20 people.
- **Fair timing:** load each model once, time from cropped image to mesh, and report model loading separately.

### Notebooks

1. `00a_shape_benchmark.ipynb` — scan error table, error-vs-severity curve, failure gallery
2. `00b_skin_color_benchmark.ipynb` — color gap table, side-by-side UV maps
3. `00c_texture_benchmark.ipynb` — texture gap and hallucination table, zoomed crops
4. `00d_summary.ipynb` — one gap table for all layers + cost table; picks the host for Chapter 1

## Chapter 1 — Fix the priors

Chapter 1 compares every way to repair a poor photo's 3D evidence, trains the ones that need training, and keeps the one that cuts scan error most on people it never saw. **Host model:** RealDenseFace ViT-S, unless Chapter 0 shows another model is clearly better and still correctable.

&#91;embedded content: where each correction method acts · 4 intervention points\]

The paper's question is simple: **is it better to fix the pixels, the hidden features, the priors, or the fitting?**

### Methods compared

| # | Method | Acts on | Training | Idea from |
| --- | --- | --- | --- | --- |
| 1 | Raw degraded photo | — | None | Baseline |
| 2 | Bicubic upscale | Pixels | None | Baseline |
| 3 | Real-ESRGAN upscale | Pixels | None (pretrained) | Image restoration |
| 4 | CodeFormer / GFPGAN | Pixels | None (pretrained) | Face restoration |
| 5 | Robust fitting | Solver weights | None | Robust statistics |
| 6 | **GARD-style Feature Corrector** | Encoder features | Small adapter | GARD |
| 7 | **Prior Corrector** | Priors | Small network | Our main method |
| 8 | Prior Corrector + slightly-worse copy | Priors | Small network | Noisier2Noise |
| 9 | Prior Corrector + trust output | Priors + solver weights | Small network | MonoCal |
| 10 | Best combination (e.g. 6 + 9) | Features + priors | Both | — |
| — | Oracle: clean-photo priors | Priors | — | Upper bound only, not a method |

### About GARD

The published GARD cannot be run as-is: it needs about 10 views of the scene, a Depth Anything 3 backbone and a 50-step diffusion denoiser. Method 6 keeps **its core idea** — clean the network's hidden features before the output heads — in a single-image, one-step version:

- A small adapter sits after a middle layer of the frozen ViT encoder.
- It is trained so the features of a degraded photo match the features of its clean version.
- The rest of RealDenseFace and the solver stay unchanged.

This gives the paper a fair **features vs. priors** comparison. If the feature fix wins, it becomes the method; the paper still holds either way.

### Our Prior Corrector, in plain words

- **Input:** the priors from the degraded photo (where each face point lands, its depth, and its uncertainty), plus the face width in pixels.
- **Output:** corrected priors, and in version 9 a trust value per point so the solver leans on the FLAME face model where the photo is too poor.
- **Version 8:** also sees the priors from a slightly worse copy of the same photo; the difference shows which way the priors are drifting.
- **Size:** a small network on the flat UV map (target under 2 M parameters), so it trains on the MX330.

### Training

- **Data:** pairs we make ourselves — clean photo and degraded copy. The target is the clean photo's priors. **No scans are used for training.**
- **Split by person:** 5 folds over the 20 NoW people (train on 16, test on 4, rotate), so no person is ever in both train and test.
- **More data (optional):** extra face photos without scans, such as FFHQ, after checking the license.
- **Cache first:** run the frozen ViT once per photo and condition, save priors (and, for method 6, the middle-layer features), then train on the saved files.
- **Degradations seen in training:** low resolution and blur. Noise and JPEG are held out to test generalization.

### How the winner is chosen

1. **Main score:** scan error on unseen people at 64 px face width, averaged per person.
2. **Also reported:** 128 and 32 px, blur, and the held-out noise and JPEG conditions.
3. **No harm on clean photos:** the method must not make clean-photo error noticeably worse.
4. **Cost:** added time and memory on the MX330.
5. **Statistics:** paired Wilcoxon test over the 20 people against the best no-training method (2–5).

The winner is the method with the lowest degraded scan error that passes 3. The headline result is how much of the gap it closes: the share of the clean-to-degraded gap from Chapter 0 that the method recovers.

## Later chapters

Later chapters reuse the Chapter 1 recipe on one layer each, starting only after Chapter 1 is finished.

| Chapter | Layer | Fix (same recipe) | Reference |
| --- | --- | --- | --- |
| 2 | Skin color | Color Corrector on the UV color map, with trust values | Clean-photo result; FAIR skin-tone benchmark if accessible |
| 3 | Skin texture | Texture Corrector that adds detail only where the photo supports it | Clean-photo result; hallucination score |
| 4 | Wrinkles | Detail Corrector on DECA/HRN detail maps | FaceScape detail maps, if accessible |
| 5 | Eyes | Fit the eyeball only when the iris is visible enough | Synthetic faces with known gaze |
| 6 | Lighting | One renderer for new views and lights | Demo only, no claim |

**One corrector per layer, not one big network.** Each layer has a different reference, each chapter must show what its own fix did, and small networks fit the MX330. A shared network can be a follow-up once each corrector works.

## Codebase, notebooks and rules for Codex

Extend the existing repository; do not restart it. Existing files such as `src/benchmark.py`, `src/evaluation.py`, `src/reporting.py`, `src/now_official.py`, `src/robustness/degrade.py` and `src/geometry/profile_flops.py` stay and are extended.

### New modules

```text
src/
├── robustness/degrade.py        # add noise and JPEG conditions
├── appearance/
│   ├── projection.py            # photo → UV map with visibility mask
│   ├── backends/                # deca_albedo, trust, opendelight, restoration
│   └── metrics.py               # ITA, ΔE, LPIPS, SSIM, hallucination, ArcFace
├── correction/
│   ├── splits.py                # 5 person-level folds over NoW
│   ├── cache.py                 # save priors + middle-layer features per photo
│   ├── robust_fit.py            # method 5
│   ├── feature_corrector.py     # method 6 (GARD-style)
│   ├── prior_corrector.py       # methods 7–9
│   └── train.py                 # one trainer for 6–9
└── reporting.py                 # gap tables, severity curves, galleries

notebooks/
├── 00a_shape_benchmark.ipynb
├── 00b_skin_color_benchmark.ipynb
├── 00c_texture_benchmark.ipynb
├── 00d_summary.ipynb
├── 01a_correction_methods.ipynb   # methods 1–5, no training
├── 01b_correctors_training.ipynb  # methods 6–10, loss curves per fold
└── 01c_chapter1_results.ipynb     # final table, gap recovered, examples
```

### Rules

- **Modular:** each model and each correction method is its own module behind one common interface; notebooks only call modules.
- **Notebooks run locally and are saved executed,** so every result is visible when opened. Each notebook explains every step in plain words.
- **Terminal first:** print every command run, and give the user the same commands to rerun.
- **Teach as you go:** after each milestone, explain what was done, show the outputs (tables, images, meshes), and list what to check.
- **Every run is recorded:** model, condition, photo, person, fold, time, memory, success or failure, and a run ID.
- **No person in both training and testing.** Folds are fixed once in `splits.py`.
- **Same crop for clean and degraded copies** of a photo.
- **Journal:** update `Journal.md` after each milestone with numbers, figures and commands.
- **Git:** never add co-author lines to commits.

## Execution order and known fixes

Work top to bottom; each step ends with a check the user can see.

1. **Trust the scorer.** Try the official NoW evaluator through its Docker setup, time-boxed to 1 hour. Otherwise keep the local scorer and confirm it on a few cases. Sanity check: DECA coarse on NoW validation should land near the published 1.18 / 1.46 / 1.25 mm (median / mean / std).
2. **Finish Chapter 0 shape.** Score the 50 saved meshes, then run all 20 people under every condition. Done when `00a` shows the gap table and severity curve.
3. **Chapter 0 color and texture.** Build raw UV projection, install the color and texture candidates that run on the MX330, fill `00b` and `00c`.
4. **Summary.** `00d` gives one gap table for all three layers and confirms the Chapter 1 host.
5. **Chapter 1, no-training methods (1–5).** Fill `01a`.
6. **Cache and train (6–10).** Cache priors and features, train per fold, fill `01b`.
7. **Results.** `01c` gives the final table, gap recovered, clean-photo check, cost and examples. Update the journal.

### Known problems to fix on the way

- **Timing:** models were timed over different stages. Re-time all of them from cropped image to mesh, model loading reported separately.
- **Precision:** the MX330 runs RealDenseFace in FP32 instead of the original bf16. Record this; check one photo against a bf16 run if a bigger GPU is ever available.
- **Old pilot:** the 0.823 / 0.090 direction result came from one non-NoW photo with a centre crop and an already-upscaled input. Treat it as a hint only; method 8 retests the idea properly.
- **Too few people:** earlier degradation results covered 1–3 people. Nothing is concluded until all 20 are run.
- **Overlap:** RealDenseFace was trained partly on FaceScape. State this in the paper; NoW results are unaffected.
- **Scope:** skin color and texture in Chapter 0 are benchmarks only. Their fixes wait for Chapter 2 and 3.
