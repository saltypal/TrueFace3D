# Comparisons before model commitment

All numbers below are **published author results**, transcribed from archived original PDFs. They are not TrueFace3D measurements. PDF page numbers are one-based local PDF pages, not always printed proceedings pages. Original hashes and extraction files are recorded in the [archive manifest](../evidence/download_manifest.json). No cross-paper row is treated as a controlled experiment unless its protocol is known to match.

## C1: geometry competitors and evidence

Use RealDenseFace as the first reproduction candidate and Pixel3DMM as the primary same-task dense-prior challenger. MICA is a neutral identity/scale reference, not a complete posed reconstruction competitor. DECA/EMOCA/SMIRK give feed-forward expression alternatives. FlowFace is sequence-oriented in its original contribution; only an explicitly single-image configuration belongs in the equal-input table. Pix2NPHM is a separate richer-representation comparison with different attachment/export work.

From [RealDenseFace](../pdfs/01_RealDenseFace.pdf), PDF p8, Table 2, neutral NoW error, lower is better:

| Method, as listed by these authors | Median, mm | Mean, mm | Std, mm |
|---|---:|---:|---:|
| MICA | 0.90 | 1.11 | 0.92 |
| FlowFace | 0.87 | 1.07 | 0.88 |
| Pixel3DMM | 0.87 | 1.07 | 0.89 |
| RealDenseFace | 0.82 | 1.02 | 0.84 |
| TokenFace, contextual row outside our 44-paper original audit | 0.76 | 0.95 | 0.82 |

This supports RealDenseFace over the listed dense-prior alternatives on that table. It does not establish universal best neutral geometry. TokenFace's row is secondary evidence reported by RealDenseFace, not our independent audit of TokenFace's original experiments.

The same paper, p8 Table 1, official NeRSemble SVFR, reports posed L1/L2/normal consistency of **1.610 / 1.086 / 0.8857** for RealDenseFace and **1.659 / 1.118 / 0.8847** for Pixel3DMM. Preserve the benchmark's exact L1/L2 definitions; do not assume L2 means squared Chamfer. The official scans are not released; reproducing our own COLMAP subset would be a different experiment. Do not relabel its extended 1,150-scan experiment as the official 391-scan benchmark.

[MICA](../pdfs/03_MICA.pdf), p9 Table 2, explicitly separates scale-invariant NoW evaluation from metrical evaluation. It reports MICA mean **1.11 mm non-metrical** and **1.37 mm metrical**, versus DECA **1.38** and **1.80**. Mixing those two columns would invalidate the comparison. The [official NoW protocol](https://now.is.tue.mpg.de/) distinguishes the two tracks and limits test submissions. Use permitted validation data for development; do not use the test server as a tuning loop.

RealDenseFace p9 reports about **83.96 FPS** on a 1,862-frame online tracking sequence on RTX 4090. This is neither a single-image benchmark nor the cost of albedo/detail/eyes/Cycles. Pixel3DMM's sequence runtime in that table must not be compared with a different device's isolated network forward pass.

**Common failure to test:** identity/expression ambiguity under pose/occlusion. Pixel3DMM p7 discusses that better posed metrics do not necessarily translate to neutral identity. Our bounded visibility/robust-fitting experiment must retain both tracks and an unchanged baseline.

## C2: albedo and materials

First compare raw projected RGB, a simple declared diffuse-only fitting baseline, and OpenDelight. If executable assets fit the resource budget, add one richer predictor among Relightify, ID2Reflectance or MARCUS. These tasks differ: image albedo, UV completion and a joint PBR bundle need separate metric columns. MoSAR, AvatarMe++, ReflectanceMM and S3-Face are important comparison/design sources even when full runtime assets are unavailable.

From [OpenDelight](../pdfs/12_OpenDelight.pdf), p7 Table 1:

| Author row | FaceOLAT PSNR / SSIM / LPIPS | 3DRFE PSNR / SSIM / LPIPS |
|---|---|---|
| SwitchLight | 32.98 / 0.9800 / 0.1031 | 31.08 / 0.9744 / 0.1032 |
| OpenDelight | 32.88 / 0.9817 / 0.0952 | 34.27 / 0.9803 / 0.0649 |
| OpenDelight* | 31.93 / 0.9809 / 0.1072 | 35.02 / 0.9812 / 0.0606 |

The paper uses a skin mask and estimates a **six-parameter channel-wise scale/bias per image and per method against GT** before scoring. These are color-aligned metrics. Our raw-color metrics need a separate column; GT-derived alignment is evaluator-only and cannot become inference input. The paper describes OpenDelight* as its open-source variant. The current downloaded-checkpoint provenance must be resolved from pinned release documentation, not inferred from the table name. SwitchLight is a proprietary paper comparator; no paid service is assumed in the schedule.

[Relightify](../pdfs/15_Relightify.pdf), p8 Table 2, on six measured subjects, reports diffuse/specular/normal-map PSNR **22.47 / 27.17 / 26.69 dB**, versus AvatarMe++ **18.30 / 19.77 / 27.26**. This is a different dataset/task from OpenDelight's image-albedo evaluation. Its normal-map PSNR is not a substitute for angular normal error. Six subjects do not justify a broad population claim.

**Common failures:** baked illumination, loss of identity-sensitive marks, ambiguous absolute color and hallucinated hidden UV regions. OpenDelight p13 §7.4 explicitly discusses potential identity loss. Visibility-aware projection is basic correctness; testing a conservative visible-detail residual is an engineering hypothesis, not automatically new. Richer material predictors must be compared as coherent bundles before mixing independently selected maps.

## C3: surface detail

Compare coarse mesh only, original DECA detail, restrained detail transfer and HRN's native detail output. HiFace and ReEnFP inform the mechanisms; their execution eligibility follows current asset checks. EMOCA is a useful expression comparator but is not an independent new displacement architecture over DECA.

[HRN](../pdfs/09_HRN.pdf), p7 Table 1, reports FaceScape-wild CD/MNE **2.91 mm / 0.065 rad**, compared with DECA **3.31 / 0.089**. On FaceScape-lab, HRN reports **3.67 / 0.087** and DECA **4.69 / 0.108**. Native HRN-versus-DECA differences include coarse geometry and training. They do not isolate the effect of transferring HRN detail onto RealDenseFace.

[HiFace](../pdfs/10_HiFace.pdf), p6 Table 1, separates coarse and detailed reconstructions and explicitly notes that image-supervised detail can worsen geometric accuracy. [DECA](../pdfs/07_DECA.pdf), p9 §7, states albedo-space limits, facial-hair confusion, extreme-pose/light/occlusion failure and noisy detail from low-resolution training. A prettier shaded wrinkle is not proof of improved depth.

**Common failure to test:** shading or hair explained as displacement and detail failing to follow expression. Evaluate on/off under the same coarse surface and shader, with bounded amplitude and held-out geometry/normal evidence where available. If no detailed scan exists, record only rendering consistency and synthetic recovery, not real personal pore accuracy.

## C4: eyes

Compare a single opaque painted eyeball fixture with explicit sclera/cornea/iris/pupil geometry; then add visible-evidence fitting. The [2016 parametric-eye paper](../pdfs/22_ParametricEyes.pdf), §§4–8, supports statistical/procedural anatomy and single-image fitting. Its complete captured database/code is not verified, so an exact pretrained baseline cannot be promised. The [2014 capture paper](../pdfs/21_EyeCapture.pdf) needs richer acquisition. [DeNeRF](../pdfs/23_EyeNeRF.pdf) uses richer person-specific capture and a neural representation. Those results are not equal-input single-photo baselines.

**Common limitation:** low-resolution/occluded eyes leave anatomy and gaze uncertain. Explicit optics addresses a rendering limitation of the painted sphere, but generic cornea/iris detail remains a prior. Synthetic camera/geometry fixtures can validate fitting and optics; a portrait alone cannot validate anatomical truth.

## C5: fitting and C6: rendering

nvdiffrast and Cycles solve different tasks, so a single “winner” score is inappropriate. Compare no refinement versus staged bounded refinement within one differentiable backend, then check that scene semantics transfer into Cycles. Compare diffuse-only versus Principled skin with SSS on the same mesh/maps/light. Use Mitsuba 3 as a later physically differentiable alternative, with the archived Mitsuba 2 paper as design context rather than an API specification.

**Common limitations:** local rasterized lighting omits some transport; unrestricted photometric optimization lets shape/material/light absorb one another's errors; exported viewers may omit optics/displacement. Shared scene assembly and a staged optimizer are established principles. Any proposed improvement must beat fixed assembly and unweighted/robust fitting under matched inputs and cost.

## Runtime comparison budget and dispositions

Every component receives a literature comparison. Initially budget two principal executable methods/configurations per learned component, not 44 full reproductions. Record unavailable, incompatible, input-ineligible, too expensive, and dominated candidates separately. Do not describe a paper with private data or missing weights as scientifically weak.

For an eligible method, acquire permitted artifacts, pin a revision and reproduce the author's intended baseline before adapter changes. Compare native outputs first; compare transferred attributes on a fixed common scene second. A difference caused by a new crop, camera, mesh, shader, checkpoint or extra inference input cannot be attributed solely to our proposed fix.
