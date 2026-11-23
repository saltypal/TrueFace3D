# 30 — NeRSemble: multi-camera videos and dynamic radiance fields

**SIGGRAPH 2023; arXiv 2305.03027.** Snapshot: 2026-10-10. [PDF](../pdfs/30_NeRSemble.pdf), [paper](https://arxiv.org/abs/2305.03027), [author project](https://tobias-kirschstein.github.io/nersemble/), [current v2 data utilities](https://github.com/tobias-kirschstein/nersemble-data), [separate single-view reconstruction benchmark](https://kaldir.vc.cit.tum.de/nersemble_benchmark/). Dataset: PDF pp. 3–4; architecture: pp. 5–6.

## Intuition

A video of a talking head is a moving cloud of tiny samples that absorb light and emit color toward a camera. A radiance field learns that cloud. NeRSemble first moves samples into a common, nearly still head coordinate system, then blends several learned feature grids to explain the remaining motion.

Think of transparent building blocks plus a few mixable descriptions of expression. This representation can show hair and subtle motion without requiring every detail to sit on FLAME triangles. That flexibility also means it is not directly a portable mesh with skin materials.

## Architecture

```text
calibrated synchronized multi-view video
 -> camera rays -> sample point x and timestamp t
 -> deformation MLP D(x, omega_t) -> canonical point x'
 -> N multi-resolution hash grids H_i(x')
 -> learned time weights beta_t -> blended feature
 -> small base MLP: density + features
 -> color MLP(features, viewing direction): RGB
 -> volume integration -> rendered image -> optimize against real images
```

The deformation is an SE(3)-based field: local transformations align time-varying points. Hash-grid features use the efficient Instant-NGP representation. Blending happens before decoding, in feature space; it is not averaging final rendered photographs. The direction-dependent color head explains appearance that changes with camera angle. Temporal code optimization and deformation training are parts of a sequence reconstruction, not an encoder that immediately predicts a generalizable head from one image.

## Mathematics and worked example

The central feature combination is

\[
f_t(x)=\sum_i\beta_{t,i}H_i(D(x,\omega_t)).
\]

Density \(\sigma\) determines opacity, and volume rendering weights each color by how much light survives previous samples along the ray. For an illustrative scalar feature, grids return 0.2 and 0.8 and weights are 0.75 and 0.25: the blend is 0.35. A learned decoder maps this feature to density/color; 0.35 itself is neither opacity nor skin albedo. This distinction prevents treating latent channels as physical materials.

## Dataset and supervision

The original paper reports **222 people, 4,734 sequences, 31.7 million frames** captured by 16 machine-vision cameras at 73 fps and 7.1 MP. Synchronization is below one microsecond. Calibrated cameras and controlled diffuse illumination support multi-view photo consistency. Capture tasks include expressions, emotions, sentences, hair movement, and free motion. The original raw storage total is 203 TB. These are paper-era statistics, not a promise about today's downloadable subset.

The original population is 157 men and 65 women and is predominantly White; detailed distribution is in PDF Fig. 3. Fine detail does not remove demographic imbalance. The reconstruction method optimizes photometric evidence from multiple views; its training inputs exceed our one-photo inference budget.

The **single-view reconstruction benchmark is a different evaluation protocol**. RealDenseFace reports the official 391-image test setting and additionally evaluates a new scan set. Do not merge those counts or imply benchmark ground truth is ordinary downloadable training data. Confirm current identity IDs and access before any split.

## Evidence, limitations, and integration

The paper studies novel-view synthesis and ablates deformation/feature ensembles. Radiance-field image quality is not a FLAME geometry score or a relighting score. Fixed-light training can embed illumination into the learned color field.

For TrueFace3D, use authorized selected capture frames as **multi-view supervision while giving the predictor only one frame**. Reserved cameras test whether an asset works beyond the input viewpoint. Reconstructing NeRSemble itself is optional reference work, not our default shared renderer.

1. **Hypothesis:** multi-view teacher consistency improves visible-to-hidden transition regions. Compare a one-frame predictor with and without teacher losses; keep test input identical.
2. **Hypothesis:** uncertainty masks help when a teacher's density surface is ambiguous around hair. Compare masked versus unmasked supervision on actual skin scans.

**Explain it back:** what does canonical deformation align, and why doesn't view-dependent RGB equal a relightable BRDF?
