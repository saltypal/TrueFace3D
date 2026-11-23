# 39 — MonoNPHM: neural head tracking from a monocular video

**CVPR 2024; arXiv 2312.06740.** Snapshot: 2026-10-10. [PDF](../pdfs/39_MonoNPHM.pdf), [paper](https://arxiv.org/abs/2312.06740). Model: PDF pp. 3–4; inverse rendering/tracking: pp. 4–6; supplemental architecture follows. This is a later short-video reference, not a one-photo baseline.

## Intuition

A video gives several observations of the same face as it moves. MonoNPHM keeps one identity and appearance description for the whole sequence while allowing expression and head pose to change frame by frame. It learns a flexible neural head shape, then adjusts its latent controls until rendered frames agree with the video.

Appearance talks to geometry inside the network: the color model uses shape features so it cannot completely ignore the shape and paint away every geometric error.

## Architecture

```text
posed-space sample xp + expression/identity codes
 -> backward deformation -> canonical xc + two hyper-coordinates
 -> 65-anchor local geometry SDF experts (query nearest 8)
 -> geometry features -> local appearance experts -> RGB
 -> SDF-based volume rendering + spherical-harmonics lighting
 -> RGB, silhouette, landmark, and regularization losses
video fitting: shared geometry/appearance codes + per-frame expression/pose
```

Unlike original NPHM's forward deformation, MonoNPHM maps observed points backward into canonical space. Hyper-coordinates increase its ability to explain changing mouth/eye topology. Geometry experts blend around facial anchors; evaluating the nearest eight limits memory use. Appearance MLPs use 16-dimensional geometry features rather than free canonical coordinates alone.

The anchor structure also supplies landmarks. Because observed-space anchors are not directly given by a backward deformation, iterative root finding numerically inverts the deformation for the landmark constraint. This is a real integration detail; “apply the inverse transform” is insufficient when the deformation is nonlinear.

## Mathematics and worked example

The shared/per-frame parameter split is

\[
\phi=\{z_{geo},z_{app}\}\cup\{z^t_{exp},R_t,t_t\}_{t=1}^{T}.
\]

Optimization sums photometric, silhouette, landmark, and regularization losses across frames. Illustrative example: frame 1 shows a closed mouth and frame 2 a smile. Identity remains shared; per-frame expression changes. If each frame learns separate identity, jaw shape can flicker. If all frames share one expression, the model cannot explain the motion.

SDF values become volume densities through a NeuS-style formulation. Surface normals come from SDF gradients. Spherical harmonics multiply appearance with approximate shading; those learned scan colors are not guaranteed clean physical reflectance.

## Training data and evaluation

The paper trains an auto-decoder end-to-end on the **NPHM v2 release with 473 identities**, sampling SDF and RGB supervision from textured scans. It reports four 12 GB RTX2080 GPUs and roughly 52 hours. This is not the original NPHM paper's 255-person capture or 87-identity experiment.

Evaluation uses monocular RGB videos with Kinect depth for geometric checking. Depth around hair, neck, mouth interiors, and eyes is masked to reduce sensor/method differences. Its reported Chamfer values are in **meters**, while some related papers report millimeters. Shared identity codes and learned prior generalization must be distinguished from per-video fitting and personalization.

## Evidence, limitations, and mapping

The paper reports improved depth agreement and ablates geometry–appearance coupling, deformation, and hyper-dimensions. Video evidence is richer than a single image, and optimization uses whole-sequence information. Its color/shading model is not our full cornea/SSS/roughness scene model. Implicit heads require meshing/rebinding for ordinary graphics applications.

Borrow the shared-identity/per-frame-state separation for our future video phase. Preserve RealDenseFace as the first baseline. The geometry–appearance connection motivates a coupling hypothesis, but a new coupling must differ from this existing work to support novelty.

1. **Hypothesis:** freeze shared identity after a confidence-weighted initialization to reduce short-video drift. Compare against unconstrained per-frame fits.
2. **Hypothesis:** geometry-conditioned material refinement helps avoid painting structural errors into albedo. Compare with independent experts and inspect actual scan error plus relighting.

**Explain it back:** what stays fixed through the video? Why is nonlinear backward deformation harder to invert than a rigid camera transform?
