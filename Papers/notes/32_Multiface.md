# 32 — Multiface: real views, tracked meshes, and neural appearance baselines

**2022 technical report; arXiv 2207.11243.** Snapshot: 2026-10-10. [PDF](../pdfs/32_Multiface.pdf), [paper](https://arxiv.org/abs/2207.11243), [official dataset/code](https://github.com/facebookresearch/multiface). Capture pipeline: PDF pp. 3–4; neural rendering comparisons: following experiment sections.

## Intuition

Many cameras film the same expression at once. A tracked mesh tells us where the face is, while those cameras tell us how it looks from different directions. The dataset can test whether a renderer reproduces a view it never saw.

Its baseline learns a compressed description of a person's expression and decodes appearance conditioned on the camera direction. This can look realistic while being quite different from estimating reusable albedo, roughness, and optical eye materials.

## Capture and modeling architecture

```text
Mugsy synchronized camera dome -> multi-view images + audio
 -> multi-view reconstruction/tracking -> tracked mesh + head pose
 -> UV unwrapping -> average texture + calibrated cameras
 -> expression/appearance autoencoder -> latent code
 -> view-conditioned decoder -> geometry / appearance
 -> differentiable render -> held-out camera comparison
```

The paper contains 13 identities and two capture setups. It describes approximately 40 views in v1 and 150 in v2; the current official README says approximately 40/160. Record the actual valid camera list instead of assuming all cameras exist. Ten captures use v1 and three use v2. Image release resolution is 2048 × 1334, UV textures 1024 × 1024, video 30 fps.

Tracking yields temporal vertex correspondence, valuable for animation and geometry supervision. Camera calibration connects world points to each view. Average UV textures are images of observed appearance under the dome illumination. They are not automatically physically separated skin reflectance.

A codec-avatar baseline compresses tracked geometry and texture into a latent representation, then reconstructs geometry/texture with view information. The supplied neural appearance code is a research baseline, not a generic one-photo asset creator. Identity-specific training must be identified explicitly when comparing it with a generalizable method.

## Mathematics and worked example

Camera projection conceptually uses \(p\sim K[R|t]X\). Illustrative example: a 3D cheek point lies 600 mm from the camera and 30 mm to its right. With focal length 1,000 px its horizontal displacement from the principal point is \(1000\times30/600=50\) px. A wrong head pose moves the predicted cheek in every view even when its texture is perfect.

For a learned renderer, a useful conceptual objective is

\[
\mathcal L=\mathcal L_{image}(R(D(z,d)),I)+\lambda_g\mathcal L_{geometry}+\lambda_z\mathcal L_{latent}.
\]

This explains responsibilities rather than prescribing the exact weights of every baseline variant in the paper. Geometry, decoder capacity, latent bottleneck, and camera selection influence image quality independently.

## Data, why it matters, and limitations

The release provides real photographs, calibrated cameras, tracked meshes, head poses, UV textures, and audio. It addresses our “one photo plus real 3D evidence” question: select one camera as the predictor input and retain the rest as training/evaluation evidence. A tracked mesh is a derived estimate, not perfect metrology ground truth.

The official repository is archived as of January 2025 and uses CC-BY-NC 4.0. It lists about 65 TB for the full release and a 16.2 GB mini package. Start with an explicit small capture selection. Thirteen identities offer rich views but weak evidence of broad identity generalization.

## Integration and proposed experiments

Use Multiface for camera/renderer validation and later short-video temporal evaluation. Check its topology and metric scale before transferring labels into FLAME. Split by identity for generalization and by cameras/temporally separated segments for personalization; those are different claims.

1. **Hypothesis:** reserving extreme side cameras reveals baked shading better than input-view metrics. Evaluate the same assets on both views.
2. **Hypothesis:** view-balanced rather than frame-count sampling improves profile reconstruction. Keep the number of identities and updates fixed.

**Explain it back:** why do 100,000 frames from one person not equal 100,000 independent identities? Which assets are captured and which are derived?
