# 24 — AudioEar — reconstruct an ear, then handle its attachment

**AudioEar: Single-View Ear Reconstruction for Personalized Spatial Audio; Huang et al., AAAI 2023, arXiv 2301.12613.** Snapshot: 2026-10-10. [Original PDF](../pdfs/24_AudioEar.pdf), [primary source](https://arxiv.org/abs/2301.12613). Read: Method/data collection; Eq. 1; depth guidance; scan2mesh benchmark and integration Figure 3.

## Intuition

Whole-face models give ears relatively few vertices and weak supervision. AudioEar specializes in a clear ear crop. It predicts ear shape for spatial audio, showing both the benefit of a regional model and the practical cost of attaching it to a head.

## Architecture and execution

1. Detect/select visible ears and use annotated landmarks for reconstruction supervision.
2. Use the UHM ear model with 2,800 vertices and 236 PCA shape coefficients.
3. Transfer a DECA-derived texture representation to the ear through alignment and nearest-neighbor correspondence.
4. Train a synthetic depth estimator and fuse multiscale depth features into the ear-regression network.
5. Fit shape/texture/pose through photometric, landmark and prior losses; separately attach the predicted ear to a body mesh for acoustic simulation.

## Mathematics and worked example

The ear model is \(M(\beta)=\bar S+\sum_{j=1}^{236}v_j\beta_jU_j\). The paper's benchmark is scan-to-mesh distance after fitting alignment/scale. Low aligned S2M does not certify absolute ear scale or its correct location on our head.

Illustrative: an excellent isolated ear mesh can float five millimeters away from the head, leaving a crack. AudioEar's attachment uses boundary triangulation and smoothing with manual effort. Our renderer needs a validated placement and seam treatment; model quality alone does not solve attachment.

## Data and supervision

AudioEar2D contains 2,000 FFHQ-derived ear images with 55 landmarks and occlusion annotations. Detection uses IBug Collection-B. Synthetic depth pretraining uses 10K generated samples. AudioEar3D is evaluation: 112 ears from 56 people, captured with structured light, each accompanied by RGB and dense point clouds. These are paired real observations/3D scans, but they are ear-specific rather than a complete face PBR dataset.

## Evidence and limitations

Ablations test depth guidance and ear priors. The reported S2M result is 1.28mm under their alignment protocol versus a mean-ear baseline of 1.78mm. Acoustic simulation is an application pipeline with manual work, not the reconstruction acceptance protocol.

The UHM ear topology differs from FLAME. Hair and earrings occlude evidence. Mirroring left ears to right ears is an evaluation/modeling assumption. The separate COMSOL simulation is outside our scope.

## Integration and implementation reading

Optional later ear specialist. Official code: https://github.com/seanywang0408/AudioEar. First retain FLAME ears; only add a specialist after geometry and skin baselines. Use world-space attachment anchors and explicit boundary blending instead of merging meshes by vertex-index coincidence.

## Improvements to test

- Condition ear fitting on confidence/occlusion and keep the coarse ear where input evidence is absent.
- Test automatic boundary attachment with geometric continuity and material-color seam metrics, comparing to unchanged FLAME ears.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

Why is isolated ear accuracy different from assembled-head accuracy? Which dataset is training and which is evaluation? What does alignment hide?

