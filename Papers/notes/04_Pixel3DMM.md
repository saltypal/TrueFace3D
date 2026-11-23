# 04 — Pixel3DMM — image-space UV and normal experts

**Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction; Giebenhain et al., arXiv 2505.00615.** Snapshot: 2026-10-10. [Original PDF](../pdfs/04_Pixel3DMM.pdf), [primary source](https://arxiv.org/abs/2505.00615). Read: §3.1 architecture/data and §3.2 fitting; NeRSemble evaluation and supplementary registration details.

## Intuition and prerequisites

Instead of asking a network for every face slider directly, ask it easier intermediate questions: which surface location does each image pixel show, and which way does that surface face? Then fit a valid face to those answers. Intermediate evidence is inspectable, but an optimizer must reconcile inconsistent predictions.

## Architecture and execution

1. Process a 512 × 512 RGB image with two DINOv2-based expert networks.
2. For each expert, add four transformer blocks, three up-convolutions and a linear unpatchification layer.
3. Predict a 512 × 512 × 2 UV-coordinate image and a 512 × 512 × 3 normal image.
4. Render/project a candidate FLAME face and compare its pixel-aligned UVs/normals with the estimates.
5. Optimize identity, expression, pose and camera with geometry-based losses, then evaluate posed and neutral shape separately.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

The declared experts are \(U:\mathbb R^{512\times512\times3}\to[0,1]^{512\times512\times2}\) and \(N:\mathbb R^{512\times512\times3}\to[-1,1]^{512\times512\times3}\). Conceptually, fit \(\min_p\lambda_U\|U_{render}(p)-U(I)\|+\lambda_N\mathcal D(N_{render}(p),N(I))+\mathcal R(p)\). This explanatory form omits the original loss's masking and exact norm choices.

Illustrative: an image pixel on the nose predicts UV (0.51,0.44) and an outward-facing normal. If the rendered mesh places a cheek at that pixel, the UV mismatch identifies the alignment error; if it places the nose but points sideways, normal supervision reveals the depth/orientation error.

## Training data and supervision

NPHM, FaceScape and AVA256 are registered to FLAME to obtain geometric targets, producing over 1,000 identities and approximately 976K images. Camera/rendering and registration quality determine the correctness of pixelwise supervision. These three sources are also used by RealDenseFace; adding them again is not new dataset coverage.

## Evidence and limits

The paper introduces NeRSemble evaluation of both posed and neutral geometry and reports substantial improvements over compared methods. Read normal-only/UV-only fitting ablations to identify complementarity. Distinguish neutral identity errors from expression-dependent errors when reproducing comparisons.

Screen-space normals and UV predictions can fail under occlusion or domain shift. Iterative fitting is slower than direct regression. Surface normals are geometric cues, not intrinsic skin color. Benchmark improvements do not establish photorealistic materials.

## TrueFace3D use and code reading

Strong geometry comparator and possible normal prior. Its image-to-UV direction is the reverse of RealDenseFace's UV-to-image map: inverse lookup requires rasterization/visibility handling, not swapping array axes. Official project: https://simongiebenhain.github.io/pixel3dmm/. Audit linked code/weights before implementation.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Test a small reliability model combining RealDense depth/correspondence and Pixel3DMM normals under controlled camera/lighting shift.
- Compare confidence-weighted fitting with unweighted concatenation while holding the fitted FLAME basis, optimizer budget and datasets constant.

## Explain it back

How do the two correspondence directions differ? Why do normals constrain shape yet leave some depth ambiguity? Which model owns the final mesh?

