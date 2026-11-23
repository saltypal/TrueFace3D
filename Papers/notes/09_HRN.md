# 09 — HRN — separate medium geometry, fine detail and skin marks

**A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images; Lei et al., CVPR 2023, arXiv 2302.14434.** Snapshot: 2026-10-10. [Original PDF](../pdfs/09_HRN.pdf), [primary source](https://arxiv.org/abs/2302.14434). Read: Figure 2; §§3.2–3.6; §4.1 and component ablations.

## Intuition and prerequisites

A smooth face base, a cheek bulge and a fine wrinkle are different scales of geometry. HRN predicts them in stages. It also tries to stop freckles and blemishes being interpreted as bumps by improving the albedo used for reconstruction.

## Architecture and execution

1. Estimate a coarse 3DMM mesh, camera, lighting and albedo with a pretrained face analyzer.
2. Unwrap texture and position information into UV and realign it.
3. Predict medium-scale deformation using a pix2pix-style module.
4. Predict high-frequency displacement with a separate module using geometric/texture context.
5. Train with scan-derived detail targets and adversarial priors; use de-retouching-derived albedo for photometric disentanglement.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

A teaching hierarchy is \(V_{final}=V_{coarse}+\Delta V_{mid}+d_{fine}n\). The de-retouching argument assumes shared shading: \(T_0=A_0S\) and \(T'=A'S\), so ideally \(A'=A_0(T'/T_0)\). Practical implementations stabilize the denominator and handle dark regions; a division formula alone is not a reliable albedo estimator.

Illustrative: a dark mole appears in UV texture but is absent from scan height. If the albedo model cannot express it, photometric optimization can manufacture a dent to explain the darkness. HRN's appearance handling supplies an alternative explanation so fine geometry need not absorb that color mark.

## Training data and supervision

The geometry training combines FaceScape, ESRC and FaceHD-100: around 9K scans from nearly 500 subjects, about 150K lab images with geometric detail targets, and about 260K in-the-wild images following its cited preprocessing. FaceScape uses 309 training and 50 testing subjects from the 359 subset. A separate 10K FFHQ set is professionally retouched to learn the de-retouching module. Do not rename the unspecified in-the-wild mixture FFHQ solely because the retouching subset is FFHQ.

## Evidence and limits

The paper compares single-view and sparse-view reconstruction and ablates detail priors/de-retouching. The multiview extension gains information our one-photo method lacks, so report it separately. Improved shading images should be checked against regional scan error.

Topology is not our FLAME2023 contract. Detail priors can hallucinate geometry, masking excludes eyes/hair, and the ratio approximation can fail under different shading or occlusion. Professional retouching and scan supervision add replication cost.

## TrueFace3D use and code reading

Strong reference for separating skin marks from geometry and for coarse-to-fine detail. Official project/code: https://younglbw.github.io/HRN-homepage/ and https://github.com/youngLBW/HRN. If borrowing detail, validate topology transfer and normal units before using it on the RealDense mesh.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Add visibility/uncertainty to deformation and displacement predictions rather than applying them globally.
- Train skin-mark preservation explicitly with known geometry and altered pigment, comparing against the original de-retouching strategy.

## Explain it back

What scale of geometry should each module own? Why can a mole cause a fake dent? Which data supervises detail directly?

