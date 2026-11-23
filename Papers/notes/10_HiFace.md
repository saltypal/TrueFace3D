# 10 — HiFace — static detail and expression-driven detail

**HiFace: High-Fidelity 3D Face Reconstruction by Learning Static and Dynamic Details; Chai et al., ICCV 2023, arXiv 2303.11225.** Snapshot: 2026-10-10. [Original PDF](../pdfs/10_HiFace.pdf), [primary source](https://arxiv.org/abs/2303.11225). Read: SD-DeTail representation; Eqs. 10–15; §4 and ablations.

## Intuition and prerequisites

A person's long-standing skin folds and a wrinkle created by today's smile are not the same thing. HiFace explicitly models both, using synthetic faces where the two causes are known instead of hoping image reconstruction will discover the distinction.

## Architecture and execution

1. Regress coarse 3DMM and detail coefficients from the input portrait.
2. Construct static displacement from a learned displacement basis.
3. Construct dynamic detail using compressed and stretched displacement components and expression-related blending.
4. Render coarse geometry with the detail displacement and estimated appearance.
5. Train jointly with direct synthetic geometry/detail supervision, self-supervised real images and age-related knowledge distillation.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

A teaching form is \(d=d_{static}(z_s)+w_c(\psi)d_{compressed}+w_s(\psi)d_{stretched}\). Direct losses compare static, compressed and stretched maps against synthetic targets. An age predictor supplies an auxiliary distillation signal to static detail; it does not measure the person's exact wrinkles.

Illustrative: a neutral forehead keeps person-specific texture/folds; raising the eyebrows activates additional expression-related lines. If all lines are stored in a static code, animation freezes a surprised forehead even after the face returns to neutral.

## Training data and supervision

The paper generates 200K synthetic faces with vertex, landmark, albedo and displacement annotations using its cited synthetic-data pipeline. It adds 400K real images; those citations correspond to CelebA and VGGFace2. A pretrained detector supplies 669 dense landmarks. The age predictor and synthetic asset library are upstream training dependencies that must also be recorded.

## Evidence and limits

Ablations compare synthetic supervision, static/dynamic decomposition and auxiliary losses; some hold coarse shape fixed to avoid confusing detail gains with identity-shape gains. This controlled comparison is directly relevant to our optional per-component novelty.

Synthetic detail distributions can differ from real skin. Age supervision can encourage plausible age patterns rather than individual truth. The implementation/model asset availability must be audited; a project page or supplementary PDF is not a runnable checkpoint.

## TrueFace3D use and code reading

Reference for an animatable detail contract. Preserve separate static and expression-dependent fields in our asset schema even if the first DECA baseline only approximates this decomposition. Project: https://project-hiface.github.io/. Do not make an unreleased implementation a critical dependency.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Use bounded FLAME-native residuals with expression conditioning and compare neutral/smile held-out scans.
- Apply balanced synthetic/real training at equal sample and compute budgets, isolating decomposition from merely more data.

## Explain it back

Why should detail change with expression? Why fix coarse geometry in a detail ablation? Why can an age loss be useful yet biased?

