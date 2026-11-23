# 18 — ReflectanceMM — learn reflectance from low-cost captures

**Learning a 3D Morphable Face Reflectance Model from Low-cost Data; Han, Wang and Xu, CVPR 2023, arXiv 2303.11686.** Snapshot: 2026-10-10. [Original PDF](../pdfs/18_ReflectanceMM.pdf), [primary source](https://arxiv.org/abs/2303.11686). Read: §§3.1–3.3; §4.1; spatial-shininess and model-update comparisons.

## Intuition

A professional Light Stage is expensive. Multi-PIE's multiple views and flashes offer a cheaper source of reflectance evidence. ReflectanceMM fits materials to those controlled observations, compresses them into a PCA model, then expands that model using ordinary photographs.

## Architecture and execution

1. Select consistent Multi-PIE views/lighting and subtract the room-light contribution to form OLAT-like images.
2. Fit BFM09 geometry, cameras and flash positions jointly.
3. Estimate diffuse color and weights of several fixed-shininess Blinn–Phong lobes in UV.
4. Learn a low-dimensional reflectance PCA model.
5. Fine-tune that model and an image regressor using FFHQ with a constrained high-order spherical-harmonics lighting space.

## Mathematics and worked example

Each vertex has RGB diffuse color plus three specular mixture weights. The paper uses Blinn–Phong exponents 1, 8 and 64; local weights control effective shine. An explanatory BRDF form is \(f=C/\pi+\sum_k W_k a_k(n\cdot h)^{p_k}\). A PCA reflectance model constrains coefficients; this is not the same parameterization as GGX roughness.

Illustrative: cheeks favor a broad dim lobe and the nose favors a narrower bright one. A single global shininess cannot fit both. Copying a Blinn–Phong exponent directly into Blender's roughness socket produces an unrelated material response.

## Training data and supervision

Multi-PIE initially has 337 identities, fifteen views and nineteen illuminations. The selected fitting set has 128 identities, nine views and twelve lighting captures, giving eleven flash-only images per view after room-light subtraction. FFHQ supplies 70K images for fine-tuning. This provides multiview training evidence while inference remains single-image.

## Evidence and limitations

Experiments assess reconstruction and relighting on CelebA-HQ and Multi-PIE and compare spatially varying shine to global exponent models. Read model-update regularization ablations: fitting uncontrolled photos can improve coverage while also corrupting measured reflectance.

Reflectance-light intensity retains a global ambiguity. Camera/flash assumptions and shadows affect extracted targets. BFM09 topology and the lobe mixture differ from our FLAME/PBR contract. PCA favors smooth likely appearance and may lose rare detail.

## Integration and code reading

Useful low-cost training precedent and possible lightweight baseline. Official code: https://github.com/yxuhan/ReflectanceMM. Inspect reflectance-model downloads and BFM asset licensing separately. Convert its shader by fitting target GGX response across light/view directions, not relabeling channels.

## Improvements to test

- Transfer measured evidence into FLAME with an explicit BRDF conversion benchmark.
- Retain source-confidence weights while fine-tuning on real photos and measure whether held-out controlled relighting degrades.
- Compare source balancing with more FFHQ images at constant training steps.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Why subtract room lighting? What does a specular mixture weight mean? Why is shininess not numerically identical to roughness?

