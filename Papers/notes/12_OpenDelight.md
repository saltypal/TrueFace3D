# 12 — OpenDelight: remove illumination before estimating skin

**Learning a Delighting Prior for Facial Appearance Capture in the Wild**, Han et al., SIGGRAPH 2026 / arXiv 2605.05636. [PDF](../pdfs/12_OpenDelight.pdf), [paper](https://arxiv.org/abs/2605.05636), [code](https://github.com/yxuhan/OpenDelight). Primary reading: §§4–7 and Figures 2–5. Snapshot: 2026-10-10.

## Intuition and the problem

A red lamp can make ordinary skin look red. A cast shadow can make the same cheek look dark. Painting those observed colors directly onto a mesh keeps the old lamp and shadow attached when we relight it. Delighting tries to infer the skin's intrinsic diffuse color.

Think of a school drawing made under a colored torch: we want the paint on the drawing, not the torch's tint. However, a single photograph cannot uniquely tell us whether a dark patch is pigment, shadow or a geometric crease. A learned prior supplies likely answers, not direct measurement.

**Critical scope:** the paper's complete appearance-capture system uses a smartphone sequence and multiview fitting. The *delighting network itself* operates on a single aligned face image. We propose adapting that network for our monocular albedo stage; we are not claiming to reproduce the complete video pipeline from one photo.

## Architecture and execution

~~~mermaid
flowchart LR
 I[Aligned skin masked RGB] --> T[Image tokens]
 S[Source aware dataset tokens] --> V[MAE initialized ViT Base]
 T --> V
 V --> D[Convolutional decoder]
 D --> A[512 x 512 diffuse albedo estimate]
 A --> U[UNet detail enhancement]
 I --> U
 U --> O[768 x 768 albedo estimate]
~~~

The base model initializes a ViT encoder from masked-autoencoder weights. It appends learned source tokens corresponding to either real OLAT data or rendered scan data. Self-attention allows those tokens to affect image features. Afterwards the source tokens are removed and the image features are decoded into an albedo image.

This is **Dataset Latent Modulation (DLM)**. Mixing two datasets naively may teach inconsistent colors or sharpness; sequential fine-tuning can forget the real-image domain. DLM explicitly conditions the shared network on the source style. At inference, the paper selects the rendered-data tokens to favor sharper, more physically correct albedo while retaining knowledge gained from real captures.

The detail enhancer is a separate refinement stage. Its higher resolution does not imply the source photograph contains every generated pore. Treat recovered visible detail and plausible prior detail differently.

## Equations and a worked example

A simplified diffuse image model is \(I=A\odot S\), where \(A\) is albedo and \(S\) is shading. This is teaching notation, not the paper's entire lighting model.

If \(I=0.30\) and shading is \(S=0.50\), then \(A=0.60\). But the same pixel also admits \(A=0.30,S=1\). That ambiguity is why the network needs training examples and why direct division by estimated shading can become unstable in dark regions.

The base network can be written

\[
\hat A=D(\operatorname{Unpatchify}(E([T(I\odot M),d_s])_{image})).
\]

Here \(M\) selects skin, \(d_s\) denotes source tokens, and only the image-token part is decoded. Training combines L1 error and LPIPS. Pixel loss encourages correct values; perceptual loss encourages coherent structures. Neither alone establishes unbiased physical reflectance.

## Training data: exact distinction between paper and release

The paper processes 134 FaceOLAT subjects: 124 training and 10 evaluation. Its source capture contains 40 viewpoints and 331 individual light directions per subject. Linear combinations synthesize relighting. Uniform illumination is used as approximate albedo, but absent polarized capture allows specular contamination; slight capture misalignment also blurs targets.

A second source is 120 purchased Light Stage scans with 4K diffuse albedo, specular albedo and normals. The authors report 15 Asian, 23 African American and 82 Caucasian scans. These labels and proportions are their dataset description, not a comprehensive demographic assessment.

A collection of 473 HDRIs is split into 373 training and 100 evaluation environments. The authors generate about 20K pairs from each source, and favor high-frequency lighting to produce difficult shadows. The base model trains at 512 × 512; the enhancer at 768 × 768. Paper hardware is two 48GB L20 GPUs for 1.5 days for the base and two RTX 4090 GPUs for 0.5 days for enhancement.

**Release difference:** the live README replaces the planned NeRSemble-Scan release with **Ava256-Scan**, and describes optimized training on two 24GB RTX 3090 GPUs over 3–4 days. The inspected doc/TEST.md explicitly says the downloadable pretrained network is still the paper model trained on FaceOLAT plus the private Light Stage set. The existence of the new public dataset is not evidence that those weights were retrained on it. Archived README and TEST.md are in evidence/.

## Evidence and limitations

DLM ablations compare naive dataset mixing and source conditioning. The downstream purpose is consistent reflectance across views, not just one attractive de-shadowed image. The full paper uses its prior to constrain geometry/material capture with a sequence; that evidence does not automatically validate our one-photo UV completion.

Outputs are diffuse albedo estimates in image space. They are not ready-made FLAME UV materials, roughness, independent specular albedo or eye assets. A face mask can accidentally exclude beard, brows, makeup or ears; decide how each is represented. Missing back-side skin must be inferred. High-quality official examples are not a test on our degraded inputs.

## TrueFace3D integration

Retain the alignment homography. Map albedo back into the original image coordinates, then project onto the fitted RealDenseFace surface using that camera. Z-buffer visibility must prevent painting cheek colors onto occluded vertices. Keep a UV validity mask and confidence map. Work in declared color space; conversion to renderer-linear values happens exactly once.

Initially use conservative shared roughness/specular/SSS constants, clearly tagged as defaults. Albedo prediction does not justify fitting every PBR parameter independently from one photo.

Read test.py, alignment/masking utilities, model construction and the inverse coordinate transform. Follow doc/ENV.md and doc/TEST.md; do not assume checkpoint/preprocessing compatibility without a smoke test. Survey validation was source inspection; inference and T4 peak memory remain unmeasured.

## Improvement experiments

- Train a small confidence predictor using held-out known-albedo renders and real OLAT comparisons. Predict low confidence for saturated highlights and deep shadows.
- Compare supervised scan-only training, OLAT-only, naive mixing, DLM and our source-aware weighting using equal training budgets and subject/light-disjoint splits.
- Preserve freckles/moles using visible-image detail evidence and a masked residual; measure albedo error and identity retention rather than rewarding indiscriminate smoothing.
- Use camera/geometry uncertainty when projecting albedo to UV, comparing uniform projection to angle-weighted projection.

These are proposed experiments. DLM and mixed real/synthetic training already exist, so repeating them is reproduction, not our novelty.

## Explain it back

Why is a texture with a baked shadow hard to relight? Why is a uniform-light OLAT image only an approximate albedo target? What do the source tokens change? Where must the crop transform be inverted?

