# 16 — AvatarMe++ — rendering-aware facial reflectance GANs

**AvatarMe++: Facial Shape and BRDF Inference with Photorealistic Rendering-Aware GANs; Lattas et al., arXiv 2112.05957 / TPAMI.** Snapshot: 2026-10-10. [Original PDF](../pdfs/16_AvatarMePlusPlus.pdf), [primary source](https://arxiv.org/abs/2112.05957). Read: §§3–5; Figures 2–3; Eqs. 13–14; patch and rendering ablations.

## Intuition

A texture generator can make plausible skin while giving wrong materials. AvatarMe++ asks whether its predicted maps produce the right appearance under several lights. This trains the bundle to behave coherently when relit.

## Architecture and execution

1. Fit a GANFIT-based 3DMM to obtain coarse geometry and UV texture containing captured illumination.
2. Upscale the texture with a domain-trained RCAN model.
3. Predict diffuse/specular albedo and separate diffuse/specular normal maps using a geometry-guided pix2pixHD-style network.
4. Train on overlapping patches with downsampled whole-texture context for global consistency.
5. Use a differentiable UV shader with approximate subsurface scattering and learned self-occlusion to penalize incorrect relit predictions.

## Mathematics and worked example

Its generator input contains texture, coarse geometry normals and whole-texture context. The renderer loss samples lighting conditions: \(\mathcal L_R=\frac1{n_L}\sum_j\|R(M,L_j)-R(M_{GT},L_j)\|^2\). Adversarial and discriminator-feature losses complement it. A realistic training shader constrains how maps interact; it does not make every material physically unique.

Illustrative: a cheek normal error may be hard to see under frontal soft light but obvious under a side light. Averaging loss over multiple lights exposes that error. Whole-texture context helps adjacent UV patches agree in skin tone instead of forming a checkerboard.

## Training data and supervision

The authors capture over 200 people under seven expressions using 168 LEDs and nine DSLR cameras. RealFaceDB releases anonymized/shuffled aligned 512 × 512 material/geometry patches rather than complete identifiable camera/mesh packages. Original UV assets reach 6144 × 4096. The paper reports demographic imbalance and applies albedo augmentation; this does not establish balanced real anatomical diversity.

## Evidence and limitations

Ablations test rendering loss, domain augmentation and patch/global context. Comparisons include captured material targets and Digital Emily. Meshes are registered to common topology for comparisons: a major hidden dependency that our evaluation must reproduce or document.

Approximate SSS and occlusion are not identical to full Cycles transport. Diffuse and specular normals serve distinct shading roles. Public patches do not supply every full-face identity or scan. GAN-derived invisible detail can be plausible without being true.

## Integration and code reading

Core foundational PBR reference, with MARCUS as newer related work. Official project: https://github.com/lattas/AvatarMe (verify current model-release instructions before use). Read material channel definitions and map-to-shader conversions; treat patch data as material supervision, not a full monocular geometry dataset.

## Improvements to test

- Blend global/patch predictions using calibrated boundary confidence and evaluate UV seam gradients.
- Compare our renderer-consistency loss with the original stochastic-light loss under equal maps, lights and compute.
- Study observed-detail preservation rather than maximizing resolution through repeated upsampling.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Why is one render insufficient to check a material bundle? Why supply whole-face context to local patches? Why are public patches different from full scans?

