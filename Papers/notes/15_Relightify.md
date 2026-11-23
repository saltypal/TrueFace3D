# 15 — Relightify — jointly diffuse texture and reflectance

**Relightify: Relightable 3D Faces from a Single Image via Diffusion Models; Papantoniou et al., ICCV 2023, arXiv 2305.06077.** Snapshot: 2026-10-10. [Original PDF](../pdfs/15_Relightify.pdf), [primary source](https://arxiv.org/abs/2305.06077). Read: §3.2–3.3, Eqs. 6–8; §4.1 and supplementary texture-domain analysis.

## Intuition

A photograph exposes only part of a face texture. Relightify learns a joint distribution of shaded texture, albedo, specular reflectance and normals. Inpainting that joint distribution can fill missing regions and infer material maps that belong together.

## Architecture and execution

1. Fit a 3DMM to the image and unwrap visible pixels into a partial UV texture.
2. Train separate autoencoders for shaded texture, diffuse albedo, specular albedo and normals.
3. Concatenate the compressed maps and train a joint latent diffusion model.
4. At inference, constrain observed texture regions while denoising the joint latent sample.
5. Decode a complete set of UV maps and apply them to the fitted geometry for relighting.

## Mathematics and worked example

The paper declares a 512 × 512 × 10 concatenated data tensor and a 64 × 64 × 12 concatenated latent tensor. Four modalities are not necessarily four RGB tensors: one reflectance map has a single channel. Its extracted text has inconsistent superscript labeling around Eq. 6; verify the equation/figure visually before hard-coding which map is scalar. The joint latent distribution is the central idea.

Illustrative: the left ear is absent in a frontal input. Inpainting may generate a plausible ear texture consistent with the face, but it cannot know the person's hidden scar. Visible evidence should stay constrained; invented regions should carry lower confidence.

## Training data and supervision

Training begins with approximately 4,700 MimicMe facial textures whose illumination is baked in. An AvatarMe++-based translator produces reflectance pseudo labels using available reflectance data. The authors render those reflectance/shape pairs under varied illumination and augment skin tones. These labels inherit teacher error; they are not independent measured reflectance for all 4,700 people.

## Evidence and limitations

The paper compares reflectance and relit results with related methods, including analysis of relighting-domain augmentation. It tests a generative reflectance prior rather than exact parameter identification from a single pixel observation.

The original geometry is LSFM identity plus 4DFAB expression space. A statement that any 3DMM fitter can be used does not mean any UV atlas or normal frame is interchangeable. Diffusion can change identity-sensitive marks. Original implementation/checkpoint availability needs an explicit audit.

## Integration and code reading

Use its joint-map completion as related work for the optional material branch. Keep observed/unobserved masks, random seed and source topology in our contracts. Compare against deterministic completion before accepting a generative dependency.

## Improvements to test

- Use confidence-conditioned inpainting with fixed observed anchors, evaluating identity retention across random seeds.
- Train with teacher confidence and a small separate measured-material validation set; compare with treating all pseudo labels as equally trustworthy.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Why jointly generate different maps? How can pseudo-label bias spread? What distinguishes a plausible invisible texture from an observed one?

