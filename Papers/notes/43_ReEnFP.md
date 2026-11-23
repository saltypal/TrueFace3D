# 43 — ReEnFP: encoding appearance and geometry into facial priors

**WACV 2023.** Snapshot: 2026-10-10. [PDF](../pdfs/43_ReEnFP.pdf), [official proceedings PDF](https://openaccess.thecvf.com/content/WACV2023/papers/Sun_ReEnFP_Detail-Preserving_Face_Reconstruction_by_Encoding_Facial_Priors_WACV_2023_paper.pdf). Architecture: PDF pp. 3–5; training/experiments: pp. 5–7. Important prior art for combining separate appearance/detail generators with one face reconstruction.

## Intuition

One learned generator supplies plausible face colors and lighting; another supplies plausible small surface bumps. A shared encoder reads the photograph and tells these generators what to produce. When a region is visible it uses image features; when evidence is absent it relies more on the prior.

This resembles our modular idea at a broad level. A new project must explain how its material semantics, eye model, confidence coupling, or evaluation differ instead of claiming that using several components is new by itself.

## Architecture

```text
appearance-prior training: latent albedo + latent illumination -> two UV branches
 -> UV texture -> 3DMM render -> real-image discriminator
detail-prior training: displacement data -> geometry generator
single image + initially unwrapped texture -> shared feature pyramid
 -> Map2Style blocks -> extended W+ style codes
 -> adaptive fusion blocks: image features blended with generator features
 -> pseudo-albedo / pseudo-illumination / displacement
 -> coarse 3DMM refinement and final render
```

The appearance generator separates **pseudo-albedo** and **pseudo-illumination**, trained through image realism and rendering, not measured reflectance. Multi-view rendering helps discourage inconsistent textures. The geometry generator predicts UV displacement along coarse mesh normals.

A shared encoder takes the image together with its unwrapped texture. Map2Style blocks produce style vectors; adaptive fusion learns spatial attention balancing encoded evidence and generative prior. Texture encoding trains with masked pixel loss, LPIPS, and ArcFace identity loss. Geometry encoding uses supervised map losses where available and adversarial loss for unlabeled cases.

## Mathematics and worked example

The adaptive fusion concept is

\[
F^i=A^i\odot E^i+(1-A^i)\odot G^i,
\qquad p'_{uv}=p_{uv}+d_{uv}n_{uv}.
\]

Illustrative example: a visible cheek receives image/prior weight 0.9/0.1, while an occluded ear receives 0.1/0.9. The ear may look plausible but remains prior-derived. Attention weights are not calibrated uncertainty unless separately evaluated. A +0.2 mm displacement moves surface points; a dark pseudo-albedo mark changes color but does not create a dent.

## Training sources and why

The appearance prior uses **FFHQ 70,000 and CelebA-HQ 30,000 images**. The geometry prior uses FaceScape displacement data; this paper describes 888 people / 17,760 maps, with 360 identities not processed for privacy protection. These paper-era counts are not the present FaceScape release count or a complete sample split. The main pipeline combines those sources. Report the exact retained subset when reproducing.

Heavy occlusions such as glasses/masks are manually filtered for prior training. Priors use StyleGAN2-style 512² training, while reconstruction inputs are 224² and outputs 512². The feature extractor uses pretrained StarGANv2 blocks. The paper reports four 32 GB V100s. Tests include MoFA, CelebA, LFW, and FFHQ subsets for texture/identity comparisons; these are not measured skin BRDF test labels.

## Evidence and limitations

Rendered-image similarity and identity retrieval support appearance reconstruction under its setup. They do not establish uniquely recovered lighting or physical reflectance. Pseudo-illumination, synthetic completion, limited displacement labels, and fixed-template geometry can create plausible errors. A rendered match cannot show that the hidden side belongs to the actual person.

## TrueFace3D use and improvements

Retain as a nearest integration baseline: independent priors plus adaptive fusion already exist. Our plan uses RealDense geometry, explicit material/normal-frame contracts, separate optical eyes, shared camera/light, and evaluation under unseen views/lighting. Those are design differences; measured benefit and prior-art differentiation are still needed for novelty.

1. **Hypothesis:** calibrated raw evidence and component uncertainty outperform uncalibrated attention for completion regions. Compare against adaptive fusion with matched data.
2. **Hypothesis:** replace pseudo-illumination textures with shared physical illumination and visibility. Test relighting, input consistency, and cross-view material stability.

**Explain it back:** what makes its maps “pseudo” intrinsic properties? Where does the prior enter when the image offers weak evidence?
