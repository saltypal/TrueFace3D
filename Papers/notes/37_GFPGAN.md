# 37 — GFPGAN: a generative face prior with spatial fidelity control

**CVPR 2021; arXiv 2101.04061.** Snapshot: 2026-10-10. [PDF](../pdfs/37_GFPGAN.pdf), [paper](https://arxiv.org/abs/2101.04061), [official code](https://github.com/TencentARC/GFPGAN). Mechanism: PDF pp. 3–5; data/training: pp. 5–6.

## Intuition

A pretrained face generator knows how clear faces usually look. A second network examines a damaged image and tells the generator which face and spatial details to preserve. The generator contributes plausible detail; spatial conditioning tries to keep it tied to the input person.

This is a useful restoration idea, but its eye and mouth modules judge small 2D image regions. They do not reconstruct eyeballs or teeth in 3D.

## Architecture

```text
LQ face -> residual U-Net -> latent feature + multi-scale spatial features
latent feature -> MLP -> StyleGAN2 intermediate style codes
 -> pretrained StyleGAN2 feature hierarchy
spatial features -> channel-split spatial feature transformations
 -> restored image
training: global discriminator + local eye/mouth discriminators + ArcFace identity
```

The U-Net removes degradation while preserving spatial organization. The paper uses seven downsampling/upsampling stages. Its latent branch finds a style representation; its spatial branch conditions intermediate GAN features so restoration is more faithful than unconstrained GAN inversion.

**CS-SFT** transforms only part of the feature channels through spatial affine parameters, preserving another part of the generative prior. This balances explicit evidence and prior appearance. Multi-scale conditioning affects both broad structure and fine image detail.

The three local discriminators cover left eye, right eye, and mouth, extracted by ROI Align. They encourage locally realistic appearance. A Gram-statistics style objective compares feature correlations; an ArcFace feature loss encourages identity similarity. None of these proves recovered metric geometry.

## Equations and worked example

An SFT operation is \(F'=\alpha\odot F+\beta\), while channel splitting concatenates an unchanged feature subset with the transformed subset. Illustrative example: channels [2,4] become [2,5] when only the second channel receives a +1 shift. Real features contain many channels and spatially varying scale/shift fields.

The loss combines

\[
\mathcal L=\mathcal L_{reconstruction}+\mathcal L_{adversarial}
+\mathcal L_{components}+\mathcal L_{identity}.
\]

Pixel/perceptual fidelity, natural image statistics, eye/mouth appearance, and identity embedding are different constraints. Strong local adversarial loss may create a sharper iris even when its exact pattern cannot be recovered from the input.

## Training data and evaluation

Training uses **FFHQ's 70,000 images** at 512 × 512 with synthetic Gaussian blur, downsampling, additive noise, JPEG compression, and color jitter. The generator prior is pretrained StyleGAN2. The paper trains with four Tesla P40 GPUs and 800,000 iterations; this is source-reported compute, not our hardware benchmark.

Evaluation includes CelebA-Test (3,000), LFW-Test (1,711), CelebChild-Test (180), and WebPhoto-Test (407 detected faces). Metrics include PSNR/SSIM/LPIPS for paired synthetic data, FID/NIQE for perceptual quality, and ArcFace identity distance. Real photographs usually lack paired undegraded targets. The paper reports failures under very large poses and severe domain mismatch.

## TrueFace3D mapping and proposed improvements

Use released pretrained restoration as an optional controlled input condition. Preserve raw input, crop, alignment, checkpoint, and version. Eye/mouth discriminator outputs never become our separate 3D eye component. Color enhancement can alter illumination/skin-color evidence, so evaluate appearance decomposition against raw photos too.

1. **Hypothesis:** a geometry-constrained restoration loss protects nose/eye/mouth positions better than recognition loss alone. Fine-tune only if compute/data permit, and evaluate scan error plus identity.
2. **Hypothesis:** region-specific acceptance masks reduce hallucinated occluded features. Compare against all-or-nothing restoration with identical backbone predictions.

**Explain it back:** why does a discriminator trained on eye patches not count as an eyeball reconstruction model? What roles do latent and spatial conditioning have?
