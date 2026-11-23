# 38 — Real-ESRGAN: realistic degradation training for general image enhancement

**ICCV 2021 Workshops; arXiv 2107.10833.** Snapshot: 2026-10-10. [PDF](../pdfs/38_RealESRGAN.pdf), [paper](https://arxiv.org/abs/2107.10833), [official code](https://github.com/xinntao/Real-ESRGAN). Degradation: PDF pp. 3–4; architecture/training: pp. 5–6. Candidate optional image enhancement, including a separately evaluated final-render branch.

## Intuition

A model trained only on neat downsampling sees an easier problem than a real phone photograph. Real images may have been blurred, resized, compressed, and resized again. Real-ESRGAN trains using a sequence of these damaging steps so its repairs better match everyday degradation.

It restores general images, not only faces. Plausible high-frequency patterns may nevertheless be invented: clearer output is not evidence that hidden pores existed in the original scene.

## Architecture and execution

```text
HQ training patch -> degradation stage 1 -> degradation stage 2 + sinc filter
 -> synthetic LQ image
LQ -> RRDB-based generator -> enhanced image
 -> L1 + VGG perceptual + adversarial losses
spectral-normalized U-Net discriminator -> image/region realism signal
```

The generator retains ESRGAN's residual-in-residual dense blocks. The discriminator uses a U-Net architecture and spectral normalization to improve training stability and provide spatial feedback. First a reconstruction-oriented Real-ESRNet is trained; adversarial/perceptual fine-tuning then produces Real-ESRGAN.

The central contribution for this project is the **training degradation distribution**. Blur, resampling, noise, and JPEG can be composed repeatedly rather than applied once in a fixed clean order. Gaussian, generalized Gaussian, and plateau kernels broaden blur patterns. Sinc filters reproduce ringing/overshoot artifacts. A training pair queue broadens degradation diversity within batches.

## Mathematics and worked example

The classical degradation is \(x=[(y*k)\downarrow_r+n]_{JPEG}\). A high-order process applies successive damage operators: \(x=D_2(D_1(y))\).

Illustrative example: a 1024-pixel face is first reduced to 512 and JPEG-compressed, then reduced to 256 and compressed again. Simply blurring and downsampling the original directly to 256 produces different artifacts. A generator trained on both paths has a better chance of recognizing real compression damage, although its output still depends on learned priors.

Reconstruction loss favors matching known target pixels. Perceptual loss favors similar learned features. GAN loss favors natural-looking patterns. These competing objectives explain why sharper images can have weaker pixel accuracy.

## Training and evidence

Training uses **DIV2K, Flickr2K, and OutdoorSceneTraining**, with 256-pixel HQ patches. The paper reports four V100 GPUs, total batch size 48, 1,000K Real-ESRNet iterations and 400K Real-ESRGAN iterations. It uses EMA and a combination of L1, VGG19 perceptual, and adversarial loss. These training corpora do not provide matched face scans or physically separated skin materials.

The paper compares real-image visual quality and ablates degradation/discriminator choices. Its published examples include failure cases under severe or unusual degradation. Ordinary no-reference image quality cannot establish correct 3D facial identity, metric shape, or new-view consistency.

## TrueFace3D integration

Evaluate two distinct branches: **input enhancement** before reconstruction and **final rendered-image enhancement** after rendering. The first can alter inferred geometry/materials. The second changes display pixels while leaving the 3D asset unchanged. Keep a native-render result for every enhanced image.

Do not call a 4K enhanced screenshot a 4K observed texture or recovered pore geometry. If images are enhanced independently over a rotating face, inspect flicker and cross-view feature consistency. Re-enhancing UV maps is a third, materially different experiment and should require seam and relighting checks.

1. **Hypothesis:** physics-consistent degradations on real camera–scan pairs improve low-quality geometry robustness. Hold model/training budget fixed; compare classical versus repeated degradation.
2. **Hypothesis:** a render-conditioned enhancement network preserves projections better than general SR. Compare landmarks, identity, temporal/new-view consistency, and human preference alongside pixel metrics.

**Explain it back:** why does the order of compression/resizing matter? Why is enhancement of a screenshot different from improving its 3D surface?
