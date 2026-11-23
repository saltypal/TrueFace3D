# 36 — CodeFormer: restoring a face by predicting discrete tokens

**NeurIPS 2022; arXiv 2206.11253.** Snapshot: 2026-10-10. [PDF](../pdfs/36_CodeFormer.pdf), [paper](https://arxiv.org/abs/2206.11253), [official code](https://github.com/sczhou/CodeFormer). Architecture/training stages: PDF pp. 4–6; evaluation and failures: pp. 6–10. Optional future input restoration; not required by the current geometry-first plan.

## Intuition

Suppose you have a box of clean facial pattern pieces. Instead of painting a blurry face pixel by pixel, predict which pieces belong where. CodeFormer learns the pieces from good photographs and uses a Transformer to choose them for a damaged photograph.

The pieces are learned latent vectors, not literal nose/eye photographs. They constrain outputs to look face-like. When evidence is weak, the selected pieces can look convincing while belonging to the wrong identity or geometric structure.

## Architecture and three training stages

```text
Stage I: HQ image -> encoder -> nearest learned codebook vectors -> decoder -> HQ reconstruction
Stage II: LQ image -> fine-tuned encoder -> nine-block Transformer -> code indices
          -> frozen codebook + decoder -> restoration
Stage III: LQ spatial features -> controllable feature transformations -> decoder refinement
```

Stage I trains a quantized autoencoder with reconstruction, perceptual, adversarial, and quantization objectives. A 512 × 512 input becomes a 16 × 16 sequence: 256 positions choose from 1,024 codebook entries, each a 256-dimensional vector. Token positions are not facial landmarks.

Stage II freezes codebook and decoder and trains the LQ encoder/Transformer to predict the HQ code indices. Global self-attention helps one damaged region use context from the whole face. Cross-entropy trains code classification, while feature matching connects LQ features to the HQ quantized representation.

Stage III adds controllable feature transformation at decoder scales 32, 64, 128, and 256. Encoder information changes decoder features through learned spatial affine parameters. A user-selected weight trades prior-driven appearance against fidelity to the observed image.

## Equations and a worked example

Nearest code selection is \(k^*=\arg\min_k\|z-c_k\|^2\). For explanatory one-dimensional features, if z=0.7 and codes are 0.2 and 0.8, the second code is nearer. For heavily degraded input, this nearest choice can be wrong; the Transformer predicts a code using other positions too.

The controllable path is

\[
\hat F_d=F_d+w(\alpha\odot F_d+\beta),\quad w\in[0,1].
\]

With feature 2, affine correction 0.5, and w=0.8, the feature becomes 2.4. Larger w generally retains more input information; smaller w permits stronger prior-driven restoration. This is not a calibrated probability that the generated face is correct.

## Training datasets and evaluation

Training uses 70,000 FFHQ images resized to 512. Gaussian blur, downsampling, noise, JPEG compression, and resizing synthesize paired LQ/HQ images. The paper samples blur sigma 1–15, downsampling 1–30, noise 0–20, and JPEG quality 30–90.

Tests include synthetic CelebA-Test (3,000 images) and real LFW-Test (1,711), WebPhoto-Test (407), and WIDER-Test (970). These datasets evaluate restored image appearance/identity, not recovered 3D scan geometry. The paper explicitly identifies profile faces as a failure mode associated with FFHQ's limited side views.

## TrueFace3D integration and limitations

Keep the original crop and original camera transform. Record restoration weight and any alignment transform. Run raw and restored images through the same geometry/appearance baseline; do not presume restored landmarks or pores are observed truth. The restorer's code tokens are not FLAME parameters.

1. **Hypothesis:** compare raw/restored geometry priors with a confidence gate instead of unconditionally replacing the input. Ablate raw, restored-only, unweighted fusion, and gated fusion on degraded images paired with scans.
2. **Hypothesis:** select w using identity/geometry agreement rather than sharpness. Test against held-out scan error and calibrated raw-image reprojection.

**Explain it back:** what is a codebook token, and why can a perceptually better reconstruction be geometrically worse?
