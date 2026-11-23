# 34 — RingNet and NoW: identity consistency and independent shape evaluation

**CVPR 2019; arXiv 1905.06817.** Snapshot: 2026-10-10. [PDF](../pdfs/34_RingNet_NoW.pdf), [paper](https://arxiv.org/abs/1905.06817), [official code](https://github.com/soubhiksanyal/RingNet), [NoW benchmark](https://now.is.tue.mpg.de/). Architecture/loss: PDF pp. 4–6; benchmark: pp. 6–7.

## Intuition

Two photographs of you smiling and frowning should still recover the same underlying identity shape. A photograph of a different person should not. RingNet turns that simple idea into a training signal without paired image-to-scan labels.

NoW separately asks whether the recovered shape agrees with a real scan. Learning a face and measuring a face are different jobs; this paper contributes to both.

## Architecture and execution

```text
images grouped by person + 2D landmarks
 -> shared image encoder -> shape / expression / pose / camera parameters
 -> FLAME -> projected landmarks
 -> landmark loss + parameter regularization
 -> rings of same-person and different-person shape comparisons
single test image -> trained encoder -> FLAME reconstruction
```

The ring training arrangement gives shared encoder weights several related images. Identity coefficients should remain close for same-person images despite pose or expression. A margin-based contrastive signal separates different identities. Expression and pose may vary while shape remains consistent.

2D landmark supervision connects reconstructed geometry to the image; FLAME supplies the articulated shape family. The method does not use a photometric albedo renderer to train its central image-to-shape relation. It therefore cannot supply a finished realistic skin asset merely because it produces a face mesh.

## Mathematics and worked example

A schematic identity constraint is

\[
\|\beta_a-\beta_b\|^2 +
\max(0,m+\|\beta_a-\beta_b\|^2-\|\beta_a-\beta_c\|^2),
\]

where a/b show the same person and c a different person. This is an explanatory contrastive form; refer to the paper's ring definitions for the exact implementation. Illustrative example: the same person's coefficients differ by 0.2, another person's by 1.0. The first pair should be pulled together without forcing the different person's face to match. A smile should be explained using expression, not by changing identity jaw width.

Landmark projection loss is \(\sum_j\|\Pi(v_j)-l_j\|\). It teaches visible geometry but cannot uniquely recover all depth from one image, motivating identity regularization and the model prior.

## Training and benchmark evidence

Training uses VGGFace2, cleaned after OpenPose landmark failures to roughly **800,000 images**, plus about **3,000 extreme-pose images**. For unlabeled extreme poses the method creates same-image crop/scale pairs. “No paired 3D training data” does not mean no supervision: identity relationships and estimated 2D landmarks remain supervision.

NoW contains **2,054 iPhone X photographs of 100 subjects** with a separate head scan per subject: 620 neutral, 675 expression, 528 occlusion, and 231 selfie images. A neutral scan supplies identity-shape truth, not an expression-matched skin material label. Use the official validation/test protocol and server rules; do not tune on hidden test scans.

The benchmark aligns predictions to scans before measuring surface error. Thus report the prescribed alignment and facial region. Scale-normalized/aligned error does not establish metric scale recovery in uncalibrated photography.

## Limitations and TrueFace3D role

This is a historical geometry baseline and a still-useful benchmark reference. FLAME constrains geometry; landmarks miss surface detail. Incorrect identity labels, noisy landmarks, and shape/expression entanglement can hurt learning.

Use NoW as external geometry evaluation where lawful and permitted. Use ring-style training as a candidate consistency term for identity, including when multiple training views are available but test inference remains one-photo.

1. **Hypothesis:** cross-view identity consistency reduces raw/restored shape drift. Compare to a plain parameter loss at equal data and updates.
2. **Hypothesis:** region confidence weighting improves occluded landmarks. Evaluate actual scan error by occlusion category.

**Explain it back:** why can a method be unpaired with 3D yet supervised by identities? Why can't NoW validate recovered roughness or eye optics?
