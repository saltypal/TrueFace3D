# 14 — MoSAR — a modular relightable avatar pipeline

**MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading; Dib et al., CVPR 2024, arXiv 2312.13091.** Snapshot: 2026-10-10. [Original PDF](../pdfs/14_MoSAR.pdf), [primary source](https://arxiv.org/abs/2312.13091). Read: §3, especially Figures 3–5; §4 training protocol; supplementary ablations.

## Intuition

MoSAR already implements the core intuition of specialized modules cooperating through rendering: geometry, texture completion, light normalization, detail and several skin attributes. It is essential prior work, not merely an optional comparison. The common UV representation lets each specialist predict compatible surface maps.

## Architecture and execution

1. Estimate geometry using an image encoder and nonlinear geometry decoder trained on scans and real photos.
2. Project the photograph to UV; complete unseen texture using an image-translation network trained with synthetic occlusions.
3. Normalize lighting in UV with a UNet so captured shadows are not stored as skin properties.
4. Predict displacement with patch PCA and a refinement network; derive detailed normals.
5. Predict diffuse, specular, ambient occlusion and translucency through attribute-specific networks and train them with supervised maps plus shared differentiable shading.

## Mathematics and worked example

Its mixed-supervision objective combines a shading reconstruction term with adversarial and direct map losses: \(\mathcal L=\mathcal L_{shading}+\sum_k(\mathcal L_{GAN,k}+\mathcal L_{sup,k})\). Direct map loss applies when ground-truth maps exist; image shading can supervise unlabeled examples. A residual-light encoder helps avoid baking unmodeled light into skin maps.

Illustrative: the diffuse specialist predicts a brown cheek while a specular specialist predicts a small white highlight. Their shared renderer must reproduce the observed cheek under the same lamp. Independently rendering each part with separate light estimates would produce an incoherent face.

## Training data and supervision

The private Light Stage set has 890 neutral subjects, twelve camera views per subject and 4K geometry/material attributes; fifty subjects are held for validation. Geometry also uses 70K FFHQ images. Texture estimation uses 54K FFHQ-UV textures, which are StyleGAN-generated identities, not 54K scanned people. A manually selected twenty-subject wrinkled subset builds the displacement patch basis. The public FFHQ-UV-Intrinsics release contains predicted maps for 10K synthetic identities, not measured materials.

## Evidence and limitations

Experiments compare geometry, reflectance and relighting, and test semi-supervised training and modeling choices. The reported results show the value of coordinating maps via shading. They do not establish that every inferred pore is person-specific ground truth.

Scan topology and shader conventions differ from FLAME/Cycles. The source Light Stage set is not the public 10K inferred-map release. Dataset availability is not model/weight availability; the project page inspected exposes a dataset but a runnable full checkpoint was not verified. Public intrinsics data carries noncommercial/no-derivatives terms; verify permitted research transformations and redistribution before using it.

## Integration and code reading

Use as architectural precedent and a strong baseline if executable assets are obtained. Preserve our module order but acknowledge it is established. Project: https://ubisoft-laforge.github.io/character/mosar/. Its separation of residual illumination, pigment and geometry is useful for our shared optimizer.

## Improvements to test

- Add calibrated reliability to observed versus completed UV regions and compare with its unweighted modular estimates.
- Evaluate teacher-generated targets separately from measured material targets and test demographic/domain shift under equal data budgets.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Which maps describe material and which describe geometric detail? What makes training semi-supervised? Why is the 10K public release not the 890-person scan dataset?

