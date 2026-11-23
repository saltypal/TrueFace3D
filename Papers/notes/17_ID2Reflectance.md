# 17 — ID2Reflectance — condition reflectance on identity

**Monocular Identity-Conditioned Facial Reflectance Reconstruction; Ren et al., CVPR 2024, arXiv 2404.00301.** Snapshot: 2026-10-10. [Original PDF](../pdfs/17_ID2Reflectance.pdf), [primary source](https://arxiv.org/abs/2404.00301). Read: §3 codebook construction/fusion and identity-conditioned reconstruction; §4.1.

## Intuition

Measured reflectance datasets are small; ordinary face photos are plentiful. ID2Reflectance first learns shared facial structure from both, then creates domain-specific codebooks for different material types. Identity conditioning aims to stop changing a person's recognizable face while estimating reflectance.

## Architecture and execution

1. Train a VQGAN-style encoder, decoder and shared codebook using RGB and reflectance images.
2. Freeze encoder/decoder and specialize codebooks for diffuse, specular, roughness, normals and RGB texture.
3. Quantize encoded features against these codebooks and fuse candidate latent features through learned Swin-transformer weights.
4. Train an identity-conditioned face-swapping/reconstruction branch to preserve subject appearance.
5. Predict reflectance in image space and map to a compatible 3D face through geometry/UV correspondence.

## Mathematics and worked example

Quantization chooses \(q(z)=\arg\min_{c_j}\|z-c_j\|\). The reported codebook contains 1,024 vectors of width 256; a 512 × 512 image uses a 16 × 16 latent grid. These discrete priors capture likely facial patterns, while a fusion module learns how much of each domain code to use.

Illustrative: the RGB codebook knows many lip shapes while measured diffuse data has fewer identities. Shared structure can improve diffuse-lip reconstruction. But an identity embedding alone cannot tell whether a hidden mole exists; geometry/material observations remain necessary.

## Training data and supervision

Reflectance acquisition contains 135 participants: 115 training, twenty testing. Joint prior training also uses FFHQ. The face-swapper trains on VGGFace2HQ after removing small images and applying GFPGAN enhancement. That restoration creates an additional prior in the labels; improved resolution does not certify identity-preserved ground truth. The paper trains its initial shared prior for 700K iterations and domain/fusion fine-tuning stages for 100K each on A6000 hardware.

## Evidence and limitations

Ablations assess shared versus domain-specific codebooks and fusion; comparisons assess material maps and relighting. Improved identity-conditioned appearance is related to our goal, so generic identity-loss additions cannot be claimed as new without closer comparison.

Small captured reflectance coverage can constrain generalization. Image-space results require visibility-aware UV transfer; codebooks quantize away some unique detail. A published repository is not proof all dependent checkpoints/data are available.

## Integration and code reading

Reflectance comparator and precedent for information sharing across specialized models. Official repo: https://github.com/xingyuren/id2reflectance. Audit pretrained download instructions and geometry version before using it; keep its image-space normals separate from FLAME tangent-space maps.

## Improvements to test

- Compare identity-conditioned fusion with our geometry/visibility-calibrated fusion using the same measured targets.
- Evaluate rare marks and asymmetric features across illumination and pose, alongside recognition similarity; do not let an identity embedding be the only metric.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

How does a codebook differ from continuous regression? What can be shared across RGB and material domains? Why might restoration of training images introduce bias?

