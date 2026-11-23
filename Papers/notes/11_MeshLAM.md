# 11 — MeshLAM — textured mesh avatars from one image

**MeshLAM; arXiv 2604.22865, 2026. Full bibliographic title is recorded in the downloaded PDF.** Snapshot: 2026-10-10. [Original PDF](../pdfs/11_MeshLAM.pdf), [primary source](https://arxiv.org/abs/2604.22865). Read: Dual-branch mesh/texture method, §4.1 and supplemental architecture details.

## Intuition and prerequisites

MeshLAM predicts an explicit deformable textured head instead of relying entirely on a radiance field or large collection of Gaussians. Its relevance is showing that single-image mesh assets can look detailed, while reminding us that an RGB texture is not automatically a physical skin material.

## Architecture and execution

1. Extract pretrained frozen DINOv2 features from the source portrait.
2. Use learnable tokens and transformer attention to obtain appearance and geometry features.
3. Iteratively refine texture through convolutional recurrent updates and geometry through MLP updates.
4. Constrain deformation differently for face, neck and hair regions.
5. Animate/render the resulting explicit mesh with predicted RGB texture under driving signals.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

Represent the mesh as \(V'=V_{template}+\Delta V\) and its texture as a UV RGB image \(T\). A region-specific bound clips \(\Delta V\) by semantic part. The supplement uses a 64 × 64 learnable texture-token grid decoded to 1024 × 1024 × 3. These are paper dimensions; they are not our proposed FLAME material resolution.

Illustrative: allowing large deformations in the hair region can produce a plausible hairstyle silhouette, while a tight face bound protects facial proportions. But a texture containing a bright cheek highlight retains that highlight under a changed lamp unless reflectance is separately inferred.

## Training data and supervision

The paper trains on VFHQ: 15,204 video clips, approximately three million frames. It estimates camera and FLAME parameters across sequences, removes background and parses the head. Inputs are 512 × 512. Training runs 100 epochs; its transformer has two layers, sixteen attention heads and hidden dimension 1,024. The official VFHQ test split uses a source frame and later driving/target frames.

## Evidence and limits

Image reconstruction/reenactment metrics include PSNR, SSIM, LPIPS and FID. They assess rendered image agreement, not independently measured roughness, specular reflectance or unseen shape. A sequence-derived training signal is compatible with single-image inference, but source/driving information in an evaluation must be explicit.

Do not interpret textured-head realism as fully relightable PBR appearance. Public executable code and weights were not verified; the project repository found during discovery was a website. Hair also exceeds our current core scope.

## TrueFace3D use and code reading

Latest explicit-mesh related work and a potential later comparator. It does not replace the requested RealDenseFace first milestone. Use its semantic deformation constraints as inspiration, with independent identity/geometry tests.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Compare confidence-conditioned facial displacement against its fixed regional bounds.
- Add physically separated albedo/specular/roughness outputs and held-out-light evaluation; study how this differs from MARCUS before claiming novelty.

## Explain it back

Why is an explicit mesh useful for editing? What is missing when we have only an RGB texture? How does training on video differ from requiring video at inference?

