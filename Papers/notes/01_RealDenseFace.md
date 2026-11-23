# 01 — RealDenseFace: dense priors → fitted FLAME

**Linzhou Li, Tianjia Shao, Kun Zhou; arXiv 2608.09238, August 2026.** Research snapshot: 2026-10-10. [Original PDF](../pdfs/01_RealDenseFace.pdf), [paper](https://arxiv.org/abs/2608.09238), [official code](https://github.com/LinzhouLi/RealDenseFace). Main method: PDF pp. 3–6; implementation and solver supplement: pp. 15–22. This is our geometry baseline. It does not supply complete skin materials.

## Start with the intuition

Imagine drawing thousands of numbered dots on a rubber face. In a photograph, you can guess where each dot should appear and whether it sticks out or sits farther back. RealDenseFace predicts those guesses, together with how uncertain each guess is. A separate optimizer bends a valid parametric face until its dots agree with the predictions.

The network is the observer; FLAME is the allowed family of faces; the solver reconciles them. Keeping those roles separate lets us inspect a bad correspondence or a bad fitting step instead of treating the whole system as an unexplained image-to-mesh box.

A UV map is a flattened chart of the surface. Here UV indexes *which point on the face* we mean. An image coordinate indexes *where that point appears in this photograph*. These are different coordinate systems.

## Exact output contract

For a UV location \(\nu\), the network predicts:

| Output | Meaning | Essential caution |
|---|---|---|
| \(C(\nu)\in\mathbb R^2\) | UV-to-image correspondence | Reverse direction from Pixel3DMM's image-to-UV output |
| \(D(\nu)\in\mathbb R\) | Relative camera-space depth | Depth relative to the FLAME neck joint |
| \(L_c(\nu)\) | Correspondence log-variance | Greater value means less weight |
| \(L_d(\nu)\) | Depth log-variance | Separate quantity from \(L_c\) |

The neck joint is near the head center in this model, not the literal bottom of a person's neck. Relative depth is \(z_i^{cam}-z_{neck}^{cam}\). Subtracting the anchor removes global translation; it does not magically recover focal length, physical scale, or unseen geometry.

## Architecture and forward pass

~~~mermaid
flowchart LR
 I[RGB crop 512 x 512] --> E[DINOv3 image encoder]
 Q[Learnable FLAME UV queries] --> C[Correspondence decoder]
 Q --> D[Relative depth decoder]
 E --> C
 E --> D
 C --> CH[DPT head: C and Lc]
 D --> DH[DPT head: D and Ld]
 CH --> S[Sample at fixed vertex UVs]
 DH --> S
 S --> G[CUDA Gauss Newton fitting]
 G --> F[FLAME parameters and mesh]
~~~

1. Detect/crop the face and retain the crop-to-original transform. The paper trains on 512 × 512 RGB crops.
2. Extract image features using DINOv3. These encode face structure; they are not 3D vertices.
3. Learnable UV queries gather image evidence into a canonical FLAME surface chart. Correspondence and depth have separate decoder branches because they solve different prediction tasks.
4. DPT heads convert decoder features into dense maps and corresponding uncertainty maps.
5. Sample each map at the UV coordinate assigned to each FLAME vertex. Dense maps thereby become per-vertex fitting targets.
6. Fit a FLAME face to these targets. Export model parameters, camera/crop metadata, mesh and diagnostic maps.

The inspected released ViT-B configuration uses target_size 512, four decoder blocks, six decoder heads, dpt_feature_dim 128, and scale_depth 8.0. These are configuration values, not permission to infer undocumented hidden tensor widths. The same configuration sets dino_frozen true; the paper's training supplement lists separate backbone/main learning rates. Record this paper/release distinction when reproducing training.

## Training mathematics, explained

For a prediction \(\hat y\), target \(y\), and predicted log-variance \(\ell\), the paper uses a Gaussian negative log-likelihood of the form

\[
\mathcal L(y,\hat y,\ell)
=\tfrac12\ell+\tfrac12\|y-\hat y\|^2 e^{-\ell}.
\]

The second term says “penalize errors more when the network claims certainty.” The first term prevents it escaping every error by declaring infinite uncertainty. Spatial weights exclude invalid regions and alter the relative contribution of face/head regions.

Both dense UV pixels and sampled vertices receive supervision. With the paper's weights,

\[
\mathcal L =
\mathcal L^c_{vert}+0.01\mathcal L^c_{dense}
+8(\mathcal L^d_{vert}+0.01\mathcal L^d_{dense}).
\]

Do not reuse these numerical weights for a new fusion loss without checking normalization: correspondence and depth have different scales.

## Solver mathematics and a solved example

The paper uses 300 identity coefficients \(\beta\), 100 expression coefficients, global rotation, local articulation, and translation. Its dynamic block has 118 variables. The local articulation block has 12 parameters; it is not just the three jaw angles.

For vertex \(i\), calculate its current projected location \(u_i=\Pi(V_i)\) and relative depth \(d_i\). Its whitened residuals include

\[
r_i^c=\sqrt{\lambda_c}\,e^{-\ell_i^c/2}(u_i-\hat u_i),\qquad
r_i^d=\sqrt{\lambda_d}\,e^{-\ell_i^d/2}(d_i-\hat d_i).
\]

The square-root weight belongs on the residual; squaring the residual then produces inverse-variance weighting. Applying inverse variance twice is a common implementation mistake.

**Illustrative numbers, not measured results:** two vertices both miss their correspondence by two pixels. Vertex A has variance 1; vertex B has variance 4. Their weighted squared errors are 4 and 1 respectively. B still contributes, but less. When \(\ell_B=\log 4\), its residual multiplier is \(e^{-\ell_B/2}=1/2\).

For a current parameter estimate \(p\), linearize residuals as \(r(p+\Delta)\approx r(p)+J\Delta\). Solve

\[
(J^\top J+\lambda_{GN}I)\Delta=-J^\top r.
\]

This is damped Gauss–Newton. The Jacobian describes how moving each parameter moves each residual. Damping stabilizes nearly singular systems. The implementation uses analytic derivatives, CUDA kernels and Cholesky factorization rather than hundreds of generic Adam updates.

The paper alternates dynamic updates with identity updates. It initializes global pose first. The supplement gives five pose initialization steps and fifteen group-descent steps for single images; confirm the released fitting configuration before reproducing these numbers.

## What they trained on, and why

| Source | Supervision construction | Why it helps |
|---|---|---|
| AVA256 | Transfer tracked meshes to FLAME; estimate SMIRK expressions; sample about 60 diverse frames per subject | Real cameras, expression variation and consistent tracked geometry |
| NPHM | FLAME registration using ICP plus an AVA-initialized prior; render textured scans with random cameras/HDRIs in Blender | Detailed 3D heads and controlled synthetic lighting/view variation |
| FaceScape | FLAME registration to scans with ICP and prior alignment | Additional identities and expressions with scan geometry |

Reported processed image amounts are approximately 349K AVA256, 384K NPHM and 244K FaceScape. They are processed training images, not counts of independent people. The supplement uses a 5% image validation split, batch size 12, 25 epochs and four RTX 4090 GPUs over two days. This does not demonstrate feasibility of retraining on a free T4, and an image split is not a substitute for our planned subject-disjoint evaluation.

The official repository currently releases inference/fitting and pretrained models. Its README still lists training data as forthcoming. Downloading the three upstream datasets does not automatically recreate the authors' registrations, selected frames or exact split.

## Evidence and limitations

The paper evaluates posed and neutral geometry on NeRSemble SVFR, neutral geometry on NoW, sequence behavior, and runtime. Its ablations isolate relative depth, pose initialization, MICA initialization and the solver. Read these as answers to distinct questions: better geometry, better initialization and faster optimization are different contributions.

Its headline 84 FPS concerns online tracking on an RTX 4090. It is not a speed estimate for TrueFace3D with albedo, detail, eyes and final path tracing. The supplementary extended NeRSemble evaluation uses 1,150 reference scans rather than the 391 official scans; do not report the reproduced evaluation as the official benchmark.

Occluded ears, specular eyes, poor lighting, extreme expression and inaccurate camera assumptions remain difficult. Uncertainty estimated under the training distribution can be confidently wrong on a new degradation. A FLAME mesh can fit the visible face and still infer the wrong hidden side.

The paper's identity energy uses a MICA prediction. The inspected public single-image entry point does not explicitly load MICA. Future agent: inspect GNFlameOptimizer initialization and its prior before asserting the released default is the exact MICA-initialized paper configuration.

## Code reading and our adapter

Read fit_single_image.py → tracker inference/optimizer → models/realdenseface → models/flame → cuda_extensions/flame_solver. Configurations are split between configs/model and configs/fitting. FLAME2023 must be obtained under its separate license.

The later **FLAME2023 Open** release has a different expression basis: inspect [the FLAME release note](02_FLAME.md) before substituting assets. Benchmark the converted meshes rather than assuming the fitted coefficients remain valid.

The adapter should preserve raw C/D/Lc/Ld, vertex UVs, camera intrinsics, original image size, crop transform, fitted parameters and solver diagnostics. Returning only an OBJ loses information needed for texture projection and reliability research.

**First acceptance experiment:** run the upstream single-image command on a clear portrait, inspect a clay render, overlay landmarks and reproject vertices into the original image. Log hardware, upstream commit, weights hashes, latency and peak memory. No inference was run during this survey.

## Improvements worth testing

1. **Visibility-calibrated reliability:** fit a small calibrator using scan-derived correspondence/depth error and occlusion labels. Compare original uncertainty, temperature calibration and learned regional calibration at the same frozen backbone.
2. **Camera sensitivity:** sweep plausible focal lengths, inspect profile/nose errors, and return a sensitivity estimate. Compare to calibrated real camera intrinsics on held-out subjects.
3. **Geometry-preserving appearance feedback:** allow appearance fitting to adjust bounded detail/material residuals first. Penalize changes in reliable raw geometry; compare joint optimization against frozen mesh.
4. **Restoration extension:** only later compare raw/restored dense priors. Require geometric error and identity evidence, not a prettier input.

Success requires improvement on held-out geometry or appearance without sacrificing identity, not merely a smoother overlay.

## Explain it back

Why are the two uncertainties separate? Why is relative depth translation-independent? What does a Jacobian column represent? Why does a two-dimensional match leave depth ambiguous? Which information would disappear if we saved only a mesh? Answers should refer to the mechanism above, not only the model name.

