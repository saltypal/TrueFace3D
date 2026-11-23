# 07 — DECA: coarse face plus animatable detail

**Learning an Animatable Detailed 3D Face Model from In-The-Wild Images**, Feng et al., SIGGRAPH 2021 / arXiv 2012.04012. [PDF](../pdfs/07_DECA.pdf), [paper](https://arxiv.org/abs/2012.04012), [code](https://github.com/yfeng95/DECA). Read §4, detail-consistency formulation, §5 and ablations. Snapshot: 2026-10-10.

## Intuition

A smooth face mesh captures nose size and jaw shape, but misses creases. DECA adds a fine height map over a coarse FLAME face. It also tries to separate the person's enduring detail from wrinkles caused by their current expression.

Imagine a rubber mask with fine bumps. Bending the mask into a smile changes crease patterns, but does not make it become another person's mask. That is the intended separation between identity detail and expression-conditioned detail.

## Architecture and execution

~~~mermaid
flowchart LR
 I[224 x 224 face crop] --> C[Coarse ResNet encoder]
 I --> E[Detail ResNet encoder]
 C --> P[FLAME shape expression pose camera light texture codes]
 P --> G[Coarse FLAME geometry]
 E --> Z[Person detail code]
 Z --> D[Detail decoder]
 P --> D
 D --> H[256 x 256 UV displacement]
 G --> N[Detailed surface normals]
 H --> N
 N --> R[Differentiable rendered image]
 P --> R
~~~

The coarse encoder estimates shape, expression, pose, camera, illumination and an albedo representation. FLAME decodes the geometry; spherical harmonics model diffuse illumination. A separate detail encoder returns a detail latent code. The detail decoder conditions on that code, expression coefficients and jaw pose to synthesize a UV displacement map.

The coarse and detail stages are trained with different goals. A good-looking shaded image can be achieved by several combinations of texture, geometry and light. Detail supervision therefore needs additional constraints, not just pixel matching.

## Equations

The coarse model is \(V=FLAME(\beta,\psi,\theta)\). At a surface sample with normal \(n\), a scalar displacement \(d\) gives an approximate detailed position

\[
V'=V+dn.
\]

A detail generator can be summarized as \(d=G(z_{detail},\psi,\theta_{jaw})\). Recompute normals from the displaced surface for shading. A normal map changes lighting response; actual displacement changes positions and potentially silhouette. They must not be confused.

DECA's detail-consistency training uses multiple images of the same identity. Swapping or sharing identity detail while retaining the target expression encourages a consistent identity code and expression-dependent wrinkles. Reconstruction, perceptual/identity, landmark and regularization terms complement this constraint; read their exact weights from the original configuration rather than inventing a universal loss recipe.

## A worked example

**Illustrative:** at one forehead point, a displacement of 0.0005 metres moves the surface half a millimetre along its normal. A grazing light may make that crease visually prominent while the silhouette barely changes. The same displacement interpreted as millimetres rather than metres becomes a thousand times too large.

If two photographs of the same person show neutral and raised-brow expressions, the intended detail identity stays similar while the decoder changes forehead creases with expression. Two independent per-photo height predictions can instead mistake temporary creases for permanent aging.

## Training data and why

The paper reports approximately two million images selected from VGGFace2, BUPT-Balancedface and VoxCeleb2. Identity labels provide same-person groups; no ground-truth detail scan is needed for every training photograph. Landmark filtering rejects inconsistent detections. The model uses 224 × 224 input and 256 × 256 UV maps.

Diverse photographs offer expression, lighting and pose variation, but image-based training can still bake shading into geometry or albedo. The training set and any upstream face-recognition loss data may overlap evaluation identities; track these separately.

DECA is a practical, openly implemented baseline. Obtain its exact pretrained assets and separately licensed FLAME model. Do not assume a FLAME2020-based detail decoder is automatically calibrated for a FLAME2023 coarse surface.

## Evidence and limitations

The paper's ablations examine detail consistency and reconstruction losses. Its animatable results demonstrate that expression-conditioned detail can be reused under changed expression. This is stronger evidence than a single fixed portrait render, but it is not measured recovery of every pore.

Detail resolution and the parametric coarse surface limit fidelity. Shadows, makeup, facial hair and occlusions can contaminate predictions. The camera/light assumptions are simplified relative to a path tracer. Detail accuracy and photorealistic skin reflectance are separate questions.

## Code mapping: a crucial trap

In decalib/deca.py, the value named uv_texture is formed from albedo multiplied by UV shading. That value is **already lit**. Our skin shader must receive the albedo output, not a baked shaded texture mislabeled as intrinsic color.

The released displacement includes predicted UV height plus a fixed displacement term. Its detailed normals come from reconstructed UV geometry. They must be interpreted in their actual frame; passing them to Blender as tangent normals without rebaking is wrong.

Read encode, decode and displacement2normal; follow coarse vertices → dense UV surface → displacement → normals → shading. Keep the underlying FLAME model/version, UV atlas and displacement units.

## TrueFace3D use and improvements

DECA is our first detail candidate, with its own original coarse geometry used only for extracting a compatible detail prediction. RealDenseFace remains the source of the final coarse mesh. Transfer/recompute detail on that mesh, mask unsupported regions, and test for doubled geometry or self-intersection.

Possible experiments:

- Gate displacement by visibility, uncertainty and local image evidence; compare against unbounded DECA detail.
- Separate albedo marks from geometric folds using albedo-assisted detail supervision. A mole should not automatically become a bump.
- Penalize unnecessary movement in reliable coarse geometry while fitting detail under multiple held-out light directions.
- Compare a global detail latent with regional residuals using the same training data and parameter budget.

Measure scan distance, normal angular error and held-out relighting, not only a sharper shaded screenshot. These are proposals, not verified improvements.

## Explain it back

Which variables stay identity-specific when a smile changes? What is the difference between albedo and uv_texture in the implementation? Why must displacement units and normal coordinates be declared?

