# 13 — MARCUS: cascaded diffusion for a complete material bundle

**Monocular Avatar Reconstruction via Cascaded Diffusion Priors and UV-Space Differentiable Shading**, Li et al., arXiv 2606.28144; official repository labels ECCV 2026. [PDF](../pdfs/13_MARCUS.pdf), [paper](https://arxiv.org/abs/2606.28144), [code](https://github.com/Luh1124/MARCUS-Avatar). Read §3, Figures 2–5, §4 and appendix. Snapshot: 2026-10-10.

## Intuition

A face photograph is incomplete and already lit. MARCUS treats reconstruction like restoring a partly visible unfolded mask: first fill its blank areas, then remove the captured illumination, then infer different physical surface properties together. A shared generative backbone is adapted for these tasks instead of training several giant models independently.

This is unusually close prior work to our proposal. A paper contribution cannot be “combine geometry and several material models” without explaining a difference from MARCUS, MoSAR and related pipelines.

## Architecture and forward trace

~~~mermaid
flowchart LR
 I[Single portrait] --> G[ConvNeXt V2 plus frozen DINOv3 geometry regressor]
 G --> M[Parametric mesh and masked UV texture]
 M --> P[Diffusion inpainting LoRA]
 P --> H[Light homogenization LoRA]
 H --> J[Joint intrinsic LoRAs with cross intrinsic attention]
 J --> A[Albedo]
 J --> N[Normal]
 J --> K[Packed roughness specular displacement]
 A --> SR[Super resolution and asset export]
 N --> SR
 K --> SR
~~~

The **MSSF** encoder combines ConvNeXt V2 local features with frozen DINOv3 semantic features to regress geometry. The inspected implementation loads **HiFi3D++ topology**, not RealDenseFace's FLAME2023 topology. Before the material pipeline, image colors are unwrapped into that mesh's UV atlas with a visibility mask.

The inpainting adapter learns missing texture completion. A second adapter normalizes illumination. For intrinsic material prediction, separate LoRA branches handle albedo, normals and a packed map. Cross-intrinsic attention allows these modalities to share evidence. It reduces contradictory outputs such as a painted wrinkle that has no corresponding geometric detail.

LoRA means adapting a large model using a low-rank weight change \(W'=W+BA\), where the inner rank is much smaller than the original matrix dimensions. Small adapters do not eliminate the memory needed to run the frozen backbone. The backbone is LongCat-Image-Edit.

## Mathematics and solved reasoning

A diffusion model learns to denoise corrupted latent data across noise levels. Joint branches exchange information through attention of the form

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}(QK^\top/\sqrt d)V.
\]

Cross-intrinsic attention uses keys/values shared across material modalities. This is different from generating each map with unrelated random seeds.

The UV-space shading constraint links predicted material maps through a reflectance model. For a microfacet specular term, the familiar structure is

\[
f_s=\frac{D(h)F(v,h)G(l,v)}
{4(n\cdot l)(n\cdot v)}.
\]

Distribution \(D\), Fresnel \(F\), masking \(G\) and the normal \(n\) jointly determine the highlight. The formula above is an explanatory form; consult the PDF for its exact GGX parameter conventions and loss weights.

**Illustrative example:** a nose highlight can be reproduced by increasing specular intensity, reducing roughness, bending the normal or brightening the light. Four independent predictors can select four incompatible explanations. Joint attention and shading losses reduce disagreement, but one photograph still does not uniquely identify the true parameters.

## Training data and resources

The paper uses fewer than 100 real professional PBR scans, plus a much larger derived data engine using FFHQ and CelebAMask-HQ and HDRI relighting. “Fewer than 100 scans” does not mean the model saw fewer than 100 faces or was trained cheaply from scratch.

The public derived datasets contain about 69,999 FFHQ samples and 29,994 CelebAMask-HQ samples. The README lists separate rendered and UV-unwrapped configurations totaling over 1TB, with streaming and shard selection. Generated reflectance/material labels are not interchangeable with independently measured scan ground truth.

The paper uses rank-32 adapters and H100 80GB training. Its reported generation settings and the currently inspected code defaults differ, including diffusion step counts. Record the chosen upstream version and inference settings. Free T4 feasibility is unverified; neither batch size one nor LoRA alone guarantees it fits.

## Evidence, release and limitations

The authors ablate geometry feature fusion and joint intrinsic prediction. They report advantages in geometry and rendering comparisons, but generated 4K maps include a super-resolution stage. Upsampling does not measure the individual's hidden pores.

The public inference release supplies code and downloadable topology/checkpoint assets. Training/evaluation reproduction remains a separate audit. The inspected pipeline's export includes albedo, normals and roughness/specular material information. A separate displacement map is not automatically actual displaced geometry in a GLB.

Different assets have different licensing obligations: repository code, LongCat weights, HiFi3D++ topology, source photos and derived data. Do not flatten them into “MIT means everything is reusable.”

## Our integration decision

Keep RealDenseFace as requested. Treat MARCUS as a strong comparator and an optional material branch after a compatibility/compute gate. It must not silently replace our geometry baseline.

For HiFi3D++ → FLAME transfer, register source and target surfaces; interpolate scalar/color maps using surface correspondences; reorient normals into a shared object frame; then rebake target tangent-space normals. A UV resize or direct image copy is incorrect even if both textures are square.

Read pipeline.py, longcat_image/face3d_recon/parametric_face_model.py and export utilities. Archive topology identifiers, map channel meanings and normal orientation. A small known-texture sphere/checkerboard transfer test should precede faces.

## Improvement experiments

1. Preserve confident observed UV colors while restricting diffusion completion to unobserved/low-confidence regions. Compare against whole-map generation.
2. Add a geometry-conditioned confidence head and shared residual optimization; keep high-confidence FLAME identity fixed. Compare with unweighted joint fitting.
3. Distill materials from a teacher into a small FLAME-native student, labeling teacher-generated targets as pseudo labels. Evaluate on separate measured materials.
4. Quantify seed variation: several samples of one image should not alter nose shape, eye spacing or recognizable skin marks.

These proposals compete with an already integrated, physically constrained system. The useful novelty is a tested identity/reliability mechanism, not rebranding its component chain.

## Explain it back

Why do small adapters still need a large GPU? Why is 4K not proof of recovered detail? Why do vector normal maps need more than UV coordinate interpolation?

