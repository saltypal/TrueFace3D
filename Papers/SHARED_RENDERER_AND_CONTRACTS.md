# How all the face modules enter one rendering environment

Research snapshot: **2026-10-10**. This is a proposed engineering contract, not implemented software. Read [the master plan](TRUEFACE3D_MASTER_PLAN.md) for the execution order and [the paper index](PAPER_INDEX.md) for the underlying sources.

## The direct answer to the shared-renderer question

Each expert should produce a **3D asset attribute**, not a separately lit photograph. The geometry module produces a surface and camera. The skin module supplies material maps. Detail supplies small geometric changes. Eyes supply geometry, pose, and optical material parameters. Put those outputs into one scene with one coordinate system, one camera, one lighting setup, and one color-management pipeline. The shared renderer then produces all pixels.

Like building a toy person: one team makes the head, another paints its skin, another inserts its eyes. Put the assembled person under one lamp and photograph it. If every team instead photographs its part under a different lamp, pasting the photos together cannot establish correct shadows, occlusion, or reflections.

The physics is

\[
I=\mathcal R(G,\Delta G,M,E,C,L),
\]

where G is base geometry, ΔG detail, M materials, E eye assets, C camera, and L lighting. **R is one rendering model with multiple material types.** Skin and cornea need different material equations while still sharing the same scene and light. Separate material shaders do not require separate image renderers.

## Two implementations of one scene contract

Use a differentiable rasterization path for learning/fitting, and a path-tracing path for final visual assessment. They consume the same asset semantics but approximate light transport differently; they will not produce identical pixels.

| Option | What it supplies | Our proposed use | Limitation |
|---|---|---|---|
| nvdiffrast + our PyTorch shader | Rasterize/interpolate/sample/antialias with gradients | Initial fitting and controlled material experiments | Cameras, lighting, BRDF, shadows and optical eyes must be supplied |
| Blender Cycles | Scene assembly and path-traced graphics rendering | Reference final renders, skin scattering and corneal refraction | Does not automatically differentiate our Python training losses |
| Mitsuba 3 | Programmable physically based differentiable rendering | Optional later inverse-rendering research | Separate dependency/compute/material compatibility audit |
| A learned NeRF/Gaussian avatar renderer | Learned representation-specific appearance | Comparator or later video branch | Not interchangeable with a portable mesh/PBR scene |

These roles follow the [nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast/), [Cycles shader documentation](https://docs.blender.org/manual/en/4.5/render/cycles/osl/index.html), and [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/en/stable/). Pin a tested engine version before implementation. The 2019 [Mitsuba 2 paper](notes/26_Mitsuba2.md) explains its design; its Enoki API is not the current Mitsuba 3/Dr.Jit API.

## Asset and coordinate contracts

All shapes below are **our proposed interface**, not a claim about every upstream network tensor. Store arrays with metadata; a square PNG cannot describe its own units or coordinate frame.

| Artifact | Required fields | Convention |
|---|---|---|
| FaceObservation | Original image, crop, original↔crop transform, landmarks/masks, validity | Never discard original coordinates or silently change image alignment |
| GeometryAsset | Vertices N×3, triangle indices F×3, UV vertices Nu×2, separate UV-face indices F×3, parameters, model version/hash | A declared canonical head frame; convert known units to meters at boundary |
| Camera | Projection type, K or focal parameters, R/t, crop affine, image size | Explicit world→camera convention; renderer adapter owns conversions |
| MaterialMaps | Albedo H×W×3, roughness H×W×1, dielectric reflectance representation, masks, provenance | Albedo in declared linear RGB; non-color scalar/vector maps stay numerical |
| DetailMaps | Displacement H×W×1 with units, normals H×W×3 with frame/channel convention, confidence | Displacement affects geometry; normal maps affect shading |
| EyeAsset | Separate sclera/cornea/iris/pupil geometry or procedural parameters, transforms, textures, optics | Bound to FLAME eye/joint frame; local→head transform explicit |
| Illumination | HDR environment or analytic lights, orientation, exposure/white balance | Shared by skin, eyes and ears; no independent per-component exposure |
| RenderPacket | Linear RGB, display RGB, alpha, depth, normals, UV, material/component passes | Preserve native render before display enhancement |

**UV topology:** many meshes share position vertices across seams but have different UV vertex indices. Do not assume one UV coordinate per position vertex. Tangents are defined by UV charts and triangle orientation; a different UV map changes tangent-space normals even if the 3D surface looks similar.

**Units:** a source mesh in millimeters needs a 0.001 conversion to meters; a source already in meters does not. FLAME model units and downstream preprocessing must be verified from the pinned assets. Never infer units only from a visually plausible head. Light distances, displacement, cornea thickness, and SSS radius depend on this choice.

**Cameras:** weak perspective and full perspective are distinct. If the source reconstruction supplies a fitted camera with uncertain focal length, preserve that estimate and its ambiguity. Do not invent real-camera calibration. A known calibrated training camera is privileged supervision, not necessarily an inference-time input.

**Color:** convert encoded RGB to renderer-linear RGB exactly once. Exposure/white balance are camera/light variables. Scalar roughness, specular coefficients, masks, displacement, and normal channels use data interpretation, not sRGB transfer. Display tone mapping belongs after light transport. Do not compare tone-mapped renders to linear targets without matching the pipeline.

## Baseline assembly, step by step

1. Reconstruct the unmodified input with RealDenseFace. Save its β/ψ/pose/camera/crop and dense C/D/Lc/Ld maps. Render a constant-material mesh to verify geometry first.
2. Predict diffuse albedo using the pretrained OpenDelight image prior if its checkpoint/preprocessing audit passes. Invert its alignment back to observation coordinates.
3. Project visible albedo onto the reconstructed UV surface using z-buffer visibility and the same camera. Save observed, unobserved, occluded, saturated, and unreliable masks. Fill missing regions conservatively; tag completion as inferred.
4. Begin with declared constant dielectric skin roughness/specular/SSS defaults. These are graphics assumptions, not recovered subject-specific values. A later material model may replace them after evaluation.
5. Add DECA-style detail after FLAME2020→our pinned model compatibility checks. Keep identity detail and expression detail distinct. Recompute geometric normals after displacement.
6. Fit the [parametric eye model](notes/22_ParametricEyes.md) concept to visible iris/limbus cues. Use conservative generic optics for hidden structures. Separate iris texture from corneal highlight so a highlight is not painted into the iris.
7. Assemble mesh/material/eyes in one scene. Eyes sit inside sockets; lids occlude them and shadows belong to that geometry. The wet cornea reflects the same environment as the skin.
8. Render input view, held-out views, and several fixed lighting probes. Keep every component pass and baseline render for diagnosis.

**Avoid detail duplication:** if a normal map already derives from a displacement map, applying both at full strength can count the same wrinkle twice. Choose displacement for resolved geometry and add only a residual normal map for finer unresolved variation. Document the partition by frequency/resolution.

## Transferring from a different topology

MARCUS uses HiFi3D++, some methods use BFM, and FaceScape has its own mesh. A PNG resize is not a topology adapter.

Register the source/target surfaces in a common frame; compute source-triangle and barycentric correspondences for target surface samples. Transfer colors/scalars using those correspondences. For normals, decode source tangent normals to the common object frame, transform correctly under the registration, and rebake into the target tangent frame. Reconstruct normals from geometry where correspondence derivatives are unreliable. Store registration residuals and exclude bad regions. Displacement should be rebaked relative to the target base surface, not copied as though the bases were identical.

Accept the adapter only after a checkerboard, an asymmetric painted landmark, a known bump, and a rotated-light test pass. A mesh that looks right from the front can still have mirrored UVs or inverted normals.

## Physical models and what each term means

Skin appearance needs diffuse response, specular reflection, and sometimes subsurface transport. Roughness controls highlight spread; dielectric reflectance controls reflection strength. Normal/displacement changes alter highlight direction and surface silhouette respectively. Metallic should stay zero for ordinary skin. A shadow is a visibility/light effect, not permanent albedo.

For local surface reflection,

\[
L_o=\int f_r L_i V\max(n\cdot\omega_i,0)\,d\omega_i.
\]

SSS transports energy from an entering surface point to another exiting point; a local BRDF cannot fully describe it. The [2001 transport paper](notes/28_SubsurfaceTransport.md) gives the principle. Modern engine models have their own implementation/parameters. Do not blindly treat a paper's specular albedo as Blender's “Specular IOR Level,” a Blinn exponent as GGX roughness, or a paper's SSS weights as a Cycles radius.

Corneal transmission/refraction requires a layered eye and refractive optics. The initial differentiable fitting path may use an approximate eye shader. Label that approximation explicitly; compare it with the path-traced reference before drawing conclusions about optics.

## Bounded joint refinement, after independent baselines

Initialize experts independently, assemble once, then minimize a proposed objective such as

\[
E=\lambda_pE_{photo}+\lambda_lE_{landmark}+\lambda_iE_{identity}
+\lambda_gE_{geometry-prior}+\lambda_mE_{material-prior}
+\lambda_sE_{seam}+\lambda_vE_{supervised-view/light}.
\]

All residuals use declared masks/units; normalize reductions so a large image or dataset does not accidentally dominate. Alternate lighting/camera, material, eyes, and **bounded** geometry/detail updates. Freeze confident identity geometry initially. Photometric loss alone can paint away geometry errors or trade dark skin against dim light. The prior and permitted update budget need ablation, not arbitrary justification.

Extra training views/lights can support E_view/light; **at strict one-image inference those unavailable observations cannot be used**. Synthetic rendered probes are useful consistency checks but are not measurements of the actual hidden person. Supervised ground-truth geometry/material losses belong to training/evaluation only.

## Diagnostics and acceptance checks

| Probe | Expected evidence | Failure it reveals |
|---|---|---|
| Reproject landmarks into original uncropped image | Same convention and spatial agreement | Crop or camera transform error |
| Gray material, moving directional light | Stable surface, continuous shading | Inverted normals, misplaced detail |
| UV checkerboard and asymmetric landmark | Correct semantic location across seams | UV flips and topology mismatch |
| Albedo-only pass | No deliberate lighting/highlight baked into texture | Shading/material entanglement |
| Same face under three environments | Identity and marks remain attached | Generative instability or baked lights |
| Close-up eyes in front/oblique view | Lids occlude eyes; reflection follows light | Floating eyes, painted highlights |
| Native versus enhanced image | Both retained and evaluated | Enhancement hiding asset defects |
| Full scene versus isolated components | Coherent shadows/reflections | Separate inconsistent light/exposure |

Acceptance numbers must be set using a reproduced baseline and valid data. Do not invent an achievable millimeter error, frames per second, or memory footprint before measuring the complete system.
