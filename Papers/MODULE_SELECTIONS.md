# TrueFace3D: chosen modules and implementation gates

**Historical candidate selection.** The [2026-10-23 execution plan](ExecutionPlan/README.md) reopens comparisons and governs the fourteen-day scope. These selections are initial reproduction candidates; a matched comparison may justify a documented replacement. Availability claims below retain their original audit date.

Decision date: **2026-10-11**. Based on the 44-paper archive, original-paper notes, alphaXiv inspection of DECA's actual detail code, and a fresh check of official releases. This is an **implementation selection**, not a measured quality ranking. No selected network has yet been run in TrueFace3D, and checkpoint downloads, license eligibility, peak memory and runtime remain to be tested.

The objective remains one RGB photo → a parametric, identity-preserving face asset that can be rendered from different viewpoints and under different lights. Short video, ears and enhancement are later phases. The existing [master plan](TRUEFACE3D_MASTER_PLAN.md) supplies the full execution protocol; this document resolves its candidate choices.

## What “best” means here

Prefer a method that accepts one photo, preserves our FLAME geometry, exposes useful asset attributes, has an identifiable official implementation/checkpoint, can be inspected and adapted, and can be evaluated under our resource constraints. Publication year and attractive example renders are insufficient selection criteria. There is no head-to-head experiment on a common dataset in our project yet.

Release evidence has three separate meanings: source code exists; the author supplies a checkpoint download instruction; the checkpoint has actually been acquired and successfully executed. This selection verifies the first two where stated. It does not establish the third.

## The selected first implementation

| Area | One selected method/reference | What we use | Availability and principal limitation |
|---|---|---|---|
| Parametric geometry | **RealDenseFace, 2026, ViT-B + FLAME2023** | Dense correspondence/depth priors and CUDA Gauss–Newton fitting | Official inference and checkpoint instructions; FLAME assets are separate and CUDA is required |
| Skin pigment / diffuse albedo | **OpenDelight, SIGGRAPH 2026** | Pretrained single-image delighting network; project its output into FLAME UV | Official inference/checkpoint instructions; the complete published capture pipeline uses a sequence |
| Surface detail | **DECA, SIGGRAPH 2021** | Detail encoder/decoder and displacement mechanism | Code, checkpoint instructions and training code; FLAME version/UV/conditioning adaptation required |
| Eyes | **Lightweight Eye Capture Using a Parametric Model, SIGGRAPH 2016** | Explicit parametric-eye design and visible-evidence fitting | Paper available; complete implementation and thirty-eye database not verified. Our first module is a simplified adaptation |
| Differentiable rendering / fitting | **nvdiffrast, SIGGRAPH Asia 2020** | Rasterization, interpolation, texture sampling and antialiasing | Public library; camera, shader, light model and loss code are our responsibility |
| Skin light response | **Blender Cycles Principled BSDF** | Dielectric highlights, roughness and subsurface scattering | Available engine shader; first material parameters are declared defaults, not personal measurements |
| Final rendering environment | **Blender Cycles** | One scene containing the face, detail and optical eye objects | Public production path tracer; it does not supply our PyTorch training gradients |
| Assembly and coupling | **Our shared scene contract; MARCUS as closest selected integration reference** | Consistent coordinates, camera, color, maps and staged fitting | Engineering work we implement; joining modules alone is not a paper contribution |

Every selected research paper already has an original PDF and teaching note in this archive. Cycles is an implementation choice supported by official documentation, not a newly claimed research paper. [Disney's shading reference](notes/27_DisneyBRDF.md) and [subsurface transport](notes/28_SubsurfaceTransport.md) explain the principles; the current Principled shader is not an exact implementation of either old paper.

## 1. Geometry: RealDenseFace

**Paper and code:** [teaching note](notes/01_RealDenseFace.md), [original](pdfs/01_RealDenseFace.pdf), [official implementation](https://github.com/LinzhouLi/RealDenseFace).

Its network predicts where UV surface points correspond in the image, their relative depth and separate uncertainty quantities. The solver fits FLAME to this geometric evidence. The mesh is the result of parameterized fitting; the network does not directly predict a complete realistic skin-and-eye asset.

**Use the released ViT-B single-image path first.** The official README identifies it as the default for single-image fitting. ViT-S is a resource fallback to benchmark if necessary, not a second simultaneous geometry expert. Retain the original `flame2023.pkl` requirement for initial reproduction. FLAME Open is a possible later substitution requiring basis/model validation; see the [release audit](evidence/RELEASE_AUDIT_2026-10-11.md).

**Why this selection:** it preserves the user's backbone and directly supplies FLAME parameters plus a camera and dense geometric evidence. MICA is useful identity initialization but does not supply the complete appearance pipeline. Pixel3DMM is a meaningful geometry comparator, not a reason to change the agreed baseline before reproduction. NPHM/Pix2NPHM introduce a different surface representation and adapter burden.

**Contract:** original image and coordinate ledger → FLAME parameters, vertices/faces/UV, camera, priors and uncertainty. RealDenseFace owns coarse identity, expression and pose in the first integrated system.

**Gate:** acquire permitted assets, pin source/configuration, run prescribed batch-one inference on a known device, export/reload a finite mesh, prove projection back into the original image, and record time/memory and geometry error under a valid protocol. A flat-material render is the first visual check. Understand network → solver → FLAME before modifying the geometry method.

## 2. Albedo: OpenDelight

**Paper and code:** [note](notes/12_OpenDelight.md), [original](pdfs/12_OpenDelight.pdf), [official repository](https://github.com/yxuhan/OpenDelight), [pretrained inference instructions](https://github.com/yxuhan/OpenDelight/blob/main/doc/TEST.md).

Albedo is the surface's pigment-like reflectance, separated as far as possible from captured shadows and highlights. OpenDelight uses a ViT image encoder, source-conditioned tokens and a decoder to predict image-space albedo, with a separate detail-enhancement stage. We select its pretrained image prior, not the full smartphone-video capture pipeline.

**Why this selection:** a recent released predictor addresses a specific component we can isolate. OpenDelight's image output can be projected onto our fixed mesh without adopting another geometry backbone. MoSAR and AvatarMe++ provide richer material stacks but involve their own geometry/material conventions and captured-data dependencies. MARCUS predicts a richer bundle but requires substantial topology and diffusion-runtime adaptation; it has a separate selected upgrade role below.

The current TEST document explicitly says the downloadable network was trained on FaceOLAT plus private Light Stage scans. The newer README's public Ava256-Scan release does not prove that the downloadable checkpoint was retrained on that dataset. Keep those provenance claims separate.

**Contract:** correctly aligned original crop → diffuse-albedo image → invert alignment → visibility-aware FLAME UV albedo, valid-texel mask and provenance/confidence. It does not directly output a complete UV atlas, personal roughness, specular response, scattering parameters or eye textures.

For the first UV completion, use a declared conservative fill/symmetry prior and mark unobserved areas. Completion is a separate algorithmic assumption, not an OpenDelight result. Do not mirror distinctive observed marks onto the invisible cheek and call them measured.

**Gate:** landmarks and UV projections align; albedo is interpreted in the correct color space; relighting does not apply captured shading twice; observed marks remain recognizable; unseen areas are labeled. Compare raw projected color versus predicted albedo on the same geometry, camera and lights. Full-image perceptual improvement alone is insufficient evidence of material accuracy.

## 3. Detail: DECA

**Paper and code:** [note](notes/07_DECA.md), [original](pdfs/07_DECA.pdf), [official repository](https://github.com/yfeng95/DECA), [actual detail path](https://github.com/yfeng95/DECA/blob/master/decalib/deca.py).

The detail encoder extracts an identity-related latent code. The decoder combines that code with expression and jaw pose to predict a UV height/displacement field. DECA also adds a fixed UV displacement prior. A displaced surface and its normals can then be rendered under a moving light.

**Why choose a 2021 model:** it has inspectable source, supplied checkpoint instructions and FLAME-related geometry. That makes it the best first compatible detail baseline in our surveyed set, not a claim that it has the strongest detail quality in 2026. HiFace's static/dynamic detail formulation is attractive, but its [official page](https://project-hiface.github.io/) did not expose an executable model/checkpoint release in this audit. HRN has [released inference](https://github.com/youngLBW/HRN), but a different coarse model/topology and no released training code in its current README. EMOCA targets expression improvement while retaining DECA-derived detail; replacing the expression owner is outside this module's first scope.

**Reuse the detail mechanism, not DECA's entire reconstruction.** RealDenseFace remains the coarse mesh/camera owner. DECA's coarse encoder may provide the conditioning needed by its pretrained detail decoder; its identity shape, albedo, lighting and camera must not silently overwrite the baseline.

FLAME2020/2023 compatibility must be demonstrated. Equal coefficient indices or matching array dimensions do not prove equal expression bases. Do not simply truncate RealDenseFace expression coefficients into DECA's decoder. Initially retain valid DECA-native conditioning, record any expression disagreement, and validate mapping/reprojection before using the target model's controls for animation. Check UV charts, displacement units, fixed-prior inclusion and seams. DECA's shaded `uv_texture` is not diffuse albedo, and its generated normals cannot be assumed to be tangent-space normal maps.

**Contract:** original image plus validated conditioning → UV displacement and detail confidence → displaced/subdivided target surface and recomputed normals. There is one owner of fine height in the baseline. Avoid counting the same crease in both displacement and a residual normal map.

**Gate:** geometry identity is preserved, displacement orientation/scale and UV transfer are correct, normals move correctly with the surface, and expression changes do not produce unexplained sliding wrinkles. Compare detail on/off under identical scene settings and use scan-derived geometry evidence where available. Predicted tiny pores are not guaranteed person-specific anatomy.

## 4. Eyes: explicit parametric fitting

**Reference:** [note](notes/22_ParametricEyes.md), [original](pdfs/22_ParametricEyes.pdf), [official publication page](https://cgl.ethz.ch/publications/papers/paperPas16a.php).

This paper is the best design reference in our surveyed set for a portable, controllable eye component with single-image fitting. Its compact eye shape model, iris synthesis and procedural veins explain what sparse observations can constrain and what must come from a prior. The 2014 capture method requires richer acquisition; the surveyed DeNeRF method depends on per-person multi-camera/gaze observations and a neural appearance representation.

**The release gap is real:** the official publication page supplies the paper/video/BibTeX; a complete public executable implementation and the thirty captured eyes were not verified. We will implement a simplified explicit model inspired by the paper. We will not describe it as reproducing the paper's learned eye database or assume a downloadable pretrained eye expert exists.

Start with sclera/eyeball geometry, a corneal shell, iris/pupil geometry, gaze and pupil controls, and observed iris color where visible. Place eye objects relative to verified FLAME eye/socket coordinates. FLAME eyelids continue to provide occlusion. Treat hidden anatomy, unresolved iris fibers and generic optical settings as inferred defaults.

**Contract:** head geometry/camera plus visible iris/limbus evidence → separate eye objects, head-frame transforms, gaze/pupil controls, materials and evidence masks. All eye objects receive the same environment as the skin; highlights belong to optical rendering rather than painted iris color.

**Gate:** projected iris boundaries agree with visible evidence, eyes fit sockets, lids occlude correctly, no obvious interpenetration occurs, and corneal highlights respond coherently to camera/light changes. Closed eyes, glasses and very small eye crops must reduce fitting confidence. A low-resolution face photo cannot recover a reliable iris fingerprint or exact corneal anatomy.

## 5. Renderer and skin shading

**Fitting reference:** [nvdiffrast note](notes/25_Nvdiffrast.md), [original](pdfs/25_Nvdiffrast.pdf), [official library documentation](https://nvlabs.github.io/nvdiffrast/).

**Final rendering:** [official Cycles overview](https://www.blender.org/features/rendering/), [Principled BSDF documentation](https://docs.blender.org/manual/en/5.1/render/shader_nodes/shader/principled.html). Pin an installed and tested Blender version during implementation; these documentation links do not imply an engine version has already been installed.

We choose two implementations for two distinct jobs. nvdiffrast supplies differentiable graphics primitives for camera/material fitting and controlled experiments. It has no built-in camera or skin/lighting model. Cycles supplies production path tracing for the assembled face, including skin scattering and corneal reflection/refraction. Its rendering path does not automatically provide PyTorch gradients.

For skin, select Cycles' current Principled material with subsurface scattering. Start with OpenDelight-derived base color, DECA-derived geometric detail, a nonmetallic dielectric response and declared conservative roughness/scattering defaults. Do not optimize every unknown optical variable independently from one image. Exposure, illumination, albedo and roughness can compensate for one another.

Current Principled documentation describes an OpenPBR-based shader; its exact sockets and parameter conventions must be taken from our pinned version. Disney2012 and the 2001 scattering paper are conceptual sources. Do not map another model's specular albedo directly to “Specular IOR Level” or treat its displacement PNG as unitless.

**Why these choices:** nvdiffrast exposes the low-level visibility and UV operations we need; Cycles gives a mature scene and optical material environment. Mitsuba 3 is a strong future inverse-rendering alternative but adds another material/backend interface before the baseline has been validated. A NeRF/Gaussian renderer changes the asset representation and does not automatically produce a portable parametric mesh/material asset.

**Contract:** one canonical scene packet → fitting renders/gradients or final rendered views and diagnostic passes. Cameras, geometry units, UVs, normal frames, color management and light/exposure semantics are shared. The rasterizer's simplified lighting and Cycles' transport will differ; compare supported terms explicitly, not pixel identity.

**Gate:** first verify a plain mesh with the same camera. Then check UV colors, normal orientation, displacement, material response, eyelid occlusion, skin scattering and eye optics separately. Export a native `.blend` scene and ordinary mesh/material assets. Generic GLB export may not preserve Cycles skin/eye optics; test exports instead of promising identical renders in every viewer.

## 6. Assembly: one scene, one owner per quantity

The scene adapter is our implementation. MARCUS is the selected recent integration reference and comparator, not a drop-in plugin that automatically joins our chosen experts.

The actual first data flow is:

~~~mermaid
flowchart TD
 I[One original RGB image] --> G[RealDenseFace: FLAME mesh and camera]
 I --> A[OpenDelight: image albedo]
 I --> D[DECA: detail code and displacement]
 G --> U[Visibility-aware projection into FLAME UV]
 A --> U
 G --> E[Explicit parametric eye fitting]
 I --> E
 G --> S[Canonical scene assembly]
 U --> S
 D --> S
 E --> S
 P[Declared skin material defaults] --> S
 S --> R[nvdiffrast fitting path]
 S --> C[Blender Cycles final renders]
~~~

Start by assembling fixed outputs. Fit camera/exposure/light on reliable visible regions with geometry and predicted maps initially constrained. Test a bounded consistency optimizer later; do not train every predictor end to end or change identity shape merely to reproduce a shadow. Shared rendering itself is compulsory engineering. A confidence-aware coupling algorithm would be a research hypothesis requiring ablations against fixed assembly, unweighted refinement and relevant prior art.

The [exact contracts](SHARED_RENDERER_AND_CONTRACTS.md) remain authoritative for proposed field semantics. Codex handles repetitive adapters and provenance plumbing; the user and Codex review network outputs, optimization variables and coordinate conversions together before integration.

## Selected later modules, without competing defaults

| Later area | One selected method | Role and adoption gate |
|---|---|---|
| Richer skin material prediction / UV completion | **MARCUS, ECCV 2026** | Preferred upgrade/comparator for a joint PBR bundle; validate memory, topology, semantics and identity before reuse |
| Dedicated ear reconstruction | **AudioEar, AAAI 2023** | Reconstruct visible ears; validate dedicated-ear-to-FLAME attachment and checkpoint dependencies |
| Input face restoration | **CodeFormer, NeurIPS 2022** | Optional fidelity-controlled raw/restored comparison; accept only if reconstruction/identity evidence improves |
| Final rendered-image enlargement | **Real-ESRGAN, ICCV Workshops 2021** | Optional presentation branch after native rendering; quantify invented detail and multi-view inconsistency |

**MARCUS is the selected fuller-material candidate, not a promise of immediate integration.** [Its official release](https://github.com/Luh1124/MARCUS-Avatar) supplies inference, model download instructions and PBR asset export. The inspected geometry uses HiFi3D++, and the pipeline includes a large LongCat diffusion backbone. There is no measured TrueFace3D T4 memory result. Read its [note](notes/13_MARCUS.md) and [original](pdfs/13_MARCUS.pdf).

Before transfer, prove surface registration and UV correspondence; transfer albedo/scalars and convert normals/displacement with correct frames and units. Its joint outputs are coupled. Replacing only OpenDelight albedo or only DECA displacement inside that bundle requires an explicit consistency experiment; independently selecting the prettiest map from each method is not justified. Compare its coherent bundle against the first modular baseline on fixed geometry, then decide which attributes have one final owner. Do not casually duplicate detail or make its stronger material predictor secretly replace the geometry backbone.

AudioEar's [official code](https://github.com/seanywang0408/AudioEar) supplies pretrained encoder/depth-model instructions, but its current usage also references a trained reconstruction checkpoint. Inspect the actual downloadable files; the README alone does not establish that every required final asset is accessible. Its custom ear model and manual attachment work are not automatic FLAME compatibility. [Note](notes/24_AudioEar.md), [original](pdfs/24_AudioEar.pdf).

[CodeFormer](https://github.com/sczhou/CodeFormer) provides released weights, training code and a fidelity control. Choose it over GFPGAN as the first restoration experiment because that control supports a deliberate fidelity sweep. Never infer that prettier pixels improve 3D truth. Preserve the raw branch, crop transforms and parameter differences. [Note](notes/36_CodeFormer.md), [original](pdfs/36_CodeFormer.pdf).

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) has official pretrained models and tiling support. Render at adequate native resolution first. Its output changes a 2D image, not the reconstructed face asset or verified geometry. [Note](notes/38_RealESRGAN.md), [original](pdfs/38_RealESRGAN.pdf).

## Dataset decisions

**Primary paired-image/geometry source: FaceScape.** It provides real multiview camera images, calibration and corresponding reconstructed meshes; use one image as model input and reserve other views/mesh as supervision or evaluation. This preserves monocular inference. Its original topology is not FLAME and ordinary captured texture is not measured albedo. [Official release](https://github.com/zhuhao-nju/facescape), [note](notes/29_FaceScape.md).

**First geometry benchmark: NoW**, using the permitted validation/test protocol and its official evaluation. It evaluates face geometry from real images/scans; it is not a complete skin/eye/material benchmark. [Official benchmark](https://now.is.tue.mpg.de/), [archive note](notes/34_RingNet_NoW.md).

**Albedo/material development:** selected permitted FaceOLAT examples and OpenDelight's public Ava256-Scan release are candidates if training becomes necessary. OLAT uniform-light targets are approximate albedo; full physical-material evaluation needs appropriate measured labels/held-out lights. Private training scans are not available merely because a pretrained model is public.

RealDenseFace already used FaceScape, NPHM and Ava-derived data. Exposure and identity splits must be audited before claiming an independent test or novelty from adding those sources. Different cameras of the same training identity are not automatically an unseen-identity evaluation. No single dataset in this plan supplies trustworthy ground truth for every skin, detail and optical-eye property. Use the [dataset protocol](DATASETS_AND_TRAINING.md); do not invent missing labels.

## Module-by-module execution and checkpoints

| Order | Deliverable | Dependency | Required evidence before continuing |
|---|---|---|---|
| 0 | Pinned assets, source revisions and device/environment record | Release/access audit | Actual essential files obtained, requirements understood, device identified |
| 1 | Unmodified RealDenseFace reconstruction and plain geometry render | 0 | Valid mesh/camera, coordinate roundtrip, exports, measured resource use |
| 2 | OpenDelight image albedo and visibility-aware UV maps | 1 | Correct alignment/color/masks; controlled same-scene comparison |
| 3 | DECA-derived detail on the preserved target mesh | 1–2 | Version/UV/units/conditioning validated; no duplicate or sliding detail |
| 4 | Explicit eye assets and fitting | 1, shared camera | Socket/lid agreement, gaze and material tests; inferred regions labeled |
| 5 | Complete Cycles face scene and export | 2–4 | Coherent light/occlusion, diagnostic passes, new-view/light tests |
| 6 | Bounded consistency refinement, then MARCUS comparison/transfer | 5 | Fixed-baseline ablations, measured memory, compatible maps and ownership |
| 7 | Ears and optional restoration/enlargement | Valid native baseline | Separate evidence for each improvement |

Minimal renderer plumbing starts during geometry diagnostics and albedo projection; it does not postpone the first mesh milestone or authorize implementing every module at once. Supporting code is Codex-owned; core understanding and implementation decisions are reviewed together. Follow each module's input/output contract and gate, including documenting failures.

Before training, follow the existing modular Git plus portable Local/Colab/Kaggle workflow. Do not assume legacy DECA, current RealDenseFace, OpenDelight and the latest renderer dependencies fit one unchanged Python environment. Isolate upstream inference if needed and exchange explicit artifacts. TPU support is not implied by CUDA-based source code.

The next concrete task is **Phase 0 plus the unmodified RealDenseFace single-image baseline**, not a new fusion network. This selection closes model choice for the first stack; runtime feasibility, topology transfer and resulting quality remain implementation work.
