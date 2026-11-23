# Literature synthesis: a monocular parametric face with coherent appearance

**Curated research snapshot: 2026-10-10; 44 original documents.** This is a broad project-focused survey covering recent 2026 work and necessary foundations. It is not a claim to include every published facial reconstruction paper. [Index and PDFs](PAPER_INDEX.md), [training/data comparison](DATASETS_AND_TRAINING.md), [novelty assessment](NOVELTY_ANALYSIS.md), [implementation plan](TRUEFACE3D_MASTER_PLAN.md).

## Survey method and evidence standard

Search/read original papers, author project pages, proceedings, and official repositories. The user-requested [alphaXiv plugin](plugin://app-6a69834b831c8191a0281d6aca1722f0@openai-curated-remote) was used to retrieve architecture, training, and experimental passages; local original PDFs provide page-level text and visual diagram verification. Search titles are discovery hints, not identity proof. Nine title-based alphaXiv responses matched the wrong paper or found none and were rejected; see [the audit](evidence/REJECTED_ALPHA_MATCHES.md). Their text was not used as evidence for the intended paper.

Each original has a stable local ID, source URL, page count, SHA-256, extracted pages, and a teaching note. The corpus includes peer-reviewed papers, preprints, and one foundational course-note document (Disney2012). Preprints are not automatically validated benchmarks; an official code link is not proof that weights/training data are usable. Some downloadable arXiv PDFs remain an older version even when a later venue exists: the local file/hash is the surveyed artifact.

Reported results belong to their authors, inputs, datasets, and hardware. Our proposals and illustrative numbers are labeled. Nothing in this pack establishes that the assembled TrueFace3D system works; no model was executed here.

## 1. Geometry: strong priors plus explicit fitting

[FLAME](notes/02_FLAME.md) provides the parametric surface: identity, expression, articulation, and blend/skin deformation. It does not infer parameters from an image and it does not supply all face materials. [MICA](notes/03_MICA.md) focuses on metric neutral identity shape using recognition features and real scan supervision. [RingNet](notes/34_RingNet_NoW.md) learns identity consistency using images/landmarks without paired image–scan training.

[Pixel3DMM](notes/04_Pixel3DMM.md) predicts image-space UV/normal priors. [FlowFace](notes/05_FlowFace.md) iteratively estimates dense UV-to-image flow. **[RealDenseFace](notes/01_RealDenseFace.md)** predicts UV-space correspondence and relative depth with separate uncertainty, then efficiently fits FLAME using Gauss–Newton. It directly supports the user's geometry-first plan: the observer and solver can be inspected separately. Its existing Ava/NPHM/FaceScape training already mixes major sources.

[SMIRK](notes/06_SMIRK.md) and [EMOCA](notes/08_EMOCA.md) address expression through image/neural-synthesis or emotion supervision. They are useful expression comparators but a learned image renderer is not a physical skin shader. [NPHM](notes/31_NPHM.md), [MonoNPHM](notes/39_MonoNPHM.md), and [Pix2NPHM](notes/44_Pix2NPHM.md) offer richer implicit heads; changing representation changes meshing/attachment/export work. [H3D-Net](notes/33_H3DNet.md) needs at least three views and a private shape prior, so it is not an equal-input default.

**Decision:** reproduce RealDenseFace unchanged. Store priors, parameters, camera/crop, and solver diagnostics. Treat another backbone as a later explicit comparison, not a silent upgrade. First establish how much error is due to dense priors, initialization, solver, or FLAME representational limits.

## 2. Skin appearance: albedo is only the first material attribute

Observed face color mixes pigment, light, surface direction, highlights, shadows, camera response, and scattering. [OpenDelight](notes/12_OpenDelight.md) estimates diffuse albedo with a source-conditioned learned prior. Its image output requires visibility-aware projection to our UV map; the full published capture system uses a sequence. Published weights retain private scan training even though a later public dataset is listed.

[ReflectanceMM](notes/18_ReflectanceMM.md) learns a statistical diffuse/specular reflectance model using lower-cost multi-light photographs. [ID2Reflectance](notes/17_ID2Reflectance.md) uses identity conditioning and domain-specific codebooks. [AvatarMe++](notes/16_AvatarMePlusPlus.md), [MoSAR](notes/14_MoSAR.md), and [Relightify](notes/15_Relightify.md) reconstruct joint intrinsic maps with captured/pseudo supervision and specialized render losses. [S3-Face](notes/19_S3Face.md) explicitly considers skin scattering; [HAFMat](notes/20_HAFMat.md) offers broader human-material fusion but uses a body representation rather than our face topology.

[MARCUS](notes/13_MARCUS.md) is particularly close to the integrated goal: cascaded diffusion, geometry features, joint materials, and UV-space shading. It is a strong comparator rather than proof that modular assembly is new. Its HiFi3D++ topology, large diffusion backbone, and private measured base data complicate direct reuse on FLAME/free GPUs.

**Decision:** begin with verified OpenDelight image albedo and conservative declared physical defaults. Compare stronger intrinsic predictors on the same scene after compatibility/memory audits. Do not rename capture textures, pseudo labels, or generative completions as measured albedo.

## 3. Detail: separate real geometry from normal shading

[DECA](notes/07_DECA.md) provides an inspectable identity/expression-conditioned detail mechanism and FLAME-related pipeline. [HRN](notes/09_HRN.md) adds hierarchical geometry refinement; [HiFace](notes/10_HiFace.md) separates static/dynamic detail using synthetic/real training. [FaceScape](notes/29_FaceScape.md) supplies scan-derived displacement and predicts expression-related detail. [ReEnFP](notes/43_ReEnFP.md) already uses separate appearance/detail priors plus adaptive fusion.

Displacement moves the surface and can alter silhouette/occlusion. Normal maps modify light response but do not move vertices. Duplicating the same wrinkle in both creates exaggeration. Expression-dependent detail should follow strain/pose rather than remain a painted static wrinkle.

**Decision:** begin with restrained DECA-style detail after pinning geometry/UV/frame compatibility; evaluate actual scan displacement/normals and relighting. Add hierarchical or generative detail only when it demonstrably preserves identity.

## 4. Eyes and ears: specialize where physics or shape warrants it

[High-quality eye capture](notes/21_EyeCapture.md) resolves separate eye structures through specialized capture and refraction-aware modeling. [The parametric eye model](notes/22_ParametricEyes.md) makes lighter fitting possible using a statistical eye prior, polar iris representation, and pupil deformation. These papers explain why one painted eyeball sphere is limited, but public complete code/database access remains unverified.

The local archive initially named `23_EyeNeRF` contains **[DeNeRF](notes/23_EyeNeRF.md)**, arXiv2308.00773, not the distinct paper called EyeNeRF. Its rich multi-view/gaze training cannot be credited as observed one-photo personal optics. [AudioEar](notes/24_AudioEar.md) fits a separate parametric ear, relevant only if ear quality becomes a real bottleneck.

**Decision:** simple explicit eyes first, fit visible iris/limbus/gaze, retain generic hidden optics and physiological priors. Add ears after measuring the baseline failure. Separate objects still share lights, camera, sockets, occlusion, and attachment transforms.

## 5. Rendering: assemble one scene before introducing neural corrections

[nvdiffrast](notes/25_Nvdiffrast.md) provides differentiable graphics primitives; it is not a full camera/material/light system. [Mitsuba2](notes/26_Mitsuba2.md) explains programmable forward/inverse rendering, while current engineering should consult Mitsuba3. [Disney shading](notes/27_DisneyBRDF.md) and [subsurface transport](notes/28_SubsurfaceTransport.md) explain physical material design rather than face prediction training.

[Monocular appearance capture](notes/41_MonocularAppearance.md) demonstrates that self-visibility prevents shadows baking into albedo, using a rotation sequence. [SFDM](notes/40_SFDM.md) couples shape/reflectance under sparse views. [Becoming Lit](notes/42_BecomingLit.md) couples FLAME-bound Gaussians to hybrid neural/physical shading using multi-view OLAT data. [MeshLAM](notes/11_MeshLAM.md) targets feed-forward textured mesh avatars; evaluate its output semantics and release readiness separately.

**Decision:** one canonical asset contract, differentiable fitting plus path-traced reference rendering. Shared light/camera is basic correctness. The prospective contribution is confidence-constrained cross-component refinement with demonstrated benefit beyond that engineering baseline.

## 6. Restoration and enhancement: optional, separate interventions

[CodeFormer](notes/36_CodeFormer.md) predicts discrete codebook tokens with controllable fidelity. [GFPGAN](notes/37_GFPGAN.md) supplies a generative face prior and spatial conditioning. Its eye/mouth discriminators operate on image crops, not eye/mouth 3D assets. [Real-ESRGAN](notes/38_RealESRGAN.md) contributes repeated realistic degradation and general image enhancement.

**Decision:** finish the geometry/material/eyes baseline first. Later compare raw versus restored input, preserving original evidence. Enhance final renders only as a separately scored display branch; it cannot establish better geometry or intrinsic texture recovery.

## Reuse decisions and practical release assessment

| Tier | Methods | Meaning |
|---|---|---|
| First reproduction targets | RealDenseFace, FLAME, DECA, pretrained OpenDelight, nvdiffrast | Official source/assets identified; licenses, dependencies and actual inference still need testing |
| Strong recent comparison / optional adaptation | MARCUS, Pixel3DMM, Pix2NPHM, HiFace/HRN, reflectance methods | Compare representation, source data, weights and compute before integration |
| Conceptual/physical references | Eye capture/parametric eyes, Disney BRDF, SSS, Mitsuba2 | Mechanism/reference, not a claim of drop-in pretrained face experts |
| Richer-input references / later phase | H3D-Net, SFDM, MonoNPHM, monocular appearance capture, Becoming Lit, NeRSemble | Few-view/video/light-stage privileges must remain explicit |
| Optional robustness/display branch | CodeFormer, GFPGAN, Real-ESRGAN | Restore/display experiments after native reconstruction is established |

“Latest” and “most available” are different filters. New papers can have private data, placeholder repositories, expensive diffusion backbones, or incomplete weights. S3-Face's inspected release and several recent methods require further release auditing; avoid promising a runnable package based on a title or teaser. Stable older foundations remain necessary to understand and evaluate current techniques.

## Research conclusion for planning

The user's module sequence is feasible as a staged engineering project. **Independent experts require explicit geometry/UV/color/optics contracts and a shared physical scene.** Single-image hidden surfaces and material/light separation remain ambiguous. The strongest current research hypothesis is reliable coupling that preserves confident identity geometry while improving held-out real-view/light appearance. Dataset/training improvements can strengthen that claim if they outperform simple mixing under matched data and compute; individual component novelty is optional.

Follow the master-plan gates before committing to a new neural architecture. The survey informs a testable plan, not a finished method or publication claim.
