# TrueFace3D: scopes for the module chats

**Current fork authority:** follow the [2026-10-23 execution plan](ExecutionPlan/README.md), its component designs and updated team policy. The earlier starter messages below remain historical context; they do not impose compulsory teaching or freeze model choices against current evidence. Main Sol owns implementation, Medium assesses, and Luna High handles bounded operational support.

Prepared 2026-10-11 for the user's planned forks. This document defines handoffs; it does not create chats or start model implementation. At preparation time, the project contains a literature/design archive and model selections. Reconstruction inference, training and integrated rendering have not been run.

All chats work in `D:\Bunker\BaseCamp\ComputerVision\TrueFace3D`. Forked conversation history is context; current project files and recorded evidence determine the latest implementation state. Read current files before relying on an older conversational claim.

## Shared starting point

Each chat reads the root `AGENTS.md`, [module selections](MODULE_SELECTIONS.md), [master plan](TRUEFACE3D_MASTER_PLAN.md), and [asset/renderer contracts](SHARED_RENDERER_AND_CONTRACTS.md). Consult [datasets and training](DATASETS_AND_TRAINING.md) and [novelty analysis](NOVELTY_ANALYSIS.md) when designing experiments. Then read its component's original paper and note.

Use Super Code for feasibility, design and validation, with Teach and Code for the core mechanism. Explain the current module's first principles and data flow, define a concrete implementation plan, and work through one acceptance checkpoint at a time. Supporting adapters/configuration can be implemented directly within the user's requested scope. No fork should build all components as a side effect of starting its own module.

## Chat scopes

| Chat | Selected method | Inputs → owned outputs | First acceptance checkpoint |
|---|---|---|---|
| **Geometry** | RealDenseFace ViT-B + FLAME2023 | One image → coarse mesh/parameters, camera, crop ledger, UV/model metadata and geometric priors | Unmodified inference reproduced; mesh export and original-image projection checked; runtime/device recorded |
| **Skin appearance** | OpenDelight | Original image + geometry/camera → diffuse UV albedo, visibility/evidence masks and declared completion | Image inference checked, alignment inverted correctly, projection/color interpretation validated |
| **Surface detail** | DECA detail branch | Original image + validated target geometry/conditioning → displacement and correctly framed detail attributes | FLAME/UV/units/conditioning compatibility demonstrated; coarse identity preserved |
| **Eyes** | Parametric-eye adaptation of Bérard et al. 2016 | Geometry/camera + visible eye cues → eye objects, head-frame transforms, gaze/pupil controls and materials | Socket/lid alignment, iris projection and simple optical tests pass; inferred detail is identified |
| **Differentiable fitting** | nvdiffrast + explicit PyTorch shading/losses | Shared scene/observation → diagnostic renders, gradients, fitting results and optimization logs | Rasterization/camera/UV behavior checked; gradients and a small controlled fitting example validated |
| **Skin shading and final rendering** | Blender Cycles + Principled skin | Validated geometry/albedo/detail/eyes → assembled scene, diagnostic passes, native renders and tested exports | Shared camera/light/color conventions hold; material and eye optics respond coherently across views/lights |

The existing parent chat can serve as the **integration and research coordination** chat: reconcile interfaces, track module gates, compare complete systems and discuss novelty. It does not need to own a seventh predictor.

## Ownership boundaries

- Geometry owns the first coarse identity, expression, pose and camera estimate. Other modules consume these assets. Later refinement of them must be an explicit experiment with preserved baseline outputs.
- Skin appearance owns pigment/albedo estimation, projection and completion masks. Skin shading owns the renderer's material interpretation, lights, exposure/color management and initial optical defaults. Roughness/scattering defaults must not be described as estimated personal properties.
- Surface detail owns fine height/displacement. Regenerate geometric normals when applying displacement and distinguish any additional shading-only normal detail. Avoid applying the same wrinkle twice.
- Eyes owns the eye asset and fitting evidence. The first implementation is a simplified adaptation: the selected paper's complete executable model/database has not been verified. Final rendering assembles those assets with the existing eyelids and common environment.
- Differentiable fitting owns its backend, explicit shader approximation, loss definitions and optimization procedure. Final rendering owns the Cycles backend and native scene/export path. Both implement the same scene semantics; they are not expected to produce identical pixels or share identical light-transport equations.
- Shared contract changes must be written down and reconciled with consuming modules before integration. Do not silently change camera axes, scale, UV charts, expression bases, map interpretation or filenames that another module consumes.

The proposed source layout is already in the master plan. Keep ownership inside those module boundaries; this handoff does not create a new repository structure. If chats run simultaneously in the same working directory, inspect Git status/current files and preserve other chats' edits. Avoid simultaneous edits to shared files or global dependency environments. Use explicit module environments/artifact boundaries where upstream requirements conflict.

## Dependencies and independent preparation

Geometry is the first execution milestone. Skin projection, detail transfer and eye attachment require a validated geometry/camera artifact. Final assembly requires their independently checked outputs. Renderer foundations may start early on a clearly labeled synthetic fixture.

While waiting for geometry, the other chats can read papers/source, audit essential assets and dependencies, explain theory, design interfaces and test isolated operations. Upstream standalone demos and synthetic fixtures are useful checks, but they are not evidence that a module already works with the TrueFace3D geometry pipeline. Do not fabricate a passed dependency gate.

Full joint fitting follows validated fixed-output assembly. MARCUS remains a later gated material/completion comparator. Ears, restoration, rendered-image enlargement and short video remain later phases.

## Record a handoff after meaningful work

Each module chat should maintain its own small status document during implementation, for example `docs/status/geometry.md`, `skin.md`, `detail.md`, `eyes.md`, `differentiable_fitting.md` or `final_rendering.md`. These are future records, not existing implementation files. Include:

1. Scope and current milestone; last passed checkpoint.
2. Files changed and important entry points.
3. Exact upstream revision, model/checkpoint identity, environment and device.
4. Input/output artifact paths and schema/convention details.
5. Reproducible commands and measured checks/results, including failures.
6. Remaining blockers and what the next module may safely consume.

Keep measured, proposed and unverified claims separate. Preserve native artifacts and logs. Before training, follow the existing modular-source, remote Git, exact-commit notebook and persistent-results workflow. No chat should download an entire large dataset merely to inspect its module.

## Starter messages for the forks

Each message below identifies the scope and first work. The receiving chat should first recheck current project state and adapt its next step to any work completed since this handoff.

**Geometry**

> This fork owns TrueFace3D Geometry. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the linked selections, master plan and contracts. Start with RealDenseFace ViT-B + FLAME2023. Explain the end-to-end data flow, audit essential assets/device/dependencies, then reproduce unmodified single-image inference and validate mesh/camera exports one checkpoint at a time.

**Skin appearance**

> This fork owns TrueFace3D Skin Appearance. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the shared plans/contracts. Use OpenDelight's pretrained image-albedo prior. Explain albedo versus observed shading, audit inference/preprocessing, and develop visibility-aware UV projection against the validated geometry artifact. Declare missing-region completion and material defaults explicitly.

**Surface detail**

> This fork owns TrueFace3D Surface Detail. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the shared plans/contracts. Use DECA's detail encoder/decoder while preserving RealDenseFace coarse geometry. Explain displacement and normals, audit FLAME versions/UV/conditioning/units, and validate the detail transfer before integrated rendering.

**Eyes**

> This fork owns TrueFace3D Eyes. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the shared plans/contracts. Build a simplified explicit parametric-eye adaptation inspired by Bérard et al. 2016. Teach eye geometry/optics first, verify the release gap, and fit visible evidence while labeling procedural or generic anatomy. Validate socket, eyelid, iris and shared-light behavior.

**Differentiable fitting**

> This fork owns TrueFace3D Differentiable Fitting. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the shared plans/contracts. Use nvdiffrast with explicit camera and PyTorch shader/loss code. Start with small labeled fixtures and validate coordinates, visibility, UV sampling and gradients. Integrate controlled fitting only after its scene dependencies pass their gates.

**Skin shading and final rendering**

> This fork owns TrueFace3D Skin Shading and Final Rendering. Work in D:\Bunker\BaseCamp\ComputerVision\TrueFace3D. Read AGENTS.md and Papers/CHAT_HANDOFFS.md plus the shared plans/contracts. Use Blender Cycles with Principled skin and explicit optical eye materials. First validate a plain mesh/camera scene, then assemble validated module outputs under one light/color pipeline. Test changed views/lights, diagnostic passes and export fidelity; preserve native renders.
