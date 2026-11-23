# TrueFace3D — the execution plan for future Codex sessions

**Historical plan. Updated execution authority: [14-day plan v2](ExecutionPlan/README.md), 2026-10-23.** The new user constraint is fourteen execution days, pretrained/no-retraining work first, comparisons before final model commitment, and reviewed component gates. The old unrestricted multi-week schedule and closed-selection wording below are retained as history and do not override that update.

**Status: literature and design pack, 2026-10-10.** This plan contains engineering recommendations and research hypotheses. No reconstruction inference, training, GPU benchmark, dataset download, or integration result has been produced in this phase. Local Git was initialized; no remote repository has been configured or published.

**Module choices resolved on 2026-10-11:** read [the selected stack and implementation gates](MODULE_SELECTIONS.md) for the decision rationale and current release limitations. The original literature snapshot above is unchanged; no model execution is implied by this selection.

## Objective and user decisions

Build an **identity-preserving, visually realistic parametric 3D face asset from one RGB photograph**, then assess how well it renders under changed viewpoint and illumination. RealDenseFace is the initial geometry backbone. The output is a face asset, not a guarantee of the actual person's invisible anatomy, hair, teeth, or skin pores.

All project work belongs in:

```text
D:\Bunker\BaseCamp\ComputerVision\TrueFace3D
```

Preserve the agreed order: **geometry understanding/baseline → skin appearance → surface detail → eyes → shared assembly/refinement**. Renderer plumbing begins early because every component needs diagnostics, but the first milestone remains unmodified geometry. Ears are an optional extension; short video is later. Restoration is optional and later, with input enhancement and final-image enhancement tracked separately. This is a multi-week project; there is no inherited one-week deadline.

The user's design is the baseline. Super Code governs research/architecture and experimental gates; Teach and Code is appropriate when explaining each core implementation. Supporting infrastructure may be implemented by Codex after a concrete plan. Do not jump into building all modules from this document in one pass.

## Read these before modifying the project

1. [Research entry point](README.md) and [paper index](PAPER_INDEX.md).
2. [Literature synthesis](LITERATURE_SURVEY.md), including nearest competing integrated systems.
3. [Dataset/training protocol](DATASETS_AND_TRAINING.md) and [novelty analysis](NOVELTY_ANALYSIS.md).
4. [Shared renderer and exact proposed contracts](SHARED_RENDERER_AND_CONTRACTS.md).
5. Core notes: [RealDenseFace](notes/01_RealDenseFace.md), [FLAME](notes/02_FLAME.md), [OpenDelight](notes/12_OpenDelight.md), [DECA](notes/07_DECA.md), [parametric eyes](notes/22_ParametricEyes.md), [nvdiffrast](notes/25_Nvdiffrast.md).

Paper PDFs and archived evidence are the source of reported mechanisms and results. A note's “proposed improvement” is our hypothesis. If upstream code, weights, or data change, preserve a dated audit rather than silently rewriting the evidence.

## Proposed baseline architecture

~~~mermaid
flowchart TD
 I[One RGB image and original metadata] --> O[Crop / masks / coordinate ledger]
 O --> G[RealDenseFace priors + solver]
 G --> F[FLAME geometry / parameters / camera]
 O --> A[Pretrained image albedo prior]
 A --> U[Visibility-aware image-to-UV projection]
 F --> U
 O --> D[Detail prediction]
 F --> D
 O --> E[Eye fitting]
 F --> E
 U --> M[Skin material maps + declared defaults]
 F --> S[Canonical scene assembly]
 M --> S
 D --> S
 E --> S
 S --> R[Shared renderer]
 R --> V[Input view / novel views / relighting / diagnostics]
 V --> B[Bounded consistency refinement: later experiment]
 B --> S
~~~

Select OpenDelight as the first **image-albedo module**, with visibility-aware projection to FLAME UV. Select DECA's detail mechanism after compatibility checks. Begin eye assembly with parametric geometry and conservative materials; the 2016 paper is a design reference with no verified complete plug-and-play release. Use nvdiffrast for fitting and Blender Cycles/Principled for final scene rendering. These are implementation targets whose execution and quality still require validation.

MARCUS is the selected recent integrated comparator and preferred later full-material upgrade. It uses a different topology and heavier diffusion components. It must pass data/license, topology, memory, and identity gates before adoption; it does not silently replace RealDenseFace. ReEnFP and SFDM are important prior art for independent priors and joint coupling.

## Phase 0 — source/asset feasibility

**Goal:** ensure our minimum stack can actually execute and be legally/reproducibly evaluated. **Input:** source pack and target hardware. **Output:** pinned dependencies/assets, access checklist, baseline run configuration. **Owner:** Codex handles audit/plumbing; user learns module boundaries.

Check official code and checkpoint availability separately. Download only necessary permitted model assets, not entire datasets by default. FLAME2023 requires separate access/licensing. Verify UV layout, vertex/joint ordering, model units, and camera conventions. RealDenseFace paper uses MICA initialization; the released single-image path must be traced to verify what is actually executed. Its frozen-backbone configuration and paper training schedule differ: log the selected source commit.

The 2026-10-11 availability audit identifies **FLAME2023 Open (CC-BY-4.0)** as an option; its expression basis needs conversion/mesh validation against the baseline model. See [the dated release audit](evidence/RELEASE_AUDIT_2026-10-11.md). Keep the exact model required by upstream inference as the first reproduction target.

Target free Colab/Kaggle NVIDIA GPU execution for heavy components if available; local MX330 2 GB/Iris Xe constraints are prior context and must be confirmed. Start with batch-one inference and measure peak memory/runtime. Do not promise that all 2026 diffusion systems fit a T4, that dual T4 acts as one 32 GB GPU, or that CUDA extensions work on TPU. CPU is useful for inspection/exports and unsupported backends must fail clearly.

**Gate:** essential checkpoint/model assets accessible; one valid input and prescribed inference path runnable on an identified device; license restrictions documented; camera/UV metadata available. If a critical asset is unavailable, preserve the baseline and identify a scoped fallback. Do not claim reproduction from an unrelated implementation.

## Phase 1 — RealDenseFace end to end

**Goal:** understand and reproduce geometry before improving it. **Input:** original image, pinned model assets. **Output:** raw priors C/D/Lc/Ld, fitted FLAME parameters, mesh/camera/crop, logs, flat-material renders. **Dependencies:** Phase 0. **Owner:** user and Codex inspect core network/solver together; Codex implements adapters and export utilities.

Trace preprocessing → DINOv3/UV queries → separate correspondence/depth heads → vertex sampling → uncertainty weighting → Gauss–Newton → FLAME. Explain analytic Jacobians, damping/Cholesky, identity/dynamic alternation, and solver failure modes. The paper's GPU timing is not our end-to-end performance.

Run frontal, oblique, expression, occlusion, and low-quality examples. Keep failures. Check identity shape stability, projection into uncropped coordinates, plausible unit scale, finite solver state, and mesh export/reload. Establish official NoW/NeRSemble evaluation where available; held-out benchmark rules remain binding.

**Gate:** a reproducible mesh and camera, correct coordinate roundtrip, documented baseline error/runtime/memory on valid evaluation data. Understanding checkpoint: user can distinguish dense priors, fitted parameters, and mesh vertices. Only then test geometry improvements such as uncertainty calibration, initialization, or bounded detail.

## Phase 2 — skin appearance

**Goal:** predict usable diffuse albedo and integrate it without baking light twice. **Input:** original crop, mesh/camera, image-albedo checkpoint. **Output:** UV albedo, observed/completed masks, confidence, declared material defaults. **Dependencies:** Phase 1 and minimal shared renderer. **Owner:** core mechanism explained with user; Codex may build projection/masking tooling.

Start with image-albedo prediction and invert all alignment transforms. Project visible pixels into UV using geometry and z-buffer visibility. Keep empirical evidence separately from completed unseen texels. Use linear color correctly. Defaults for roughness/specular/SSS remain defaults until supervised or validated fitting replaces them.

Compare input texture, delit albedo, and candidate stronger materials on the **same geometry/camera/light**. Do not improve geometry secretly while crediting an albedo experiment. Check facial marks, skin-tone/exposure ambiguity, saturated highlights, and shadows. Material GT requires measured/controlled sources, not ordinary FaceScape texture.

**Gate:** map alignment and masks correct; physical relighting passes; same input/novel-view evaluation defined; resource costs recorded. Only then consider MARCUS transfer or a small student model.

## Phase 3 — surface detail

**Goal:** improve resolved surface detail without changing identity arbitrarily. **Input:** image, pinned FLAME mesh/UV, detail checkpoint. **Output:** displacement with units, residual normal map with declared frame, expression/identity detail separation. **Dependencies:** Phases 1–2. **Owner:** user understands displacement/normal difference; Codex handles validated adapters.

DECA's model version/UV/frame differs from arbitrary FLAME assets; prove compatibility. Transfer detail relative to the target base if needed, regenerate normals, and preserve seams. Start with restrained amplitude. Measure depth/normal agreement where scanned labels exist; inspect silhouette, moving-light behavior, and expression dependence. Do not count shader-only pores as improved geometric reconstruction.

**Gate:** topology/frame tests pass, no unresolved double-counted detail, held-out detail/geometry measures and failure cases retained.

## Phase 4 — eyes

**Goal:** add coherent eyeballs/cornea/iris and fit visible evidence. **Input:** mesh eye joints/sockets, visible iris/limbus landmarks and masks. **Output:** eye objects, head-frame transforms, gaze/pupil parameters, iris/sclera materials and declared generic optics. **Dependencies:** stable geometry and shared camera/light. **Owner:** core optics/geometry taught; procedural asset scaffolding implemented after contract review.

Read capture/parametric papers to understand refraction, iris depth, sclera and pupil deformation. Begin simpler than the full capture pipeline; exact invisible eye geometry cannot be identified from one portrait. Fit visible silhouette/iris/gaze and render highlights with the shared environment. Avoid supervising on privacy-mosaiced eyes.

**Gate:** front/oblique renders have correct lid occlusion, no socket intersections, highlights respond to lights, fitted gaze agrees with visible evidence. A pretty close-up without valid eye labels is a visual demo, not ground-truth anatomical recovery.

## Phase 5 — assembled asset and required integration research

**Goal:** make modules agree and test whether the coupling has research value. **Input:** independently verified geometry/material/detail/eye assets. **Output:** canonical scene, export, joint-refinement experiments, ablation report. **Dependencies:** previous gates. **Owner:** user participates in coupling design; Codex builds experiment infrastructure.

Use nvdiffrast/PyTorch for initial differentiable fitting and Cycles for final reference light transport, under one scene contract. Test export fidelity: OBJ lacks material optics, GLB support varies, displacement must be applied or explicitly retained, and a .blend reference may carry features other exports omit. Preserve native render and diagnostic passes.

Experiment with a small confidence/visibility-conditioned adapter and **bounded** alternating updates. Freeze reliable identity geometry first. Compare naive assembly, joint fitting, and proposed coupling at equal inputs/data/compute. Inspect conflicting explanations of dark areas and highlights; use measured unseen views/lights where possible. Follow [the novelty analysis](NOVELTY_ANALYSIS.md) before making paper claims.

**Gate:** independently measured benefit over credible integrated baselines, with geometry/identity preservation and runtime costs; or a candid negative result that narrows the hypothesis. Integration is required engineering work; scientific novelty remains unproven until that evidence exists.

## Phase 6 — optional enhancement and later video

Input restoration: compare raw, restored-only, and reliability-aware use of both. Final render enhancement: keep native renders and quantify identity/projection/new-view consistency. These are two different interventions. Neither should hide a mesh/material defect.

Later short video: share identity/materials through frames; vary pose/expression/gaze/lighting as explicitly permitted. Use temporally separated test segments and distinguish causal online tracking from full-sequence optimization. MonoNPHM and monocular appearance capture are references with richer temporal inputs; do not compare them as one-photo methods.

## Proposed implementation layout — create during implementation, not now

```text
src/trueface3d/
  contracts.py         observation, geometry, material, detail, eyes, scene
  geometry/            upstream RealDense adapter, parameter/mesh export
  appearance/          albedo, UV projection, optional material predictor
  detail/              displacement, normal frames, compatibility adapters
  eyes/                procedural geometry, fitting, attachment
  rendering/           scene assembly, differentiable backend, Cycles adapter
  data/                manifests, licensed local adapters, identity splits
  evaluation/          geometry, image, relighting, eye and stability metrics
configs/               versioned baseline and experiment settings
scripts/               thin inference/train/evaluate entry points
notebooks/             standalone clean-sync experiment launcher
tests/                 meaningful coordinate/contract/integration checks
results/               run metadata only; heavy outputs in persistent store
Papers/                this research pack
```

No module above exists yet. Prefer isolated upstream checkouts with recorded commits instead of copying opaque code throughout our package. Respect upstream licenses. Keep learning-critical equations close to readable implementation rather than trapped inside notebooks.

## Experiment reproducibility and HPC workflow

Before training, configure a local **and remote** Git source repository. Keep source reusable; the standalone Local/Colab/Kaggle notebook is a launcher. Its clean-reset cell deletes only the explicitly verified temporary project clone, synchronizes the requested commit, detects the environment/backend, and checks compatible model execution. It must not delete persistent datasets/results.

Support CPU and compatible NVIDIA GPUs through one experiment interface, with measured configurations for T4/P100/A100 and optional multi-GPU support. TPU requires a separate compatible backend; CUDA-only rasterizer/solver paths are not portable simply because a notebook detects a TPU.

Every run records run ID, Git commit (including dirty state), upstream commits/checkpoint hashes, data-manifest hash, identity/camera/light splits, seed, package versions, accelerator/backend, config, timings, peak memory, and preprocessing. Save checkpoints/metrics/logs/plots/configs to configured persistent storage before runtime resets. Colab mount/Kaggle output/artifact store behavior must be verified for the actual environment. Empty/failed results remain explicit; no fabricated plots.

## Future-agent restart checklist

- Read this plan and current implemented files; identify the active phase and last passed gate.
- Check Git status and preserve user changes. Work only in TrueFace3D; the ChatGPT mirror sources are read-only.
- Recheck drift-prone weights, data access, and requirements for the chosen phase.
- Teach the current module's important mechanism and proposed interface before burying it in adapters.
- Complete one phase's reviewable work, run meaningful checks, and record measured/failed/unverified outcomes.
- Update project documentation when implementation changes; do not silently promote a hypothesis into a proven novelty claim.
- Do not download terabytes, train a large diffusion model, alter the geometry baseline, or publish source/assets as a side effect of reading this plan.

The immediate next milestone is **Phase 0 followed by the unmodified RealDenseFace single-image baseline**. The survey is complete as a curated pack; model reproduction and research claims are separate future work.
