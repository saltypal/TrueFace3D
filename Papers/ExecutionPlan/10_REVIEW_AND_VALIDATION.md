# Review and validation

## Review record and conclusion

**Date:** 2026-10-23. **Review:** requested Sol extra-high, bounded design/scientific critique after the complete plan draft. **Scope:** the 18 Markdown plan documents, `plan_state.json`, the recorded release audit and supporting shared-renderer contract. The source tree is not an implemented reconstruction system. This pass made no installations, model/data downloads, inference runs, GPU measurements or external submissions. It did not independently reproduce the paper tables or freshly verify their release URLs.

**Conclusion:** the pretrained-first route is defensible as a conditional engineering project. The strongest achievable fourteen-day result is a reproducible coarse face asset, a correct common scene and a small matched comparison, with optional detail/eyes/refinement governed by explicit fallbacks. Completing all proposed comparisons and two useful improvements is a stretch dependent on assets, hardware, labels and daily hours. No model is established as a universal winner, and no proposed improvement has empirical support yet.

The main objections concern delivery contracts and experiment readiness, rather than a need for a larger model or architecture. Fix the six items below before relying on the affected gates. Day 1 asset/device/data preflight can proceed without resolving distant training questions. Severity denotes priority within this plan: **P1** can defeat a promised deliverable or the critical path; **P2** can invalidate a comparison or a later acceptance decision. A confirmed specification gap is not a confirmed runtime defect.

Locations below refer to the reviewed draft before main-thread corrections. The lead owns corrections, verification and dispositions; this review edits only this report. An assessment marked accepted is not evidence that a model executed successfully.

## Findings

### R1 — P1: the first actual Cycles integration is too late in the critical path

- **Class/confidence:** confirmed schedule risk; high confidence in the dependency, unmeasured implementation cost.
- **Location:** `06_14_DAY_ROADMAP.md`, Days 2–3 and 9–11; `plan_state.json`, `G6_fixture`; `04_ARCHITECTURE.md`, startup gates.
- **Evidence:** the roadmap first explicitly schedules the minimal Cycles fixed scene on Day 10, after appearance, detail and eye work. `G6_fixture` depends only on G1, so there is no dependency reason to wait that long. A Day 2 plain mesh/camera render does not specify the Cycles conversion, color/UV semantics or export roundtrip that the final asset requires. The documents themselves identify camera/units/UV/material transfer as the hardest risk.
- **Impact:** a camera, import, topology or shader incompatibility can remain hidden until the integration reserve is nearly consumed. Optional modules may then have been developed against an assembly path that cannot meet the final rendering contract.
- **Smallest correction:** immediately after the first G1 output, build the minimal real-geometry Cycles scene using a constant material and, when available, a raw/checker UV map. Give it an explicit early deadline, preferably Day 3 and no later than Day 5. Keep later eye/SSS/detail polish separate. Run the next challenger only within the remaining budget.
- **Acceptance:** save a native C1 artifact → canonical scene → Cycles render → reload bundle roundtrip, with original-camera projection, known units, oriented normals and asymmetric UV checks. A synthetic fixture alone does not establish real adapter integration. Reuse this exact scene path for later components.

### R2 — P1: reloadable meshes do not yet establish the promised parametric asset

- **Class/confidence:** confirmed acceptance-contract gap; high confidence.
- **Location:** `04_ARCHITECTURE.md`, final-asset paragraph; `05_LOW_LEVEL_DESIGN.md`, GeometryAsset/DetailAsset/EyeAsset and C1/C3/C4/C6 functions; `components/C6_RENDERING.md`, G6.
- **Evidence:** the HLD promises a parametric base and expression controls. The LLD stores named parameters/basis IDs and tests mesh/export reload, but does not require an executable parameter re-evaluation path or a control-change roundtrip. A baked mesh plus unused coefficient arrays can currently satisfy the listed scene/export checks. Detail is expression-conditioned, and eyes are attached to a head transform, so simply editing a coefficient need not update the assembled asset coherently.
- **Impact:** the delivered result may be a static reconstruction snapshot while being described as a controllable parametric asset. Changing jaw/expression/gaze can detach eyes, preserve wrinkles from the source expression, invalidate displaced normals or silently lose the source model needed for regeneration.
- **Smallest correction:** define whether the required control path is Python regeneration from pinned parametric assets, native Blender controls, or both. Python regeneration is sufficient if declared; a full animation rig need not be added. Specify model resolution, parameter names/shapes/basis, supported control ranges, dependent geometry/detail/eye updates and the supported export subset. Respect model redistribution terms by recording asset resolution/hash requirements rather than embedding restricted payloads.
- **Acceptance:** reload from a fresh process, reproduce the source parameters, then apply at least one nonzero expression/jaw change and one gaze change using the declared controls. Verify head/eyes/lids, displacement ownership, recomputed normals and saved/reloaded output. If DECA detail cannot be reconditioned in the target basis, label it source-pose-only and disable/revert it on unsupported control changes; do not imply expression-following detail. Static OBJ/GLB snapshots remain valid exports when labeled accordingly.

### R3 — P2: light and exposure need an explicit scale gauge before parameter recovery

- **Class/confidence:** confirmed under-specification with a mathematical non-identifiability; high confidence. No numerical failure has been observed.
- **Location:** `components/C5_FITTING.md`, fitting flow and light/exposure paragraph; `05_LOW_LEVEL_DESIGN.md`, `build_variables`/`compute_losses`; `03_METRICS_AND_BENCHMARKS.md`, synthetic parameter recovery.
- **Evidence:** C5 proposes fitting light and exposure while fixing or constraining albedo. That removes one ambiguity but still leaves a light-energy/exposure ambiguity. In a linear scene, `I = e R(a,L)` and `R(a,kL) = k R(a,L)`, so `(e/k, kL)` produces the same image. White balance introduces further channel ambiguity if it is also free. Bounds can limit the family without identifying a physical solution.
- **Impact:** an optimizer may return arbitrary light/exposure parameters; synthetic recovery can be reported as a failure or success for variables that the image cannot identify. Conditioning and cross-renderer transfer can also depend on the chosen arbitrary scale.
- **Smallest correction:** fix exposure and white balance for the first fitting experiment, or fix a declared illumination-energy normalization/reference while fitting exposure. Keep albedo/camera fixed initially. State which quantities are recovered and which remain graphics conventions. Introduce additional variables only after a separate identifiability check.
- **Acceptance:** include a synthetic fixture demonstrating the chosen anchor and bounds. Test parameter recovery only for identifiable variables; otherwise compare rendered equivalence and explicitly report the gauge ambiguity. Do not call estimated lamp energy or exposure a physically measured quantity.

### R4 — P2: approximate-renderer gains need a final-renderer adoption gate

- **Class/confidence:** confirmed gate gap; high confidence in the risk, transfer failure untested.
- **Location:** `04_ARCHITECTURE.md`, candidate acceptance before Cycles; `components/C5_FITTING.md`, G5b; `components/C6_RENDERING.md`, ablations and G6.
- **Evidence:** nvdiffrast plus the project shader and Cycles intentionally approximate transport differently. G5b selects a candidate before the full C6 milestone. C6 requests a later no-refinement/refinement comparison, but the gate does not explicitly revoke an earlier accepted refinement when its benefit disappears in the final scene. SSS, shadows and corneal transport are outside the initial differentiable approximation.
- **Impact:** fitting can compensate for shader error, producing a lower approximate loss but worse final Cycles appearance or material interpretation. Identical camera/UV semantics alone do not ensure a learned/fitted parameter change transfers.
- **Smallest correction:** make C5 acceptance provisional until the same baseline/candidate ScenePackets are rendered through the pinned Cycles path without further target-driven adjustment. Start optimization in a mutually supported diffuse/lighting subspace and hold unsupported skin/eye terms fixed or masked. Define a rejection/rollback edge from final-scene preservation checks to the original scene.
- **Acceptance:** on development cases, compare the baseline and candidate with the same Cycles settings, camera/light policy, linear/display conversion and masks. Require the predefined benefit/preservation checks in the delivered backend as well as numerical C5 validity. With no valid GT, limit adoption to declared engineering/demo behavior. No guarantee of physically accurate optics follows from a rasterizer fixture.

### R5 — P2: H1/H2 still need concrete matched experiment cards before coding

- **Class/confidence:** confirmed next-increment design gap; high confidence. General protocol is present and should be retained.
- **Location:** `09_LIMITATIONS_AND_NOVELTY.md`, H1/H2 and experiment template; `components/C2_APPEARANCE.md`, experiments; `components/C5_FITTING.md`, experiments; `03_METRICS_AND_BENCHMARKS.md`, adoption rules.
- **Evidence:** the plan lists appropriate families and requires future cards, but does not yet choose the actual optimized variables, loss units/reductions, bounds/initialization, termination, fixed masks, common compute budget or primary per-experiment endpoint. “Robust-only” and “staged bounded” can otherwise change both the loss and the allowed degrees of freedom. A naive projection can also be an intentionally weak baseline if its missing visibility handling is standard correctness.
- **Impact:** a gain can be attributed to the wrong mechanism, additional iterations, masked-out difficult regions or simply freezing variables. The current list is enough to organize research, but not an executable or falsifiable H1/H2 protocol.
- **Smallest correction:** before each improvement is implemented, freeze one compact card. For H1, retain a correct z-buffer/external-occlusion projection as the credible simple baseline, then isolate reliability weighting or the residual change; use a common evaluable region and report coverage separately. For H2, list fixed/optimized variables, normalized losses/weights, bounds, initial scene, iteration/render-evaluation budgets and failure policy. Isolate robust loss, staging and bounds when attributing a mechanism; if time permits only a combined policy comparison, describe it that way. Do not force an expansive ablation grid.
- **Acceptance:** the card names an actual available development manifest or explicitly synthetic fixture set, one main quality endpoint, a preservation constraint, a time/memory ceiling and the rule for accept/reject/inconclusive. Set tolerances from development repeatability and task requirements before held-out runs. Keeping identity fixed is a useful constraint, but zero parameter drift by construction is not independent evidence of recovered identity accuracy.

### R6 — P2: final review after held-out evaluation needs a strict post-exposure rule

- **Class/confidence:** confirmed sequencing risk; high confidence. This review found no executed leakage because no experiments have run.
- **Location:** `06_14_DAY_ROADMAP.md`, Days 12–13; `03_METRICS_AND_BENCHMARKS.md`, held-out/freeze policy; `08_TEAM_AND_CHANGE_WORKFLOW.md`, review cadence.
- **Evidence:** Day 12 evaluates held-out cases; Day 13 performs integration review and regression repair. The roadmap correctly requires affected results to be invalidated and rerun, but a rerun alone does not restore an untouched evaluation boundary after method/threshold/selection decisions use test outcomes.
- **Impact:** late changes selected because of held-out failures can turn the same identities into development data, even if every result has a valid run ID. Component-wise adaptation can accumulate this exposure across the six modules.
- **Smallest correction:** perform the final correctness/integration review and fresh-kernel rehearsal before freezing the held-out configuration. Use one shared exposure/freeze register across components. After exposure, distinguish a mechanically justified correctness fix from a methodological change; document the exact change and all invalidated results. A new method/threshold/candidate choice requires an untouched evaluation set or an explicit downgrade to a development-exposed exploratory result.
- **Acceptance:** the evaluation bundle records the freeze revision/config, evaluated identities, which targets were visible to selection, and post-freeze changes. No per-identity test-GT best-candidate selection. Rerun affected baseline/challenger comparisons together; do not retain favorable pre-fix rows.

## Boundaries retained after critique

- The C5/C6 dependency is **not inherently cyclic**: `G6_fixture` can precede C5 while complete G6 follows it. R1 concerns its scheduled timing and actual evidence, not a need to merge the components or add a renderer service.
- The plan correctly separates native-method comparisons from fixed-base transfers; neutral from posed shape; color-aligned author results from raw project metrics; and synthetic correctness from real personal accuracy. Preserve those separations in the implementation.
- The release audit records 16 checked sources and explicitly says no artifacts were acquired/executed. This is a source-access record, not an installation certificate or coverage of every release behind all 44 papers. The remaining entries must retain their dated/unverified states. This review does not add a fresh remote availability claim.
- The fallback route is legitimate: captured color must remain labeled captured texture; generic eyes/materials are declared priors; no-detail is preferable to unvalidated transfer. These fallbacks cannot simultaneously support claims of measured personal albedo, pores, anatomy or full optical recovery.
- Exact cross-topology correspondence, displacement rebaking and normal frames remain genuine implementation tests. The current contract already warns against PNG resizing and duplicate detail; no new registration framework is justified until a compatible branch fails.
- Notebook cell logging, executed-notebook/rich-output preservation, partial-run evidence, actual environment reporting and one `KAGGLE_RUNS.md` are required preflight work. Their documentation is not proof the capture/retrieval mechanism works. Verify a representative roundtrip before an expensive run and retain the no-interruption rule for healthy runs.

## Main-thread disposition register

The lead fills this after making corrections. Do not mark an item resolved merely because this report exists.

| ID | Initial disposition | Correction/check evidence | Residual limit |
|---|---|---|---|
| R1 | Accepted, corrected in plan | Day3 real Cycles/parameter roundtrip, no later than Day5; `G6_fixture` target recorded in state | Actual Cycles adapter execution untested |
| R2 | Accepted, corrected in plan | LLD adds `regenerate_from_parameters`; experiment-card control/reset/reload tests and explicit detail-update capability | Parametric regeneration/attachments untested |
| R3 | Accepted, corrected in plan | H2-v1 freezes exposure/WB/albedo/camera and specifies bounded lighting variables and scale convention | Synthetic identifiability/recovery and optimizer behavior untested |
| R4 | Accepted, corrected in plan | C5, C6, HLD and gate metadata require final Cycles checks; failed candidate rolls back to preserved baseline | Cross-renderer benefit unmeasured |
| R5 | Accepted, next increment specified | H1/H2 cards define serious baselines, variables, loss reductions, initial bounds/budgets and primary endpoints; manifest/tolerance freeze required before benchmark coding | Actual data/assets and development-derived tolerances remain pre-execution gates; no novel mechanism or gain established |
| R6 | Accepted, corrected in plan | Day11 final assessment/fresh-kernel rehearsal precedes Day12 holdout; single exposure rule governs post-freeze fixes and claims | No held-out experiment has executed |

The main thread also removed a conflicting secondary schedule from the all-paper matrix, aligned its 16 freshly checked access entries with the release audit, and corrected the video-only appearance-capture entry. The original 18-document critique is retained above; the subsequent experiment-card document and corrections were checked by the main thread, not represented as a second independent extra-high review.

## Reproduction of static checks

From `D:\Bunker\BaseCamp\ComputerVision\TrueFace3D`:

```powershell
python .\Papers\tools\validate_execution_plan.py
python .\Papers\tools\build_catalog.py
```

The first checks 44-paper action coverage, six component designs, gate references/DAG, JSON, local links and code-fence pairing. The second checks the original archive's PDF titles/hashes/pages and local Markdown links. Results are saved in `evidence/plan_validation.json` and the archive validation report. They do not parse or execute Mermaid diagrams, run proposed function APIs, test notebook persistence, fetch all external links or execute any model. All numeric performance remains author-reported or unmeasured as labeled.

## What this review does and does not validate

This is a document-level assessment of mechanisms, dependencies, acceptance criteria and scientific inference. Structural validation can establish link/schema/count consistency only. No check in this pass establishes checkpoint integrity, hardware compatibility, true parameter recovery, geometric/material accuracy, render fidelity, portable notebook execution, complete output persistence, measured runtime/FLOPs or novelty. No new broad literature search or independent audit of all 44 original experiments was performed for this bounded critique. Residual uncertainty is resolved by the planned asset preflight and discriminating experiments, not by another model agreeing with the plan.
