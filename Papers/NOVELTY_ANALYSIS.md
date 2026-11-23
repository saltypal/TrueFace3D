# What could be scientifically new in TrueFace3D?

**2026-10-10: proposed claims, not established results.** Integration is a required project phase. Combining published models or training on additional existing data is **not automatic scientific novelty**. The paper must demonstrate a specific mechanism, useful evidence, and an improvement over credible alternatives under the same input assumptions.

## Nearest prior work that constrains our claim

| Existing work | Overlap with our idea | What we cannot claim as new by itself |
|---|---|---|
| [DECA](notes/07_DECA.md), [HiFace](notes/10_HiFace.md), [HRN](notes/09_HRN.md) | Base geometry plus separate detail/appearance pathways | Coarse face + displacement, identity/expression detail separation |
| [ReEnFP](notes/43_ReEnFP.md) | Separate texture/detail priors, shared encoder, adaptive image/prior fusion | Using multiple face generators and attention-based fusion |
| [SFDM](notes/40_SFDM.md) | Geometry/reflectance interaction, template plus offsets and SSS approximation | Geometry and material helping each other or adding a prior |
| [MARCUS](notes/13_MARCUS.md) | Geometry-guided intrinsic materials and integrated physical shading | Joint material prediction guided by 3D face geometry |
| [MoSAR](notes/14_MoSAR.md), [AvatarMe++](notes/16_AvatarMePlusPlus.md), [Relightify](notes/15_Relightify.md) | Joint normals/reflectance and realistic relighting | Producing several PBR maps from one image |
| [OpenDelight](notes/12_OpenDelight.md) | Mixed-source albedo learning with source-conditioned prior | Real/synthetic mixing or dataset tokens alone |
| [MonoNPHM](notes/39_MonoNPHM.md) | Geometry-conditioned appearance and parametric reconstruction | Generic geometry/appearance feature exchange |
| [Monocular appearance capture](notes/41_MonocularAppearance.md) | Explicit visibility prevents baked shadows | Including self-occlusion in inverse rendering |
| [Becoming Lit](notes/42_BecomingLit.md) | FLAME-bound detail and hybrid physical/neural shading | Attaching appearance to FLAME or neural skin residuals |
| [RealDenseFace](notes/01_RealDenseFace.md) | Dense priors with uncertainty and fast fitting | Using log-variance to weight geometric residuals |
| [Parametric eyes](notes/22_ParametricEyes.md), [AudioEar](notes/24_AudioEar.md) | Specialized anatomical components | Separate eyes/ears or parametric optical/ear fitting |

These papers form a strong initial prior-art set, not an exhaustive proof that a proposed mechanism has never appeared. Before writing a novelty claim, conduct a focused search using the final mechanism's terms, including general inverse-rendering work beyond faces.

## Axis 1 — datasets and training methodology

**Low novelty:** train the same network longer, concatenate FaceScape/NPHM/Ava, add FFHQ, increase resolution, or report better performance after more compute. These can be useful engineering/data-scale studies, but they do not imply a new method.

**Potentially meaningful contribution:** define a missing paired-data protocol; make supervision provenance explicit; show a training method addresses a known discrepancy between real scans, captured colors, controlled-light targets, and pseudo maps. A new dataset needs consent/access, calibration, quality assessment, repeatable splits, and a defensible contribution beyond re-exporting existing files.

### Candidate A: provenance- and quality-aware supervision

**Problem:** a texture photograph is sometimes labeled albedo, and a teacher-generated roughness map is sometimes treated like measured truth. Mixed training can systematically teach wrong physical relationships.

**Proposed mechanism:** keep source/provenance labels and train a small quality-conditioned loss/gate. Supervise geometry strongly when scans are valid, albedo cautiously when illumination separation is approximate, and uncertain hidden regions through explicit priors. Calibrate quality on independently measured held-out examples rather than arbitrary hand scores.

**Prior-art boundary:** OpenDelight's DLM already conditions on data source. A new contribution must do more than change token names: demonstrate per-target quality/provenance handling, geometry-dependent reliability, or a clear effect on shared scene consistency.

**Required experiment:** scan-only; pseudo-only; naive mixing; existing source conditioning; proposed weighting. Match identity/sample/update count and use source/identity/light-disjoint tests. Report geometry, albedo, relighting, reliability and compute. A result driven solely by extra examples does not support the mechanism.

### Candidate B: one-photo prediction with withheld real-view/light supervision

Train using multiple measured views/lights but feed only one view to the predictor. Test on unseen identities and withheld measured cameras/lighting. This supervision pattern is established; novelty could lie in a new coupling objective or evaluation protocol that measures contradictory component explanations. Simply saying “trained using multiple views, tested monocular” is insufficient.

**Risk:** pretrained RealDense exposure can invalidate unseen-identity language. Audit exact upstream splits; if unavailable, report the uncertainty and use verified independent evaluation where possible.

## Axis 2 — component novelty, optional

The user does not require every component to be a new method. A reproduced baseline with thoughtful adaptation can be enough if the main contribution is elsewhere. Component candidates below are optional and should be picked only after the baseline exposes the failure they address.

| Component | Specific proposed change | Required ablation and risk |
|---|---|---|
| Geometry | Calibrate C/D uncertainty under pose/quality shifts and constrain update budgets | Original weights vs calibrated weights; scan risk–coverage. Generic calibration alone may be incremental |
| Skin | Preserve observed marks while completing only unreliable UV regions; supervise physical consistency | Whole-map generation vs masked completion vs confidence-guided completion, independent albedo/relighting evidence |
| Detail | Separate identity detail, expression strain, and unresolved shading detail using confidence | DECA-style baseline vs proposed residual; actual displaced geometry/normal agreement, no duplicated wrinkles |
| Eyes | Geometry/visibility-constrained monocular limbus/iris/gaze fitting with common light | Generic eyes vs independently fitted vs coupled fit; ground-truth or appropriate visible-cue metrics. Personal optics remain ambiguous |
| Ears | Fit a separate ear shape only when visible evidence supports it | Mean FLAME ears vs independent ear model vs confidence attachment; seam/topology/calibration checks |
| Renderer | Adapt a low-cost visibility/eye approximation to a matched path-traced teacher | Simple shading vs reference vs adapter; radiance/gradient error under unseen scenes; existing approximation literature is extensive |
| Enhancement | Preserve geometry/material identity while improving degraded input or displayed render | Native/raw vs generic enhancer vs constrained enhancer; clear input/output distinction |

Changing a backbone, adding an MLP, concatenating features, or tuning loss weights is not automatically novel. A component claim needs a measurable failure, an explanatory mechanism, and controlled evidence.

## Axis 3 — integration, required engineering and strongest current research target

### Candidate C: confidence-constrained cross-component scene consistency

**Failure to address:** independent experts explain a shiny/dark region differently. One changes a normal, another changes albedo, another changes lighting. The assembled face matches the input view but fails under a new camera/light or alters identity geometry.

**Proposed mechanism:** use a shared FLAME scene with explicit camera, visibility and material semantics. Compute masks/confidence from geometric uncertainty, visibility, photometric residuals, and label provenance. A lightweight controller selects trusted evidence and bounds allowable updates to each component. High-confidence identity geometry is anchored; uncertain material/detail regions may change more. Alternate updates instead of simultaneously releasing every parameter.

Possible inputs: per-UV geometric confidence, view angle/visibility, predicted materials/detail, input residuals, component agreement. Outputs: regional gates, update budgets, and small residual map corrections. Exact architecture should be designed after observing baseline failures; do not preselect a large network because “fusion” sounds novel.

**Why it could matter:** physically coupled explanations may make a realistic render while preserving recognizable geometry and enabling portable assets. The prospective contribution is the reliability constraint and demonstrated generalization, not the generic shared renderer.

**Closest competitors:** ReEnFP adaptive fusion, SFDM attribute synergy, MARCUS integrated geometry/materials, source-aware OpenDelight, RealDense uncertainty fitting. Every claimed difference must be stated against these mechanisms and reproduced under a comparable one-photo protocol where applicable. Multi-view methods can be references/upper bounds, not equal-input baselines by default.

### Required ablation ladder

| Experiment | Change from preceding stage | What it establishes |
|---|---|---|
| B0 | RealDense geometry + simple observed texture/default material | Starting geometry/appearance baseline |
| B1 | Independent albedo/detail/eye experts with correct adapters | Benefit of adding modules, not learned integration |
| B2 | Shared camera/light and physical renderer, no learned controller | Benefit of correct engineering consistency |
| B3 | Ordinary joint fitting with the same scene/data | Benefit/cost of optimization |
| B4 | Visibility masks and geometry-prior bounds, hand-designed | Strong simple control baseline |
| P1 | Learned reliability controller, same experts/data/budget | Actual effect of the proposed controller |
| P2 | Provenance-aware training added to P1 | Separates training novelty from coupling novelty |

Remove one reliability input or constraint at a time: geometry confidence, visibility, source/provenance, identity anchor, alternating schedule, seam constraints. If B4 matches P1, keep the simple mechanism and revise the claimed contribution. If improvement only follows a different material expert, credit that expert rather than the controller.

## Success, evidence, and honest claim language

Predefine primary outcomes after reproducing baselines. A sensible target is better **held-out real-view/real-light appearance with geometry/identity preserved**, accompanied by reliability calibration and runtime cost. Geometry noninferiority needs a justified margin set before examining the final test data; do not invent one now. Use per-identity confidence intervals and document failed cases.

Use a fixed scene/input protocol and comparable optimizer budget. Report both quality and cost of stronger 2026 methods. A method whose weights/private data cannot be reproduced remains a source-described comparison; do not populate a numerical table with unmeasured values.

Useful wording before experiments: “We propose a confidence-constrained assembly/refinement method for one-photo parametric face assets.” After sufficient results: state the concrete measured gain and domain limits. Avoid “first,” “most realistic,” “state of the art,” “geometrically faithful hidden face,” and “novel by default” without evidence and a refreshed search.

## Recommended research direction

Build the user's modular baseline first. If naive assembly exhibits the anticipated view/light failures, pursue **Candidate C** as the central hypothesis and **Candidate A** as a controlled training extension. Keep individual component inventions optional. This concentrates the claim on one problem we can explain, test, ablate, and compare against existing integrated methods.

Stop or revise a candidate when its benefit disappears under matched data/compute, violates identity geometry, needs unavailable labels, or cannot be distinguished from the nearest prior work. A careful negative result is useful; renamed existing functionality is not a defensible paper.
