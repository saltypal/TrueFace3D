# Initial experiment cards and parameter-control acceptance

These cards make the first proposed changes concrete. They are **planned, unrun protocols**, not discovered improvements. Their numeric settings below are provisional development defaults. Main Sol freezes the final values/config/hash after development and before held-out evaluation. Any different mechanism receives a new card/version; no outcome-driven rewriting of the original comparison.

## H1-v1: conservative visibility-aware UV appearance

**Failure/question:** a single photo projects foreground occluders, grazing-angle samples or incorrect crop coordinates into the surface. Correct visibility is baseline engineering. Does a conservative reliability blend improve edge/occluded appearance without erasing visible personal marks? Prior art already covers visibility and confidence weighting; novelty remains unverified.

**Prerequisites:** G1 geometry/camera, fixed OpenDelight checkpoint/preprocessing, masks and UV chart, valid development targets if making accuracy claims. Keep geometry, albedo network, crop, output UV resolution, renderer/light and completion prior identical across variants. No weights change and no added inference image.

**Variants:** H1-A naive geometric projection with the same image mask; H1-B z-buffer/self-visibility plus explicit external-occlusion mask and hard grazing rejection; H1-C H1-B plus a declared soft blend in low-reliability observed texels. First prove projection invariants on fixtures. H1-B is the serious engineering baseline; beating only H1-A cannot establish a novel algorithm.

For an observed UV texel, let P be its projected predicted albedo and A0 the same conservative completion prior used by every variant. For H1-C, set r = clamp(cos(theta),0,1)^gamma for accepted facing samples, then A = r P + (1-r) A0. Outside valid observation use A0 and mark completed. Use a development-provisional gamma of 1, reject cos(theta) below 0.2, and set r to 1 for a declared high-reliability region at or above 0.85. These values are heuristic, not calibrated probabilities. Resolve multiple samples with a documented sampler; store raw P as well as blended A.

The prior A0 can be a robust median skin albedo plus explicitly masked symmetry completion, selected once on development. Do not invent observed anatomy. A confidence-weighted normalized average with only one source can cancel its own weight and change nothing; the explicit prior blend above states the actual mechanism and its risk. It may smooth marks or introduce color bias, which would reject H1-C.

**Evaluation:** raw masked albedo error when valid labels exist; coverage/excluded fraction; separate observed-versus-completed regions; visible-mark retention and same-scene held-out light/view evidence. All variants use the same evaluator masks in the common-region score, plus full coverage/failure reporting. Count projection/completion time and extra memory, not just network FLOPs. If only fixtures/unpaired portraits exist, report correctness and diagnostics, not material accuracy.

**Adoption:** finalize benefit/preservation/cost thresholds on development, lock variant/config, then report held-out outcomes once. H1-C must beat H1-B meaningfully without unacceptable observed-region degradation. Otherwise keep H1-B. Neither masking nor a heuristic angle blend is claimed novel by itself.

The primary endpoint is common-valid-region raw linear-albedo MAE when appropriate GT exists; otherwise synthetic known-albedo UV MAE is the endpoint and real portraits remain diagnostics. Preservation endpoints are observed-region/mark error and excluded-area fraction. The experiment is not ready to code as a benchmark until the card names the actual available dev/holdout manifests, freezes the endpoint's minimum meaningful delta and preservation tolerance, and records a stage-time ceiling plus maximum observed-device memory budget. Suggested engineering ceilings are at most 2x H1-B projection/completion time and no additional neural forward, subject to Day 1 resource validation. These are proposed policy bounds, not measured costs or universal acceptance thresholds.

## H2-v1: bounded scheduling of a small photometric fit

**Failure/question:** can a small staged optimizer improve appearance while avoiding harmful detail deformation compared with an equally bounded joint fit? This first experiment does not refine FLAME identity, expression, camera or a free UV texture. It is an inference-time optimizer, not network retraining.

**Prerequisites:** fixed scene/geometry/albedo, G5a gradients, G6_fixture camera/material semantics and validated C3 displacement if testing detail variables. If C3 is unavailable, run only the light-only baseline and defer the staged-detail claim. Full shape/material joint optimization remains a later separately reviewed card.

**Gauge and variables:** freeze exposure at 0 EV, white balance at unity, albedo/color scale, camera, all FLAME parameters and eye optics. Use three fixed-direction white directional lights plus a fixed-form ambient term in the initial differentiable shader. Optimize their four nonnegative intensities in [0,4] relative shader units, initialized provisionally to [0.6,0.3,0.1,0.05]. Keep directions and the unit irradiance convention fixed, export that convention to Cycles, and record initialization. This removes an unconstrained exposure/light scaling pair; it does not recover absolute personal albedo or calibrated physical illumination.

When valid displacement exists, optionally optimize at most six deterministic region amplitudes s in [0,1], initialized to 1, multiplying the existing height map. Define nonoverlapping masks and their seam blending once from the same UV template. Use forehead, left/right cheek, left/right nasolabial and chin regions only when supported by the actual chart. Unavailable regions reduce K and are recorded. No new wrinkle is generated and coarse identity remains frozen.

**Loss:** normalized mean pseudo-Huber RGB residual on valid observed skin, using rho(r) = delta² (sqrt(1+(r/delta)²)-1), averaged over valid pixels and channels in linear RGB. Provisional delta = 0.05. Add 0.01 × mean((s-1)²) when K>0 and 0.001 × mean((l-l0)²) for light intensities. All coefficients and variables are dimensionless in the declared normalized shader convention. Record each term, valid count and mask. Ignore external occlusions and explicitly identified saturated observations; do not derive the mask from GT or optimize it to hide errors. Empty/insufficient valid evidence returns baseline plus failure status.

**Variants and budget:** F is fixed scene, zero steps. L is light-only, up to 40 steps. J optimizes the same four lights and K amplitudes jointly for 40 steps. S fits lights for 15 steps then the same joint variables for 25. J/S use identical initialization, masks, bounds, optimizer, losses and total iteration cap. Provisional Adam step size is 0.03, with parameter projection onto bounds after each step and finite-state checks. Record actual forward/backward counts and time; equal iteration caps are not equal FLOPs. Do not add a weak, unregularized giant-variable strawman just to make S win.

**Selection:** use development data to freeze defaults and compare J/S against L and F. Validate any chosen candidate in the final Cycles scene with the identical asset/camera/light convention. A rasterizer-only gain that disappears or causes unacceptable regressions in Cycles is rejected or restricted to a diagnostic. Freeze the chosen policy before held-out evaluation; the evaluator cannot choose F/J/S per test identity using GT. Numerical failures return F and remain counted.

**Claim limits:** this tests scheduling/bounds on low-dimensional lighting/detail, not recovered physical anatomy or a new general joint inverse-rendering method. If only input-view RGB improves, call it photometric fitting and do not infer better geometry. Use paired geometry/held-out-light evidence where available, costs and failures. Generic staged fitting and robust losses are prior art.

The primary endpoint is controlled held-out-view/light linear-RGB MAE when valid matched evidence exists; scan/depth/normal changes and observed-region quality are preservation endpoints. Without those labels, use synthetic held-out-render error and restrict the real-image claim. Before coding the benchmark, name actual manifests/fixture IDs, freeze endpoint/tolerances and a measured-device time/memory ceiling. The initial workload ceiling is 40 optimizer steps per declared variant, K<=6 detail amplitudes and no neural-weight updates. Record actual work because J and S need not use identical FLOPs. Fixed coarse coefficients imply zero coefficient drift by construction; measure actual shape/detail errors rather than claiming this tautology validates identity recovery.

## Parametric-control regeneration and export gate

An exported mesh or .blend alone does not establish an editable parametric reconstruction. Deliver the named FLAME parameters, model/basis/hash references, authorized asset-resolution instructions, UV/material linkage and regeneration command. Do not redistribute separately restricted model files by assumption.

Required acceptance cases are:

1. Reload saved parameters and the exact model to regenerate the base surface within recorded numerical tolerance. Compare array/landmark/projection equivalence with native output after known coordinate conversion.
2. Change a small documented head-pose control and reset it. Verify geometry, eye-local attachment and projection follow the declared transforms, then return to baseline within tolerance.
3. Change a supported expression/jaw control and reset it. Verify the FLAME base changes and the UV/materials remain attached. Report basis/conditioning mismatch rather than forcing another model's coefficients into the detail decoder.
4. Apply a supported gaze/pupil change when C4 is present and test socket/lid agreement. If eye fitting is deferred, the generic eye's control behavior is still a fixture, not a recovered gaze claim.
5. Reload the complete asset bundle in the tested environment and repeat a control change. A baked static OBJ/GLB can be a supplemental export, but its lost controls must be declared. The regeneration bundle is the parametric deliverable.

Detail policy is explicit: `validated_dynamic`, `source_pose_only`, or `disabled_on_expression_change`. The first scope can preserve source-pose detail and disable it when an unsupported expression changes; it cannot claim dynamically correct wrinkles. Eye attachment recomputes from the verified head/joint transform. These tests belong to C1/base and C4/C6 integration gates, not a vague later animation promise.

## Single holdout-use rule

Main Sol selects/fixes the complete method, component policies, thresholds, masks, renderer, dataset protocol and configuration hash using development only. Run the held-out identity/view/light suite for that frozen configuration and preregistered comparators. Do not make per-test-case choices with labels, silently drop failures, or tune the next component on a previously final holdout.

If a held-out result triggers a correctness fix, preserve the original result and version, report the fix, rerun affected evaluations and disclose that this holdout is now development-exposed. Use another untouched set or restrict generalization claims. Insufficient data does not authorize recycling the same people as an untouched test. Official test submission limits and user authorization remain binding.
