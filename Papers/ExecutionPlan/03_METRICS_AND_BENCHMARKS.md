# Metrics and benchmark protocol

This is a specification. Empty results mean unmeasured, not zero error or zero cost. The [published comparison](01_COMPARATIVE_STUDY.md) and our future run tables are different records.

## Evidence tracks and samples

1. **Contract fixtures:** tiny known camera/mesh/UV/color/eye scenes. Establish implementation correctness; do not claim real-image reconstruction quality.
2. **Smoke suite:** six permitted images covering frontal, oblique, expression, occlusion, strong light and low resolution. May overlap development. Used for failures and execution checks only.
3. **Development suite:** target up to 12 distinct identities with available labels; choose parameters here. Document actual counts and pretrained exposure.
4. **Held-out evaluation:** target up to 24 other identities or an allowed official validation subset. Freeze method/config before running. This is an exploratory subset study, not a leaderboard reproduction. More frames of the same person do not increase the independent identity count.
5. **Official benchmark:** optional, subject to authorized data/protocol and server response time. Public test scans may be inaccessible. Do not send emails or publish benchmark entries as an implicit step; preparation of local evaluation packages is sufficient until the user explicitly requests submission.

Targets are proposed budgets, not datasets we possess. If only fixtures and unpaired images are available, label real geometry/albedo accuracy `not_evaluated_no_valid_gt`. Do not manufacture GT by treating another predictor's output as a scan.

Split source identities before rendering, augmentation, restoration or pseudo-labeling. Keep all derived records with their source identity. Audit source overlap and upstream pretrained exposure. Unknown overlap remains disclosed. Tune on development, not held-out scans/lights/test submissions.

## Component metric checklist

| Component | Main quality measurements | Diagnostics / preservation | Cost and scope metadata |
|---|---|---|---|
| C1 geometry | Official scan-to-mesh distances in mm, neutral and posed tracks separately; correct normal consistency definition | Landmark reprojection, silhouette IoU where valid masks exist, failure rate, neutral identity drift; metric and scale-aligned tracks distinct | Network and solver latency, iterations, initialization, FLOP coverage, parameters, peak memory |
| C2 appearance | Masked raw-color albedo MAE/PSNR, SSIM and LPIPS against valid labels; separately color-aligned author-style scores if reproduced | Observed-region coverage, unobserved fraction, seam error, mark retention, same-scene relighting; skin-tone/exposure drift | Image-network and projection time, map size, completion method and confidence definition |
| C3 detail | Scan distance and angular normals on a fixed base; depth/displacement error only after valid registration | Coarse identity drift, wrinkle alignment, seams, flipped faces, bounded amplitude, moving-light behavior | Detail-network time, subdivision/normal generation/export cost, triangle count and displacement resolution |
| C4 eyes | Iris/limbus reprojection in px and normalized eye width; gaze angle only with valid gaze GT | Lid occlusion, socket penetration, left/right consistency when justified, low-confidence rejection, highlight response | Optimization iterations, eye mesh size, fit time; optics parameters/default provenance |
| C5 fitting | Primary task metrics before/after fitting; controlled synthetic parameter recovery | Loss by term, gradient check error, failures/divergence, identity drift; held-out views/lights | Forward/backward time, iteration count, peak memory, unsupported operations |
| C6 rendering | PSNR/SSIM/LPIPS against controlled matched real/synthetic views where valid | Silhouette/depth/normal/material passes, relighting plausibility, native-versus-export parity, skin/eye occlusion | Pixels, samples per pixel, bounces, denoising, render time, engine/version/device, CPU/GPU memory |
| Optional restoration | Downstream geometry/material deltas and identity preservation under matched raw/restored inputs | Observed evidence altered, hallucination, transform fidelity | Extra compute and total pipeline latency |

Identity embedding similarity is a supporting perceptual proxy, not proof of identity geometry or fairness. Declare the recognition model/pretraining and keep it consistent. FID is unsuitable as the primary score on a tiny identity set. Native output and any 2D enhancement have separate results.

## Exact metric semantics

For scan vertices q_i and predicted surface M, one-way scan-to-surface distance is d_i = min over x in M of ||q_i-x||. Report mean, median and spread using the benchmark's unit conversion and aggregation. Symmetric Chamfer is a different metric. State whether distances are squared, how surfaces are sampled and which regions/masks are used. Do not call raw squared distances “mm.” NoW follows its official scan-to-mesh protocol; scale-invariant alignment cannot establish metric scale accuracy.

Reprojection error is ||project(v_i)-u_i|| in pixels at a specified image resolution. Normal angular error is acos(clamp(dot(n_pred,n_gt),-1,1)) in degrees for oriented unit normals. Using absolute dot products hides flipped normals and must be separately labeled if a source requires it.

Masked image MSE averages squared error over valid pixels and channels. PSNR is 10 log10(data_range²/MSE); declare range and linear/display color space. Zero MSE requires explicit infinity handling. For SSIM/LPIPS, pin implementation, network/version, expected input range and mask treatment. Do not zero backgrounds differently between methods and call the score a skin metric. Use the same valid region; report excluded area. LPIPS on a bounding crop includes context and must be labeled accordingly.

Color-aligned albedo scores, such as OpenDelight's per-image six-parameter alignment, are evaluator-only. Store raw and aligned scores. Alignment with GT must not alter inference outputs or conceal an exposure/skin-tone failure. Scan geometry and ordinary captured RGB do not constitute measured diffuse/specular/roughness targets.

Held-out-view evaluation can use calibrated target cameras for rendering/evaluation, but those cameras/images must not improve the one-photo reconstruction. Register coordinate frames by a stated permitted transform; do not nonrigidly warp predictions to GT. Different expressions or unknown changing lights make direct pixel comparison confounded. Label those evaluations diagnostic, or use a controlled matched protocol. Generated novel views are consistency diagnostics, not held-out real GT.

## FLOPs, MACs and missing work

Report operation counts with the method, convention, scope, input shape and tool version. For dense convolution, a common MAC estimate is Hout × Wout × Cout × (Cin/groups) × Kh × Kw. For a linear layer it is input_width × output_width per item. Under a multiply-plus-add convention, those MAC terms correspond to roughly twice as many arithmetic FLOPs. Bias/normalization/activation/attention handling must be declared separately.

[fvcore's source documentation](https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/flop_count.html) counts one fused multiply-add as one operation under its FLOP label and exposes unsupported operators. Save its raw totals and `by_operator` breakdown with that convention. Do not blindly double a mixed aggregate that also includes non-MAC operations. A separately computed conv/linear MAC total can receive an explicit 2× arithmetic estimate.

[PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html) estimates FLOPs for supported operations; it is not a complete count of custom CUDA fitting, rasterization, all attention kernels or path tracing. Pin the actual runtime version and consult its matching documentation. Save unsupported/untraced operations. State `partial` coverage; missing work is `null/unsupported`, never zero. Tool output is an estimate, not a hardware performance counter for every instruction.

For a diffusion pipeline, include the actual number of denoiser evaluations, guidance branches, text/conditioning encoders, decoder and upsampler. For iterative fitting, include forward/backward/solver iterations and stopping policy. Report per-pass, per-stage and total known counts; do not advertise one network forward as end-to-end FLOPs. If a CUDA solver has no valid operation counter, give measured latency plus a labeled analytic complexity estimate, not a fabricated exact GFLOP value.

For Cycles, workload depends on pixels, samples, ray depth, scene geometry/material/light complexity and denoising. Mandatory reporting is those settings plus render time/memory and output quality. Do not compare rasterizer neural-forward FLOPs to path tracing as if they had identical work. Parameter count is `not_applicable` for a procedural shader, not evidence of zero computation.

## Timing, memory and reproducibility

- Record model loading/cold start separately from warm network, preprocessing, fitting, projection, export and final rendering. Full fresh-process latency includes startup and data transfer; cached runs identify the cache policy.
- Use batch one for the primary latency table. Default engineering timing budget: five warmups and twenty timed repetitions for inexpensive deterministic stages. For expensive diffusion/render jobs, three timed runs may be all that is practical; report n and range rather than a stable p95 claim. Extend only if results are unstable.
- Synchronize CUDA around wall-clock intervals or use valid CUDA events for GPU-only time. CPU timer/thread counts must be controlled. Avoid tracing/profiling overhead in the ordinary timing run. [Benchmark utility documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html) explains synchronized timing.
- Reset and record allocated/reserved GPU memory peaks per stage and the measured whole-process peak. PyTorch peaks exclude memory from Blender/other processes and some external allocations; use appropriate process/device measurements for those. Never sum unrelated stage peaks and call that the measured pipeline peak.
- Report total unique model parameters, frozen versus trainable neural weights, fitting-variable count, checkpoint bytes, dtype, precision/autocast/TF32 and actual resolution. Two GPUs do not pool memory automatically. Hardware changes invalidate a direct latency ranking.
- Every record contains source commit plus dirty state/patch identity, upstream commits, weight/model hashes, manifests/splits, seeds, device/backend, package versions, configs, transformations and failures.

## Adoption rules and statistics

Before each experiment, specify the target metric/direction, development data, held-out data, time/memory budget, failure cases and a tolerated preservation regression. Choose tolerances from baseline repeatability and task requirements on development data, freeze them, and record them in the experiment card. Do not choose thresholds after viewing held-out gains.

Compute paired per-identity deltas on the same samples; report failure counts and success-conditional quality separately. Do not silently drop failures or give them arbitrary perfect/worst scores. Bootstrap by identity, not by correlated frames. For very small n, call confidence intervals exploratory and show per-case results. No universal p-value or 1% gain establishes a meaningful contribution.

Adopt an upgrade only when its predefined benefit is supported, preservation/cost checks pass, and its added complexity is justified. Otherwise retain the baseline and record a negative/inconclusive result. If preservation GT is absent, adoption can be limited to engineering/demo use and cannot support an accuracy claim.

## Required result files

Keep `published_results.csv`, `our_runs.jsonl`, `per_identity_metrics.csv`, `aggregate_metrics.json`, `cost_profile.json`, `failures.jsonl`, `environment.json`, `config.yaml`, artifact manifests and native diagnostic renders per run. Published rows include paper/hash/page/table/protocol and `evidence_kind=author_reported`; our rows contain run ID and `evidence_kind=measured`. Hypotheses are experiment cards, not numeric result rows. Use null plus a reason for unavailable metrics.
