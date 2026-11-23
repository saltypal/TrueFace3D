# Low-level design and function contracts

This is a proposed implementation interface, not existing executable code. Implement the minimum functions needed by the active checkpoint; do not generate the entire repository in one pass. Preserve the earlier module layout. Concrete signatures can evolve through a recorded contract version and consumer migration.

## Proposed source boundaries

```text
src/trueface3d/
  contracts.py             typed records and version validation
  observation.py           original image, masks, reversible crop ledger
  geometry/                upstream adapters and coarse geometry validation
  appearance/              albedo, visibility, UV projection/completion
  detail/                  native detail extraction and target transfer
  eyes/                    procedural assets, cue fitting and attachment
  rendering/               scene assembly, fitting backend, Cycles adapter
  fitting.py               bounded optimization policy
  evaluation/              quality metrics and runtime/cost reporting
  provenance.py            asset hashes, revision/environment and manifests
  pipeline.py              explicit component orchestration
configs/                   versioned baseline/challenger/experiment settings
scripts/                   thin audit, infer, benchmark, render entry points
notebooks/                 portable exact-commit experiment launchers
tests/                     contract, geometry, gradient and integration checks
docs/status/               component handoffs and main project status
docs/decisions/            architecture decisions, scope and rejected choices
```

Upstream checkouts/assets remain separately configured and pinned; their licenses and source versions belong in the asset manifest. Heavy private inputs, checkpoints and results are not committed as ordinary source files. A notebook orchestrates reusable modules rather than defining a second hidden implementation.

## Common artifact schema, version 1 proposal

Every record has `schema_version`, `artifact_id`, `producer`, `producer_revision`, `source_run_id`, `input_artifact_ids`, `payload_paths`, `payload_hashes`, `validity`, and `provenance`. Use JSON metadata plus numerical payloads such as NPZ/images/mesh files. Relative paths are resolved against the artifact manifest directory and checked for integrity. Status/error metadata survives a failed run.

| Record | Essential payload and semantics |
|---|---|
| Observation | Original image hash/size; crop image and original-to-crop plus inverse transform; mask coordinate frames; landmark definition; color encoding and exposure assumptions |
| GeometryAsset | Float vertices N×3 in declared canonical meters; faces F×3; UV vertices U×2 and independent UV-face indices F×3; model/topology/UV hashes; named parameter arrays and basis IDs; native camera/transform; priors if supported |
| Camera | Explicit perspective or weak-perspective type; parameter model; world/head-to-camera transform; image size; crop/intrinsic relationship; source/estimated calibration status. Do not invent K for weak perspective |
| AppearanceAsset | UV diffuse H×W×3; color encoding; observed/completed/invalid masks H×W; visibility/heuristic reliability evidence; completion method; roughness/specular/SSS defaults with provenance |
| DetailAsset | Displacement H×W in meters; source/target base IDs and UV map; optional residual normals H×W×3 with tangent/object frame, orientation and encoding; fixed-prior inclusion; expression condition and transfer method |
| EyeAsset | Geometry/material payloads and component IDs; eye-local-to-head transforms; gaze/pupil/iris controls with conventions/limits; observed cue mask; generic optics provenance |
| ScenePacket | References to validated artifacts; camera; illumination/exposure/color pipeline; material interpretation; render mesh/subdivision; active variable bounds; component fallback states |
| RenderPacket | Linear native RGB, display RGB, alpha, depth/normal/UV/component/material passes; backend/version/settings and scene ID. View/illumination transforms explicit |
| RunRecord | Config and exact revisions; assets/split hashes; environment/device; stage timings/memory; quality/operation-count status; failures and persisted artifact list |

Confidence is not automatically calibrated probability. Use distinct `visibility`, `heuristic_reliability`, `predicted_log_variance`, and `calibrated_confidence` fields where supported; record the method and calibration data. Do not copy depth uncertainty into material confidence without validation.

## Foundation functions

| File / callable | Input → output | Validation and failure behavior |
|---|---|---|
| contracts.validate_artifact(record, expected_type, version) | Manifest and payloads → validation report | Required fields, arrays, finite values, indices, versions and hashes; fail with artifact/field name |
| contracts.validate_scene_dependencies(scene) | Scene → compatibility report | Units, UV/base/basis IDs, frames, color and linked input versions; reject incompatible consumers |
| observation.load_observation(image_path, config) | Image → immutable original record | Allowed decoding/channel type and finite dimensions; never overwrite original |
| observation.make_crop(observation, transform, output_size) | Original → crop/ledger | Invertible transform; record padding/resize and mask interpolation |
| observation.map_points(points, source_frame, target_frame, ledger) | Points → mapped points/valid mask | Explicit image-coordinate convention, no silent transpose/reflection |
| provenance.resolve_assets(manifest, roots) | Asset requirements → existing allowed files/hashes | Missing/access-ineligible are explicit; no automatic bulk dataset download |
| provenance.capture_run_context(config, assets, split) | Context → RunRecord draft | Commit, dirty state/patch hash, upstream revisions, device and package versions |
| provenance.persist_run(record, destination) | Record/payloads → durable manifest | Write temporary then finalize, checksum/readback; runtime reset destination handled explicitly |

## Component callable inventory

The component guides explain experiments and gates. These are their minimum reviewable APIs. Names express responsibilities; a single adapter can implement several without extra abstraction.

| Component / callable | Input → output / important condition |
|---|---|
| C1 geometry.audit_backend(config, asset_manifest) | Requirements/device/assets → eligibility report and missing dependency list |
| C1 geometry.run_native(observation, config) | Image/crop → upstream output paths, raw config and stage logs; unchanged baseline first |
| C1 geometry.decode_native(native_result, model_assets) | Native result → GeometryAsset plus explicit native camera metadata |
| C1 geometry.canonicalize_geometry(asset, frame_spec) | Known model units/axes → canonical asset and inverse transform; unknown units stop conversion |
| C1 geometry.project_landmarks(asset, camera, ledger) | Landmarks → original-image projections/valid mask; never compare mismatched definitions |
| C1 geometry.export_geometry(asset, destination) | Validated base/parameters → reloadable mesh/parameter bundle and hashes |
| C1 geometry.regenerate_from_parameters(bundle, controls) | Saved model/basis references and declared control edits → regenerated parametric base and attachment transforms; verify reset/roundtrip |
| C2 appearance.predict_image_albedo(observation, config) | Preprocessed crop → image-albedo artifact plus inverse alignment/preprocessing metadata |
| C2 appearance.compute_visibility(geometry, camera, masks) | Surface/camera → z-buffer visibility, facing/grazing-angle evidence; separate external occlusion mask |
| C2 appearance.project_to_uv(albedo_image, geometry, camera, ledger, visibility) | Visible image evidence → UV albedo/sampling weights and observed mask |
| C2 appearance.complete_unobserved(uv_asset, policy) | Missing regions → conservative completion plus completed mask; preserve high-confidence observed texels |
| C2 appearance.validate_color(asset, renderer_color_spec) | Encoded/aligned maps → checked conversion metadata; scalar maps stay non-color |
| C3 detail.extract_native(observation, config) | Crop → detail code, native condition and displacement/fixed-prior metadata |
| C3 detail.validate_transfer(source, target, registration) | Model/UV/frame/units → transfer eligibility and error report |
| C3 detail.transfer_displacement(native_detail, target, registration) | Compatible height field → target DetailAsset; no arbitrary UV resizing as registration |
| C3 detail.apply_displacement(base, detail, subdivision) | Parametric base → derived render mesh/normals with source links |
| C3 detail.measure_artifacts(base, render_mesh, masks) | Meshes → drift, seam, flip and amplitude diagnostics |
| C4 eyes.make_eye_asset(template_config) | Declared generic optics/geometry → local eye components; no claim of learned paper database |
| C4 eyes.attach_to_head(eye_asset, geometry, transforms) | Local eyes/head → head-frame placement with ownership of lids retained by geometry |
| C4 eyes.fit_visible_cues(asset, observation, camera, bounds) | Cues → bounded pose/gaze/pupil estimate and fit evidence; no valid cues keeps default with low evidence |
| C4 eyes.check_socket_and_lids(eyes, head, camera) | Assembly → penetration/occlusion/iris agreement diagnostics |
| C5 rendering.rasterize_scene(scene, settings) | Canonical scene → triangle/depth/barycentric buffers with documented camera conversion |
| C5 rendering.shade_differentiable(buffers, materials, lighting) | Buffers → linear approximate render and passes; explicit supported light terms |
| C5 fitting.build_variables(scene, policy) | Scene → declared optimizable variables/bounds and frozen baseline snapshot |
| C5 fitting.compute_losses(render, observation, priors, masks) | Prediction/evidence → named losses, weights and valid counts; empty masks are errors or explicit skips |
| C5 fitting.optimize_bounded(scene, observation, policy) | Scene → candidate scene and full iteration log, with early-stop/failure reason |
| C5 fitting.select_candidate(baseline, candidate, gate_report) | Reports → accepted scene or unchanged baseline; no selection on held-out test GT |
| C6 rendering.assemble_scene(artifacts, camera, light, config) | Compatible assets → ScenePacket; required versus optional/fallback components explicit |
| C6 rendering.build_cycles_scene(scene, engine_config) | ScenePacket → native .blend plus conversion report and pinned node/socket semantics |
| C6 rendering.render_cycles(scene_path, views, lights, settings) | Fixed asset, prescribed cameras/lights → native images/passes and costs |
| C6 rendering.export_bundle(scene, target_formats) | Native asset → supported exports plus limitations; displacement/optics handled explicitly |
| C6 rendering.verify_export_roundtrip(bundle, reference) | Reloaded export → geometry/material/camera comparison and unsupported features |

## Evaluation and orchestration functions

| Callable | Input → output |
|---|---|
| evaluation.validate_dataset_manifest(manifest, split) | Records → access/provenance/identity-disjoint and label-availability report |
| evaluation.compare_geometry(prediction, target, protocol) | Mesh/scan → benchmark-defined measurements with unit/alignment metadata |
| evaluation.compare_images(prediction, target, masks, metric_config) | Native maps/renders → raw and separately aligned scores, valid counts and implementation versions |
| evaluation.compare_eyes(prediction, cues_or_gt, protocol) | Eye fit → reprojection/occlusion and optional GT gaze/shape metrics |
| evaluation.profile_stage(callable, inputs, timing_config) | Measured execution → cold/warm time, n and memory scope |
| evaluation.count_operations(model, inputs, counter_config) | Countable path → raw per-op estimates, convention, unsupported list and coverage status |
| evaluation.aggregate_paired(records, identity_groups) | Same-case outcomes → per-identity deltas, exploratory intervals and failure accounting |
| pipeline.run_component(component_id, config, inputs) | Valid dependency artifacts → outputs, gate report and persistent run record |
| pipeline.run_scene(config, observation) | Selected validated components → baseline scene/results with explicit fallbacks |

## Errors, caching and tests

Distinguish `missing_asset`, `unsupported_backend`, `incompatible_schema`, `unknown_units`, `topology_mismatch`, `empty_valid_region`, `numerical_failure`, `out_of_memory`, and `evaluation_label_missing`. Catch errors at orchestration boundaries; save trace/context; never substitute a random/default model output as a successful prediction.

Cache keys include input bytes, crop/preprocessing, source/weight/model hashes, config, dependency artifact IDs and relevant precision/backend behavior. A stale crop, mesh or UV chart invalidates downstream caches. Canonical scenes are immutable snapshots; optimization creates a candidate and preserves the baseline.

Required tests address invariants: coordinate roundtrip, camera projection, seams with independent UV indices, known unit conversion, invalid topology rejection, single fixed-prior application, finite normals, gradient behavior on supported differentiable fixtures, cache invalidation, export/reload and failed-run persistence. Do not create tests that merely restate every helper's implementation. Real pretrained smoke tests are opt-in and require actual assets; skipped asset-dependent tests cannot pass a model gate.

The [parameter-control acceptance cases](11_EXPERIMENT_CARDS.md) require regeneration after pose/expression edits and tested eye attachment, including explicit source-pose-only detail fallback. Static mesh reload is insufficient. The same document defines concrete initial H1/H2 variables, gauges, losses, bounds, comparisons and held-out-use rules; changes are versioned rather than inferred by a Low implementer.
