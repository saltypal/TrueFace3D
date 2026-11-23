# Feasibility and scope

## Verdict

The modular direction is technically plausible: image-to-parametric geometry, albedo, displacement and explicit eyes can form one scene. That is a **design feasibility judgment**, not proof the selected releases execute together. The archive currently contains papers and notes; the project has no implemented model pipeline or measured device results.

The hardest work is the transfer between components: camera/crop coordinates, FLAME versions, UV charts, displacement units, normal frames, lighting and color interpretation. Separately attractive model outputs may be mutually inconsistent. A shaded texture cannot serve as intrinsic albedo, a normal map does not move the silhouette, and a different topology cannot accept a resized FLAME map by assumption.

Single-view reconstruction has unavoidable ambiguity. Hidden facial shape, true skin reflectance versus illumination, and exact corneal/iris anatomy are not uniquely recoverable from one ordinary photograph. Priors and fitting can improve plausible consistency and measured accuracy on a stated test distribution. They cannot certify invisible personal anatomy. A neural architecture change can run without retraining only when its trained interfaces and computation remain valid; changing layers or latent interpretation usually requires training or adaptation evidence.

## Fourteen-day deliverables

**Required engineering target:** one-photo input, reproducible coarse parametric mesh/camera, visible-region skin appearance, declared completion/defaults, optional validated detail and explicit eyes, one Cycles scene, native renders and reloadable assets. If detail/eye integration fails its gate, the valid coarse baseline remains the deliverable and the missing feature is stated.

**Required research target:** comparisons for all six components, per-paper dispositions for all 44 sources, a small executable benchmark set, a decision log for adopted/rejected changes, measured latency/memory/parameter counts and honest operation-count coverage. At least one full pipeline configuration should be rerunnable from a pinned source revision and persisted artifacts.

**Stretch:** a matched test of one or two inference-time mechanisms, an additional runnable geometry or material competitor, and a bounded joint refinement experiment. No publication acceptance or state-of-the-art improvement is promised.

**Deferred:** all network retraining, large diffusion fine-tuning, learned eye-database reproduction, new large datasets, full head/hair/teeth, causal video and production deployment. A missing checkpoint is not an instruction to train immediately.

## Feasibility risks and response

| Risk | What must be verified | Response within this window |
|---|---|---|
| Essential FLAME/checkpoint access | Actual allowed files, model/version/hash and usable configuration | Audit on Day 1. If essential assets remain unavailable after a bounded attempt, try a verified permitted challenger or deliver the source/asset blocker report |
| Local GPU cannot execute solver/model | Device, VRAM, driver/runtime and smoke inference | Use an available authorized Colab/Kaggle GPU. CPU supports documentation/asset tests/Cycles reference renders; it is not an automatic replacement for CUDA solver inference |
| Dataset access is delayed | Usable paired image/scan or material targets with allowed split | Use a declared synthetic fixture suite and real-image diagnostics. Leave real geometry/material accuracy unevaluated until valid labels exist |
| Topology transfer dominates schedule | Target model/UV, registration error, units and normal frame | Prefer compatible branch. Disable unvalidated detail/material transfer rather than hide a distorted identity |
| Photometric fit improves input view but worsens geometry | Held-out real cameras/lights or scanned geometry; synthetic checks | Freeze coarse identity first, bound updates, and retain no-refinement baseline |
| A rich material model exceeds compute | Actual batch-one peak memory and total time | Keep lighter albedo plus declared physical defaults; do not claim lightweight diffusion support without measurement |
| Planning consumes implementation time | Daily baseline artifact and stop rules | Freeze optional scope by Day 3, reserve Days 11–14 for integration/evaluation and repair |

Resource assumptions are provisional. The earlier laptop/free-cloud discussion is not a live inventory. Day 1 must establish actual available hardware, asset access, daily working time and storage. Plan by relative days; elapsed calendar time is not a passed scientific gate.

## Model rejection is conditional

Do not reject a useful model merely because one limitation cannot be eliminated. Every monocular method has residual ambiguity. Decide whether a limitation violates the current task, can be contained by masks/defaults, justifies a separate experiment, or makes the model unsuitable for this phase. Record concrete incompatibility, resource/access failure or worse measured tradeoffs. Do not invent a post-hoc story to justify a preferred winner.

The existing model table is the first candidate set. A challenger may replace it after matched evidence and a documented architecture decision. Changing a surface representation from FLAME to an implicit head is a larger project decision than exchanging two FLAME-compatible observers.
