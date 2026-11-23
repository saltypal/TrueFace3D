# Notebook and execution workflow

This specifies a future standalone experiment launcher for Local Jupyter, Colab and Kaggle. It is not a notebook already run or a promise of every accelerator working. Reusable project logic stays in Git source modules. Build one small baseline launcher first, then enable component configurations through the same interface.

## Cell-by-cell specification

| Cell | Content | Required output / checks |
|---|---|---|
| 0, Markdown | Experiment goal, one-photo input boundary, expected artifacts, known limitations | Clear component, baseline/challenger or improvement, links to plan/experiment card |
| 1, configuration | Remote URL, exact commit/ref policy, safe temporary clone path, input/model roots and persistent output root | No secrets in committed notebook; explicit mode and settings |
| 2, clean reset and sync | Resolve environment-specific temporary clone, verify allowed parent/ownership marker, discard only that disposable clone when requested, fetch/clone exact source and checkout | Never delete workspace, data/model mounts, persistent results or HOME. Verify final absolute path and symlink resolution before removal |
| 3, revision proof | Print checked-out commit, dirty state and requested versus resolved ref | Use exact commit for reported runs; latest branch must resolve once and be recorded |
| 4, environment | Detect local/Colab/Kaggle, CPU/CUDA/TPU availability, driver/runtime/Python | Backend eligibility report, actual memory/device, unsupported component states |
| 5, dependencies | Install pinned component lock/environment or upstream requirements validated for this runtime | Record versions; do not silently upgrade all packages; isolate incompatible upstream environments |
| 6, persistence | Configure mounted storage/Kaggle output or artifact store; write/read test; retention/checkpoint policy | A destination's pathname alone is not proof it survives runtime termination |
| 7, assets/data | Resolve only selected licensed weights/inputs, verify hashes, manifest/splits and label availability | Fail clearly on missing access. Public code is not proof of model/data authorization |
| 8, reproducibility | Seeds/config snapshot, precision, deterministic settings where compatible | Record nondeterministic CUDA/render operations and repeatability scope |
| 9, contract fixtures | Run targeted preflight tests for coordinates/model/UV/color/backend | Test report; asset-dependent skipped tests remain skipped |
| 10, native baseline | Execute original upstream inference and save native outputs | Original command/config/output, status and cost; no hidden adapter changes |
| 11, adapter/integration | Convert into proposed versioned artifacts and validate | Artifact manifests and compatibility reports before downstream consumption |
| 12, comparisons | Run fixed baseline/challenger and declared improvement variants | Same evaluation manifest; save failures; no test-set model selection |
| 13, metrics/profiling | Quality, parameter count, time/memory, supported operation counts | Raw records, coverage gaps and aggregation; profiling separate from ordinary timing |
| 14, visual diagnostics | Same camera/light montages, masks, depth/normals, UV/eye/detail and changed-view/light renders | Native and enhanced outputs separate; labels distinguish real/synthetic/GT/predicted |
| 15, final persistence | Flush metrics/logs/config/assets; checksum/readback; environment + Git/upstream revision summary | Durable run manifest; interrupted/resumed runs remain traceable |
| 16, Markdown | Result interpretation, passed/failed gates, limitations and next handoff | Measured, proposed and missing evidence separated |

The first executable setup section must offer the complete clean-sync path. For local execution, disposable clones must be separate from the real development checkout. For Colab/Kaggle, deleting a clone never deletes mounted persistent data/results. A clean reset invalidates stale local imports; restart the interpreter if imported modules cannot be safely refreshed. Do not claim that `git pull` alone resets a dirty or wrong-branch checkout.

## Hardware/backend policy

CPU is valid for file/contract/data checks, selected lightweight components and a Cycles reference path. RealDenseFace's CUDA solver and nvdiffrast require compatible NVIDIA execution; detecting a TPU does not make them work. T4, P100 and A100 support is established per pinned environment/model by actual smoke tests, not by a generic device name. Precision/architecture compatibility differs. Dual T4 requires explicit process/data partitioning; it is not one 32 GB GPU.

No automatic external cloud submission, paid job, remote repository publication or dataset-wide download is implied by this planning document. Prepare configs/launchers first. User-requested local/authorized notebook runs can then be performed and monitored by Luna High through available tools. If interactive account access is unavailable, report it; do not claim a notebook ran because it was generated.

## Cell observability and run artifact map

Every setup/computation cell prints a short purpose/start message, important configuration and input/output paths, useful progress, and completion status with elapsed time and key artifacts. Long operations write flushed or structured progress at affordable intervals. Do not dump tensor elements, every iteration or secrets. A cell must be distinguishable as active, stalled, failed or complete.

The environment cell prints the actual interpreter path, Python/package versions, resolved repository/source revision, intended dependency specification and observed hardware/backend. On Kaggle, record the managed runtime and packages installed into the active kernel. A local virtual environment does not transfer by uploading source. A required restart must occur before expensive execution, followed by a fresh setup/preflight.

Keep an executed `.ipynb` with cell IDs/execution counts/rich outputs, and incrementally persisted per-cell stdout/stderr/error logs where practical. Each cell record includes start/end/status, timing, run/source identity and failure traceback. Save figures, metrics and important tables separately with stable cell/stage identifiers. Retain useful partial outputs when a cell fails. Document any truncation, redaction or inaccessible native-process/kernel output; do not promise capture of a killed kernel's final buffer.

Before launch, create or update the single `KAGGLE_RUNS.md` artifact map at the project root. It records planned notebook identifier/purpose, source revision, configuration, intended runtime, inputs/dependencies and expected output locations. After launch record the actual notebook version/run ID, status, verified output paths, downstream consumers and retrieval/resume instructions. Expected files and files actually found use different fields. The initial map can record that no executable notebook/run exists yet.

Verify log capture and a representative persistence/retrieval roundtrip during preflight. Periodically flush artifacts before long work. Do not interrupt a healthy expensive cell merely to retrofit prints; inspect current outputs first and schedule instrumentation for the next run. Resume is distinct from loading weights: future training resumes require optimizer/scheduler/scaler/progress/RNG state and compatibility checks. Nothing here starts training in the fourteen-day window.

## Persistent artifact layout

```text
<persistent_root>/<experiment_id>/<run_id>/
  run.json, config.yaml, environment.json, source_state.json
  inputs_manifest.json, assets_manifest.json, split_manifest.json
  native_outputs/, artifacts/, renders/, diagnostics/
  metrics/per_identity.csv, aggregate.json, failures.jsonl
  profile/stage_times.json, memory.json, operations.json
  logs/, checkpoints_if_any/, final_manifest.json
```

Kaggle `/kaggle/working` is a session output directory; verify saved notebook-version outputs or export to the chosen durable store before treating results as persistent. Colab ephemeral `/content` is not durable storage. Local storage durability also requires a known path outside disposable clones. Log an interrupted run and resume metadata instead of merging it invisibly with a new run.

## Planned command interface

These commands are **proposed and do not exist yet**:

```powershell
python scripts/audit_environment.py --config configs/baseline.yaml
python scripts/run_component.py --component C1 --config configs/geometry/realdenseface.yaml
python scripts/benchmark_component.py --component C1 --manifest <permitted_manifest> --config <frozen_config>
python scripts/render_scene.py --scene <scene_manifest> --config configs/rendering/cycles_reference.yaml
```

The implementing chat must replace proposals with exact working commands in its status record. Native upstream commands are independently pinned and saved. An implementation should not rely on variables or secret cells from a different notebook.
