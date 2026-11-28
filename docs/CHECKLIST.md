# FaceScape-C execution checklist

Companion to `PLAN.md` (what and why) and `docs/KAGGLE_LESSONS.md` (how Kaggle runs must be built).
Tick a box only when its check has been **run and printed** in a notebook or test. Branch: `facescape-c`.

## Who does what (agents)

| Role | Model | Does | Never does |
|---|---|---|---|
| Lead | Opus | data/geometry contracts (frames, units, cameras), model adapters' coordinate handling, GARD-Face design and training code, every gate decision, review of every worker diff, commits | — |
| Module workers | Sonnet, **medium** effort | well-specified modules with their tests: metrics, restorer wrappers, adapter plumbing, notebook generators, plots | commit, change contracts in `src/facescape/`, edit files outside their brief |
| Test runners | Haiku, low effort | run `pytest`, execute notebooks headless, read Kaggle logs, report pass/fail with the failing lines | edit code |

Not Sonnet *low* for coding: the bugs that hurt this project are silent (a wrong frame, mm vs metres, a flipped
yaw). The first worker module (`src/robustness/profiles.py`, 72 tests) was written at medium and needed one
parameter change.

## Module map (one job per module)

```
src/facescape/      constants · sources (zips) · mesh (OBJ) · geometry (frames) · crop · splits
                    refine (ICP to scan) · build (clean/refine/degrade stages) · dataset (reader) · inspect (checks)
src/robustness/     profiles (14 x 3 degradations) · degrade (old NoW controls, kept)
src/models/         one adapter per mesh model, common interface fit(image, camera?) -> MeshResult   [to build]
src/restorers/      one wrapper per restorer, restore(image) -> image                               [to build]
src/metrics/        mesh (ICP protocol, Chamfer, NC, R2.5, F, CR, regions) · appearance (reuse) · robustness (CE/mCE, gap) [to build]
src/correction/     prior_corrector (v1, reused) · gard_face (denoiser, flow matching, training loop)  [to build]
src/kaggle/         notebook_kit (clone/zip cells, metadata) · runner (2 GPUs, progress) · setup (wheels)
scripts/            kaggle_push.py (kaggle-a / kaggle-b) · upload_dataset.py [to build]
notebooks/          make_*.py generators -> *.ipynb
```

## Notebooks

Kaggle notebooks share the same structure:
- **first cell:** fresh `git clone` of the latest `facescape-c` commit;
- **middle:** progress lines every ~20 s and one line per job;
- **last cell:** a zip of the results only.

All are pushed with `scripts/kaggle_push.py`.

| # | Notebook | Where | Account | Produces |
|---|---|---|---|---|
| 30 | `30_dataset_inspection` | local | — | every data check, PASS/FAIL table |
| 31 | `31_degradation_profiles` | local | — | profile gallery, input-quality tables, restorer-training alignment |
| K00 | `K00_readiness` | Kaggle | A and B | builds/wheels OK, dataset found, one smoke fit per model per GPU |
| K10 | `K10_mesh_realdenseface` | Kaggle | A | RealDenseFace S/B (MICA-init + zero-init) on pilot, then E1; priors + block-1 tokens saved |
| K11 | `K11_mesh_baselines` | Kaggle | B | DECA, MICA, EMOCA v2, SMIRK, HRN on pilot, then E1 |
| K12 | `K12_appearance` | Kaggle | B | DECA, TRUST, OpenDelight, HRN texture, raw projection |
| K20 | `K20_restorers` | Kaggle | B | Real-ESRGAN, GFPGAN, CodeFormer, DiffBIR, DA-CLIP outputs + reruns |
| K30 | `K30_gard_face` | Kaggle | A | token extraction (train pool), training, eval, ablations |
| K31 | `K31_prior_corrector` | Kaggle | A | prior corrector v2 train + eval |
| 32 | `32_mesh_benchmark` | local | — | E0 + E1 tables, mCE, crossover (H1), heatmaps |
| 33 | `33_appearance_benchmark` | local | — | appearance tables and galleries |
| 34 | `34_wrinkles_and_eyes` | local | — | stretch goal |
| 35 | `35_restoration_controls` | local | — | restorer effect on mesh and appearance |
| 36 | `36_gard_face` | local | — | method, curves, before/after, gap recovered |
| 37 | `37_prior_corrector` | local | — | v1 vs v2 |
| 38 | `38_corrections_all` | local | — | every correction side by side |
| 39 | `39_final_showcase` | local | — | faculty-facing story |

## Checklist

### A. Data
- [x] TU-Model 301–400 (15.2 GB) present; 321–359 have 20 meshes each; 323 invalid
- [x] Photos 341–359 (`fsmview_trainset_images_341-359.zip`, 28.0 GB) present and readable
- [x] Scans 321–340 (`fsmview_trainset_shape_321-340.zip`, 11.4 GB) present
- [ ] **Photos 321–340** (`fsmview_trainset_images_321-340.zip`) — test subjects 340, 325, 339 depend on it
- [ ] **Scans 341–359** (`fsmview_trainset_shape_341-359.zip`) — exact camera-frame GT for 344, 350, 352
- [x] Overlay check: TU mesh lands on all three views (344, 20 expressions)
- [x] Measured toolkit misalignment: 1.6–2.9 mm / 0.5–2°; after ICP to scan 0.23–0.24 mm median
- [x] Clean stage for 341–359 (1,140 crops, 0 issues); degraded stage running
- [ ] Rebuild with all 38 people (clean → refine → degrade), rerun notebook 30, all PASS
- [ ] Upload private Kaggle dataset `satyapaladugu/facescape-c`; add account B as collaborator (website)

### B. Environment and Kaggle
- [x] Linux venv `.venv-linux` (numpy, cv2, trimesh, scipy, pytest, jupyter)
- [x] `kaggle-a` / `kaggle-b` work (both list datasets)
- [x] `notebook_kit` (clone cell, run-info cell, results-only zip cell, pinned metadata) + `scripts/kaggle_push.py`
- [ ] K00 readiness on A and B: wheels installed from cache, dataset mounted, smoke fit per model per GPU
- [ ] EMOCA v2, HRN (code + assets incl. BFM), SMIRK, restorer weights uploaded to a private weights dataset

### C. Code modules (each ships with tests; Haiku runs them on every change)
- [x] `src/facescape/*` + `tests/test_facescape.py` (6 tests)
- [x] `src/robustness/profiles.py` + `tests/test_profiles.py` (72 tests)
- [ ] `src/models/` adapters: realdenseface (MICA init + zero init), deca, mica, emoca, smirk, hrn — common `MeshResult` (vertices mm in camera frame, faces, landmarks, timing)
- [ ] persistent worker mode (load once, fit many) in `runner`
- [ ] save RealDenseFace priors + block-1 tokens per fit
- [ ] `src/metrics/mesh.py`: NoW-style landmark+ICP alignment, scan-to-mesh, Chamfer, NC/MNE, R2.5, F-score, CR, regions, mean-face baseline
- [ ] `src/metrics/robustness.py`: Δ, CE/mCE, relative mCE, crossover, gap recovered (paired), Wilcoxon, bootstrap CI
- [ ] `src/restorers/`: realesrgan, gfpgan, codeformer, diffbir, daclip
- [ ] `src/correction/gard_face.py`: DiT denoiser, interpolated flow matching, prior-consistency loss, 1–4 step sampler
- [ ] prior corrector v2 (gap-table pairing fix A5, degradation-aware gate)
- [ ] collector → one tidy table per benchmark

### D. Pilot checks (before any large GPU spend)
- [ ] every model fits all 60 clean pilot photos (344 × 20 × 3 views); success rate printed
- [ ] predicted meshes overlaid on photos look right (clay renders)
- [ ] clean errors in a sane range (≈1–3 mm after ICP) and ordered roughly as the papers report
- [ ] one degraded condition per profile scored end to end
- [ ] **Sun 29, 14:00 checkpoint:** go on FaceScape, or fall back to NoW

### E. Benchmark
- [ ] E0 pilot: 344 × 20 expressions × 3 views × 43 conditions × 6 models
- [ ] E1: 6 test people × 4 expressions × 3 views × 43 conditions × 6 models
- [ ] appearance on clean + S2
- [ ] restorers on the subset, then mesh/appearance reruns
- [ ] oracle (clean tokens / clean priors) on RealDenseFace

### F. Training (only after the E2 gate is written down)
- [ ] GARD-Face: train-pool token pairs (all 3 views mixed), training curves, 1 vs 4 steps, ±prior loss, position ablation, clean-photo harm
- [ ] prior corrector v2

### G. Reporting
- [ ] notebooks 31–39 executed with every metric visible
- [ ] Journal.md updated after each experiment (numbers cite the notebook that printed them)
