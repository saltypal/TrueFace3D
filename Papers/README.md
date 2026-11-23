# TrueFace3D research pack

**44 original PDFs and 44 individual teaching notes**, surveyed through **2026-10-10**, covering geometry, skin materials, surface detail, eyes/ears, rendering, datasets, integrated methods, video references, and optional restoration. All files belong to the TrueFace3D workspace.

Pack finalized on 2026-10-11 (Asia/Calcutta). The original-source snapshot remains dated 2026-10-10.

## Start here

- **[Current 14-day execution plan, comparisons and component designs](ExecutionPlan/README.md)** — 2026-10-23; pretrained first, model choices reopened for evidence-based comparison.
- **[Selected module stack and implementation gates](MODULE_SELECTIONS.md)** — decisions and current release audit, 2026-10-11.
- [Module chat scopes and fork starter messages](CHAT_HANDOFFS.md).
- [Every paper: note + PDF + source](PAPER_INDEX.md).
- [First-principles vocabulary](FIRST_PRINCIPLES.md).
- [Cross-paper literature survey and method choices](LITERATURE_SURVEY.md).
- [Datasets, author training sources, and our experimental protocol](DATASETS_AND_TRAINING.md).
- [Three novelty axes and the required comparisons](NOVELTY_ANALYSIS.md).
- [How modules enter one renderer: contracts and physics](SHARED_RENDERER_AND_CONTRACTS.md).
- **[Master plan for future Codex sessions](TRUEFACE3D_MASTER_PLAN.md)**.
- [Archive validation report](VALIDATION_REPORT.md).
- [Final availability updates: FLAME Open and current dataset links](evidence/RELEASE_AUDIT_2026-10-11.md).

## What a paper note contains

Each note explains the problem with a simple analogy, traces the actual architecture/data flow, gives equations and an illustrative worked example, identifies training data and why it matters, discusses evidence/limitations, maps the method into our project, and proposes improvements with comparisons. The core geometry, albedo, detail, and recent integrated-method notes have additional depth. Source sections/pages and code entry points are included where verified. Unavailable internals/assets remain explicitly unknown rather than invented.

Worked-example numbers illustrate the mechanism; they are not experimental measurements. Proposed improvements are hypotheses. Author results remain separate from our measured results. A paper's public PDF does not mean its weights/private training scans are available.

## Recommended reading order

1. [FLAME](notes/02_FLAME.md) and **[RealDenseFace](notes/01_RealDenseFace.md)**: what gets predicted, fitted, and exported.
2. [MICA](notes/03_MICA.md), [Pixel3DMM](notes/04_Pixel3DMM.md), [FlowFace](notes/05_FlowFace.md): geometry alternatives and prior direction.
3. **[OpenDelight](notes/12_OpenDelight.md)**, [ReflectanceMM](notes/18_ReflectanceMM.md), [AvatarMe++](notes/16_AvatarMePlusPlus.md): albedo versus complete skin reflectance.
4. **[DECA](notes/07_DECA.md)**, [HiFace](notes/10_HiFace.md), [FaceScape](notes/29_FaceScape.md): geometry detail and expression dependence.
5. [Eye capture](notes/21_EyeCapture.md), **[parametric eyes](notes/22_ParametricEyes.md)**: optics and component fitting.
6. [nvdiffrast](notes/25_Nvdiffrast.md), [Disney](notes/27_DisneyBRDF.md), [SSS](notes/28_SubsurfaceTransport.md), then the shared-renderer guide.
7. **[MARCUS](notes/13_MARCUS.md)**, [ReEnFP](notes/43_ReEnFP.md), [SFDM](notes/40_SFDM.md): closest integration ideas and novelty constraints.
8. Dataset notes and training/novelty guides, then the master-plan gates.
9. Optional restoration and later-video notes after the native single-photo baseline.

## Archive layout and reproducibility

```text
pdfs/         originals, one stable ID per document
notes/        one Markdown explanation per original
evidence/     source catalog, hashes/page text, alphaXiv/code audit snapshots
figures/      rendered original pages used for diagram inspection
tools/        download, evidence-reading, rendering, catalog/validation utilities
```

The old filename `23_EyeNeRF` is retained as a stable archive ID; its actual paper/method is **DeNeRF (2308.00773)**. FLAME's author download returned HTML, so its verified original PDF was retrieved from a mirror; the manifest records that fallback. Rejected alphaXiv title matches are retained only for audit. The current FaceScape and paper-era dataset counts differ; notes preserve that difference.

From the project root, rebuild the local index and validate originals/links:

```powershell
python -m pip install -r .\Papers\tools\requirements.txt
python .\Papers\tools\build_catalog.py
```

Read already archived source passages without new network calls:

```powershell
python .\Papers\tools\read_evidence.py 01_RealDenseFace --pages 3,4 --budget 9000
```

Reacquire a missing original from the catalog (network and source availability required):

```powershell
python .\Papers\tools\download_papers.py --ids 01_RealDenseFace
python .\Papers\tools\build_catalog.py
```

Existing PDFs are retained. The manifest pins the surveyed content with hashes;
if a source replaces an unversioned download, review the version change rather
than treating different bytes as the same snapshot. Large PDFs and raw extraction
files are kept locally and ignored by Git; notes, catalog, and tools remain source-controlled candidates.

No reconstruction model has been run, datasets/weights have not been acquired in this phase, and no new method is experimentally validated. The next milestone is the master plan's asset/dependency audit followed by unmodified RealDenseFace inference.
