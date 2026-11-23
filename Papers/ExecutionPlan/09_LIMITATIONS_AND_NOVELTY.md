# Limitations, hypotheses and deferred training

## Three different claims

An author-stated limitation needs an original page/section. An integration limitation is our inference about using that method in this stack. A measured project failure requires a saved run and evaluation. Keep all three fields in the limitation register; neither source reading nor a plausible explanation is a measured failure.

Combining methods is useful engineering. Fixing a known failure can become a research contribution only if the mechanism differs meaningfully from prior work, the evidence supports it, and the baseline/input/data/compute comparison is fair. A shared limitation does not imply we can solve it or that its solution is new. We need not claim novelty in every component.

## Prioritized no-retraining experiments

| Hypothesis | Source limitation / prior-art boundary | Baseline and proposed test | Evidence needed / reason to defer |
|---|---|---|---|
| H1: visibility and conservative reliability improve UV appearance without corrupting observed identity marks | DECA p9 §7 discusses albedo/hair/occlusion; OpenDelight p13 §7.4 discusses identity loss. Geometry-aware projection, visibility and albedo priors already exist | Naive UV projection; z-buffer + external-occlusion mask; angle/visibility weighting; optional restrained visible-detail residual. Keep geometry, predictor and light fixed | Valid masks, mark/relighting scores and color/coverage; do not claim calibrated uncertainty from heuristic weights |
| H2: bounded scheduling limits harmful detail/appearance changes compared with a matched joint fit | Pixel3DMM p7 identity/expression ambiguity; SFDM, H3D-Net and monocular appearance capture already stage/couple geometry and appearance | Fixed scene; light-only; equally bounded joint lighting/detail; staged fit with the same variables/loss/cap. Use the exact H2 card | Paired scan/held-out-view evidence, artifact changes and cost. Coarse identity frozen by construction is not evidence of improved identity recovery |
| H3: conservative detail selection reduces false wrinkles from shading/hair while retaining useful structure | DECA p9 §7 low-resolution/noisy detail and hair confusion; HiFace p6 Table1 warns detailed shape may worsen geometry | Coarse; native detail; bounded displacement; mask/amplitude gating. Preserve coarse geometry and one detail owner | Registered detail/normal evidence or limited synthetic recovery; no real pore-accuracy claim from shading alone |
| H4: explicit eye optics plus evidence-limited fitting improves coherent view/light response | Parametric eyes §§4–8 already fit a prior; specialized eye geometry/optics is established | Painted opaque eye; explicit default eye; bounded visible-cue fit, same scene/camera/light | Synthetic optics/cue recovery plus real-image diagnostics. Exact anatomical personal recovery and iris identity remain unvalidated |
| H5: audit and cache invalidation prevent stale-coordinate artifacts and improve reproducibility | A systems failure hypothesis, not a new inverse-rendering algorithm | Cached pipeline without a complete dependency key versus versioned input/model/UV/crop keys | Demonstrate intentionally changed crop/model/config invalidates affected stages. Engineering contribution unless broader novel evidence exists |

H1 and H2 are the primary candidate experiments in the fourteen-day window. H3/H4 are component validation opportunities, not mandatory independent paper contributions. The improvement policy can use pretrained predicted uncertainty as an input, but empirical calibration on development labels must be separately specified. A fitted scalar or heuristic is not a new trained confidence network.

## What existing work already covers

RealDenseFace already uses uncertainty-weighted geometric residuals. OpenDelight already uses source-aware mixed-data conditioning. ReEnFP already combines appearance/detail priors with adaptive fusion. SFDM already couples geometry and reflectance. MARCUS already uses geometry and joint intrinsic materials with UV shading constraints. MoSAR/Relightify/AvatarMe++ already produce coordinated materials. Monocular appearance capture already uses self-visibility. Merely renaming those mechanisms does not establish novelty.

A focused primary-source search on 2026-10-23 for face inverse rendering, uncertainty, confidence, visibility and albedo also surfaced WildCap among related work. That search was a discovery check, not a complete originality clearance. Before a final method claim, read and add any newly material prior art to the evidence register with its original and note, including general inverse-rendering work beyond this 44-paper corpus. The current proposed hypotheses remain **novelty unverified**.

## Training and ultimate backlog

| Proposed later work | Why training/access is needed | Inputs and resource dependency | Re-entry gate |
|---|---|---|---|
| Learned confidence / cross-component gate | A new predictor changes learned behavior; no pretrained weights for our new mechanism | Error/occlusion labels from authorized scans or controlled renders, held-out identities/lights; matched-budget baselines | H1/H2 establish a remaining measurable failure; tiny deterministic alternatives tested first |
| Improved dense-prior network | New head/layer/feature fusion usually invalidates pretrained interfaces | Correspondence/depth targets, registrations, upstream training pipeline and GPU budget | Core inference reproduced; training data/labels and objective independently available |
| Better albedo/skin materials | New domain tokens or material heads need training/adaptation | Measured/approximate targets tracked separately; permitted OLAT/scans/HDRIs; private-data gaps resolved | Existing predictor and renderer audited; raw/aligned score definitions frozen |
| New identity/expression detail decoder | Different conditioning/basis/strain model requires learned association | Registered scans/expressions/detail maps, own-basis validation, subject split | Detail transfer works and its failure is isolated from albedo/camera |
| Statistical learned eye prior | Exact selected paper's database/release is unverified | Acquired allowed eye geometry/optics data, alignment/dilation supervision | Procedural eye baseline works; acquisition and meaningful eye GT feasible |
| MARCUS fine-tuning/distillation | Its backbone/adapters and base material supervision cannot be recreated by downloading FFHQ alone | Actual compatible weights, topology, allowed material targets and measured memory; independent teacher-label bias tests | Native MARCUS inference and coherent-bundle comparison pass; resource allocation explicit |
| Missing-weight competitor reproduction | No checkpoint does not prove full trainability | Complete architecture/training code, target construction, legal data/model access and adequate compute | Reproduction feasibility audit, estimated budget and user priority justify it |
| Video consistency | Temporal data changes fitting/evaluation assumptions | Short sequence with shared identity/materials; causal vs offline policy | One-photo stack and reproducible evaluations complete |

These entries are holds, not scheduled jobs. No private scan source is assumed accessible. Re-estimate time/memory/data before activating one. The all-paper matrix provides additional baseline/improvement/ultimate suggestions for each source; it does not assert each suggestion is new or feasible to train immediately.

## Experiment card template

Before coding an improvement, record ID, task/failure, source/page versus integration inference, exact mechanism, nearest prior art, baseline/challenger versions, input privileges, fixed/optimized variables, train/no-train classification, dev/evaluation manifests, metric thresholds set before evaluation, cost budget, expected failure case, stop/rollback rule and reviewer. After execution, attach measured deltas/failures and accept/reject/defer with a concrete reason. Negative results remain in the report.
