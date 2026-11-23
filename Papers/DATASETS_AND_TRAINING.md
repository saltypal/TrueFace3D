# Datasets, supervision, splits, and training methodology

Snapshot: **2026-10-10**. Dataset counts are version-specific; current availability is not the same as automatic access or successful download. No face datasets were downloaded in this phase. Original research PDFs are downloaded separately under pdfs/.

## Does “one/few real photos plus 3D” exist?

**Yes.** NoW pairs real iPhone photos with an identity scan; FaceScape supplies calibrated real multi-view photos and reconstructed shapes; H3DS pairs real photos with scans; Multiface and Ava-256 supply real camera views with derived tracking/reconstruction assets. The crucial choice is what the predictor receives versus what training/evaluation can inspect.

A **render** is a synthetic image made from an asset. A **real photograph** is measured by a camera. A **scan** is measured/reconstructed geometry. A UV **capture texture** is observed surface color under lighting. A **diffuse-albedo label** attempts to remove illumination. They are different targets. Many datasets provide excellent geometry and ordinary texture but no measured roughness, specular intensity, SSS, or eye optics.

At strict monocular inference, the model gets one photo and explicitly available metadata. During training, other cameras/scans/lights may provide supervision. This is valid if disclosed. At evaluation, those withheld measurements score the output; they must not be fed back into reconstruction or used to choose its best result.

## Dataset decision table

| Resource | Real captured inputs and 3D evidence | Material/eye truth | Best role and caveat |
|---|---|---|---|
| [NoW / RingNet](notes/34_RingNet_NoW.md) | 2,054 phone photos / 100 identities, separate scans | No measured PBR/optical maps | External identity-shape evaluation; neutral scans vs expressive input |
| [FaceScape](notes/29_FaceScape.md) | Original 938-person study; current 847-person TU release; calibrated multi-view photos on smaller subset | Capture textures and displacement; ordinary textures retain light | One-photo geometry/detail training and reserved-view evaluation; privacy mosaics and model topology |
| [H3DS / H3D-Net](notes/33_H3DNet.md) | Original 23 subjects; later 59-subject v2, real photos/scans | No general measured material set | External geometry checks once authorized; paper model uses 3+ views and private 10K-person prior |
| [NPHM](notes/31_NPHM.md) | Original 255 people / 5,200 scans; later release versions expand IDs | Textured scans are not fully separated reflectance | Geometry/detail labels and synthetic calibrated renders; not automatically paired original camera photos |
| [NeRSemble](notes/30_NeRSemble.md) | Original 222 identities, 16-view video, 4,734 sequences | Fixed capture lighting; no full physical material labels | Held-out real-view assessment, selected supervision, later video; distinguish SVFR benchmark |
| [Multiface](notes/32_Multiface.md) | 13 identities, many calibrated cameras, tracked meshes/UV/head pose/audio | Derived textures under dome illumination | Renderer calibration and later temporal checks; small population, large data volume |
| [Ava-256](notes/35_CodecAvatarStudio_AVA256.md) | 256 people, dome + headset captures and derived geometry | Ordinary dome appearance, not OLAT materials | Geometry/correspondence training; already used by RealDenseFace |
| [Goliath-4](notes/35_CodecAvatarStudio_AVA256.md) | Four people, varying-light head/hand captures and paired phone modalities | Controlled light evidence, modality-specific assets | Material/light validation and richer supervision; too few IDs for broad generalization |
| [Becoming Lit capture](notes/42_BecomingLit.md) | Ten people, 16 cameras, 40 light configurations | OLAT evidence supports decomposition; not every intrinsic independently measured | Held-out lights/views, optional teachers; access needed |
| [FaceOLAT via OpenDelight](notes/12_OpenDelight.md) | 139 source subjects; 134 processed, 124/10 split, 40 views/331 lights | Uniform-lit approximate albedo has residual specular/misalignment | Delighting supervision with source-quality tags |
| [RealFaceDB via AvatarMe++](notes/16_AvatarMePlusPlus.md) | Over 200 captured people in source study; release is selected patches | Reflectance/normals from professional capture | Real material evidence; patches aren't all full identities/meshes |
| [FFHQ-UV-Intrinsics via MoSAR](notes/14_MoSAR.md) | FFHQ-derived processed UV assets | **Predicted** intrinsic labels | Pseudo-label training only; noncommercial/no-derivatives restrictions must be checked |
| [MARCUS derived data](notes/13_MARCUS.md) | FFHQ/CelebAMask-derived render/UV data with a private measured-scan base | Many generated labels, not independently measured materials | Heavy optional teacher/student experiment; topology and source provenance |
| Eye capture/parametric studies | Specialized multi-camera optical acquisition | Detailed source eye geometry/optics | Explain eye priors; no verified full easy-access model/data release for this project |
| [AudioEar3D](notes/24_AudioEar.md) | 112 ears / 56 people with structured-light RGB geometry | Ear geometry rather than full face materials | Optional ear-shape validation; not a population-scale face dataset |
| FFHQ, CelebA-HQ, VGGFace2 and related 2D corpora | Unconstrained photos; identity labels for some sources | No true paired scans/BRDF | Weak/pseudo supervision and restoration; access/license/identity leakage audits |

Use official access pages linked in each note. For FaceScape, some eye regions are mosaiced and the publishable portrait list restricts display. The current webpage's multi-view tuple arithmetic is inconsistent; count valid records. Benchmark scans may be withheld. Credentials and sensitive face data stay outside Git.

## What the surveyed methods actually train on

This table summarizes source choices; detailed protocols/limitations are in the linked notes. “Why” is an interpretation of the methodological need, not a claim that these are the only usable datasets.

| Paper(s) | Reported training supervision | Why it is used / consequence for our project |
|---|---|---|
| [RealDenseFace](notes/01_RealDenseFace.md) | Ava 349K + NPHM 384K + FaceScape 244K prepared examples | Dense UV/depth/vertex targets across real and rendered domains; changing sources is not automatically novel |
| [MICA](notes/03_MICA.md) | 2,315 unified identities from several scan sets; ArcFace/Glint360K prior | Metric identity geometry from real scans; scan fitting/neutralization choices affect labels |
| [Pixel3DMM](notes/04_Pixel3DMM.md) | Ava/NPHM/FaceScape, ~976K prepared samples | Pixel correspondences/normals; closely related geometry-pretraining exposure |
| [FlowFace](notes/05_FlowFace.md) | FaceScape, Stirling, FaMoS | Dense correspondence/flow supervision, not observed skin materials |
| [SMIRK](notes/06_SMIRK.md) | FFHQ/CelebA/LRS3/MEAD and fixed shape initialization | Self-supervised expression through image reconstruction; generative image renderer has different semantics from PBR |
| [DECA](notes/07_DECA.md) | VGGFace2, BUPT-BalancedFace, VoxCeleb2, ~2M selected images | Weak image/identity supervision and detail consistency; predicted detail is not scanned GT |
| [EMOCA](notes/08_EMOCA.md) | DECA-style geometry/detail data plus AffectNet emotion stage | Emotional supervision improves expression; does not itself solve skin physics |
| [HRN](notes/09_HRN.md) | FaceScape, ESRC FaceHD, lab/wild images; FFHQ texture cleanup | Hierarchical geometry/detail targets; some source labels/identity manifests require reproduction audit |
| [HiFace](notes/10_HiFace.md) | 200K synthetic, 400K real CelebA/VGGFace2 | Static/dynamic detail disentanglement and broad image variation |
| [MeshLAM](notes/11_MeshLAM.md) | VFHQ video, ~3M frames from 15K sequences | Animatable appearance from video; different training/inference/asset goal |
| [OpenDelight](notes/12_OpenDelight.md) | FaceOLAT + 120 purchased PBR scans, 473 HDRIs | Source-aware albedo prior; public new scan dataset is not evidence released weights were retrained |
| [MARCUS](notes/13_MARCUS.md) | <100 real PBR scans + FFHQ/CelebAMask-derived data, pretrained diffusion | Full geometry/intrinsic pipeline; private seed data and heavy training prevent a cheap-from-scratch claim |
| [MoSAR](notes/14_MoSAR.md) | 890 private captured subjects + FFHQ / FFHQ-UV-derived samples | Intrinsic geometry/material separation with private physical supervision |
| [Relightify](notes/15_Relightify.md) | 4,700 MimicMe textures and AvatarMe++ predicted PBR targets | Joint diffusion prior over appearance channels; targets include teacher bias |
| [AvatarMe++](notes/16_AvatarMePlusPlus.md) | Professional multi-expression reflectance scans, public patches | Detailed physical maps; full captured collection differs from distributable set |
| [ID2Reflectance](notes/17_ID2Reflectance.md) | 135 captured identities, 115/20 split + FFHQ/identity corpora | Domain codebooks and identity-aware material prediction |
| [ReflectanceMM](notes/18_ReflectanceMM.md) | 128 selected Multi-PIE IDs × 9 views/11 directional lights, FFHQ | Learn statistical diffuse/specular components; lighting/BRDF assumptions become label bias |
| [S3-Face](notes/19_S3Face.md) | 100 captured people, 90/10 split + purchased albedos | SSS-compliant appearance domains; private sources and diffusion training |
| [HAFMat](notes/20_HAFMat.md) | 147 OpenHumanBRDF subjects, 127/20 split | Full-body material reconstruction; body topology/model does not directly match facial FLAME |
| [DeNeRF](notes/23_EyeNeRF.md) | ETH-XGaze multi-camera/gaze samples | Animated eye appearance from rich per-person observations; not one-photo optics recovery |
| [AudioEar](notes/24_AudioEar.md) | 2,000 FFHQ ear crops/landmarks + 10K synthetic ear examples | Weak 2D and generated 3D evidence; dedicated ear topology |
| [FaceScape method](notes/29_FaceScape.md) | Registered scans/textures and expression deformation/displacement | Learn dynamic detail with true scan correspondence |
| [NeRSemble](notes/30_NeRSemble.md) | Per-sequence calibrated multi-view videos | Sequence radiance-field optimization, not ordinary feed-forward face material prediction |
| [NPHM](notes/31_NPHM.md) | Original 87-identity experiment from scans | Auto-decoder SDF/deformation supervision; no RGB encoder in original paper |
| [Multiface / Codec Avatar Studio](notes/35_CodecAvatarStudio_AVA256.md) | Real multi-camera tracking/texture assets | View-dependent avatar rendering/driveability; don't relabel primitive RGB as albedo |
| [H3D-Net](notes/33_H3DNet.md) | Internal 10,000-person scan prior | Public H3DS is evaluation, not those training scans |
| [RingNet](notes/34_RingNet_NoW.md) | ~800K cleaned VGGFace2 + ~3K extreme poses | Identity/landmarks provide weak geometry supervision without paired 3D |
| [CodeFormer / GFPGAN](notes/36_CodeFormer.md) | FFHQ70K with synthetic degradation; GAN/quantized priors | Restores plausible 2D face detail; no scan constraints by default |
| [Real-ESRGAN](notes/38_RealESRGAN.md) | DIV2K/Flickr2K/OutdoorSceneTraining | General-image repeated degradation, not face geometry supervision |
| [MonoNPHM](notes/39_MonoNPHM.md) | Textured NPHM v2, 473 identities | Joint geometry/appearance prior for video fitting |
| [SFDM](notes/40_SFDM.md) | FaceScape ten template IDs/20 refined IDs, multiple views | Attribute coupling using image evidence without measured BRDF labels |
| [Monocular appearance capture](notes/41_MonocularAppearance.md) | ~250 frames per captured subject | Per-person inverse optimization under fixed expression/lighting environment |
| [Becoming Lit](notes/42_BecomingLit.md) | Personalized multi-view OLAT sequences | Hybrid light transport fit from calibrated changing-light observations |
| [ReEnFP](notes/43_ReEnFP.md) | FFHQ/CelebA-HQ + FaceScape displacement | Separate learned texture/detail priors and adaptive feature fusion already exist |
| [Pix2NPHM](notes/44_Pix2NPHM.md) | NPHM/FaceScape/Ava prepared dense priors + neural head model | Recent alternative representation; inspect exact subset and privileges |

FLAME's statistical-model data and the eye parametric database are described in their notes. nvdiffrast, Mitsuba, Disney BRDF, and the foundational SSS paper are rendering/modeling methods, not face-predictor training papers. They should not be assigned an invented FFHQ training dataset.

## Minimum viable data strategy

**First reproduce pretrained inference.** No new large training corpus is needed to establish geometry exports and camera/UV correctness. Use a small permitted set of images plus valid independent scan/view evidence. Capture failures before optimizing.

For supervised development, prefer selected authorized FaceScape image/camera/scan tuples and selected scan-derived synthetic renders. Add a smaller external geometry domain such as H3DS/NoW validation within benchmark rules. For material evaluation seek controlled lighting or independently captured physical maps: selected OLAT/Goliath/allowed RealFaceDB resources. Ordinary camera textures remain observed-color labels only.

For the eye phase start with visible landmark/iris/gaze evidence and generic optics. Do not claim measured corneal or iris anatomy from a face-scan benchmark that omits them. Public eye optical ground truth is an access gap, not something a face dataset automatically supplies.

Use identity-level lightweight manifests rather than copying full releases. Include original source ID, session/camera/light/frame IDs, GT provenance, license/access status, crop/camera/calibration, topology/version, scale, privacy/validity flags, and hashes. An availability label needs distinct values: accessible public, application pending, gated authorized, private unavailable, or benchmark-only.

## Split and leakage rules

Split **identities first**, before augmentation, rendering, restoration, or generating pseudo-labels. Every derived record follows its source identity. If a person appears in more than one source, reconcile identity overlap using metadata where possible. Do not scrape identities or use biometric matching beyond what authorized dataset protocols permit. Unknown overlap remains an explicit limit.

Audit upstream pretraining: RealDenseFace and Pixel3DMM already use Ava/NPHM/FaceScape. Holding out their pretrained training identities from our small adapter does not make those people globally unseen. Label test settings as upstream-exposed, upstream-exposure unknown, or verified external/held-out. NoW hidden test access is evaluation only.

Reserve cameras, lighting configurations, and temporally separated segments independently according to the claim. Novel-view test is not novel-identity test; new light is not new expression. Keep an untouched external test set and one development split. Never choose restoration weight, diffusion seed, or optimization checkpoint using test GT.

## Proposed training experiments

Start with frozen geometry/material experts and train only a small confidence/adapter module, if the baseline demonstrates a real failure. Cache deterministic expert outputs with source hashes. This controls GPU cost and makes the contribution inspectable.

Use four supervision tiers: independently measured scans/materials, controlled-light photographic evidence, synthetic known-scene renders, and teacher-generated pseudo-labels. Record them separately. A proposed confidence-weighted loss is

\[
\mathcal L=\sum_{s\in sources}\alpha_s\frac{1}{N_s}
\sum_{i\in s}q_i\mathcal L_i.
\]

Here α controls source balance and q label quality; learning them is a hypothesis requiring calibration/ablation. Do not present this generic equation as new by itself. Targets with uncertain albedo/shading should not supervise all intrinsic channels equally.

For dataset novelty experiments control identity count, total examples, updates, image resolution, augmentation, initialization, and compute. Compare base source; added source with naive mixing; equal-size resampling; source/quality-balanced mixing; geometry/material-aware curriculum. Report results per domain and quality stratum. If improvement disappears after matching data/compute, the evidence supports scale rather than a novel methodology.

## Evaluation and reporting

| Goal | Evidence | Important limit |
|---|---|---|
| Identity geometry | Official scan error/protocol, regional error, normals | Alignment can remove scale/pose errors; disclose it |
| Detail | Scan displacement/normal agreement and silhouettes | Shader detail is not displaced geometry |
| Material | Known albedo/reflectance error, controlled unseen-light renders | RGB match alone does not identify unique intrinsic properties |
| Novel-view realism | Held-out real cameras, masked PSNR/SSIM/LPIPS | Fix crop/mask/exposure conventions; input view alone is inadequate |
| Eye quality | Iris/limbus/gaze error, intersection/occlusion checks | Generic optics vs measured personal anatomy must be separated |
| Identity appearance | Recognition similarity and human comparison | Recognition embeddings can miss subtle geometric changes |
| Reliability | Calibration, risk–coverage, region error vs confidence | Uncertainty magnitude is not calibrated by naming it confidence |
| Engineering | Full-pipeline latency, peak VRAM, failures, dependency versions | Paper GPU FPS is not our system performance |

Compute uncertainty/statistics at the identity level, not treat adjacent video frames as independent people. Use confidence intervals and report number of identities, not only pixels/frames. Keep measured, synthetic, pseudo-labeled, and unverified outcomes separate.

## Can this be dataset/training novelty?

More datasets alone usually produce an additional experiment. A carefully defined new benchmark/protocol with clean provenance, previously missing supervision, or a tested training mechanism can support novelty. Our best current hypothesis is **provenance- and geometry-confidence-aware training/coupling** across heterogeneous supervision, while retaining one-photo inference. Related source conditioning, fusion, and teacher/student methods already exist; [the novelty analysis](NOVELTY_ANALYSIS.md) defines the required comparisons.
