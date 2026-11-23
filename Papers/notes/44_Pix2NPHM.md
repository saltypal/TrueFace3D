# 44 — Pix2NPHM — regress a richer parametric head

**Pix2NPHM: Learning to Regress NPHM Reconstructions From a Single Image; Giebenhain et al., arXiv 2512.17773.** Snapshot: 2026-10-10. [Original PDF](../pdfs/44_Pix2NPHM.pdf), [primary source](https://arxiv.org/abs/2512.17773). Read: §3, Table 1, §4.1 and inference refinement ablations.

## Intuition and prerequisites

FLAME restricts geometry to a linear articulated mesh family. NPHM describes geometry with neural fields, allowing richer detail. Pix2NPHM learns to predict that field's compact identity/expression controls from one image, then optionally refines them using geometric priors.

## Architecture and execution

1. Predict image-space normals and canonical-point cues with pretrained domain-specific vision transformers.
2. Fuse features using transformer layers and regress NPHM identity/expression parameters.
3. Decode an implicit signed-distance surface through the NPHM model.
4. Train with 3D SDF supervision for registered scans and normal-based rendering supervision for unpaired video images.
5. Optionally optimize predicted latent parameters against normal and canonical-point evidence at inference.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

An SDF \(f(x;z_{id},z_{exp})\) defines the surface by \(f(x)=0\). Registered surface samples supervise \(|f(x)|\); latent regularization prevents extreme shapes. Unlike a fixed FLAME vertex mesh, extracting the implicit surface requires a meshing procedure and does not automatically inherit FLAME UVs.

Illustrative: a point outside the cheek should have positive signed distance; a point on its surface should have zero. Changing the latent identity code reshapes the field. The improved cheek detail cannot simply be copied into FLAME coefficients because the representations have different expressive spaces.

## Training data and supervision

Table 1 lists NPHM, FaceScape, DAViD, MimicMe, LYHM and 2D video sources. It reports about 102K registered 3D shapes and 1.13M total images, with about 250K video images. Dataset-specific counts in that table are training selections, not official release totals. Registration required roughly 2,500 GPU-hours. Pretraining the cue experts and building registrations are major costs.

## Evidence and limits

The paper reports single-image posed/neutral geometry and NoW comparison, plus feed-forward and optimized variants. Its feed-forward stage is reported at eight FPS on RTX 3080 10GB; one hundred refinement steps take about 85 seconds. The regressor trains on A100 80GB for four days; cue experts use two A6000 GPUs. These are author measurements, not our runtime.

Rich neural geometry requires different meshing, topology, parameter and material handling. More expressive fields can become harder to constrain on occluded surfaces. Reproducing all training data preprocessing is beyond a small free-GPU baseline.

## TrueFace3D use and code reading

Comparator for the possible geometric ceiling of FLAME and a later upgrade investigation. Keep the first system FLAME-based; don't combine incompatible latent vectors. Audit project-linked code/weights before execution.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Distill local high-confidence field geometry into bounded FLAME displacement and evaluate on independent scans.
- Compare latent refinement with confidence weighting under fixed inference time rather than granting one method vastly more optimization.

## Explain it back

What does a zero level set mean? Why is an implicit parametric model still parametric? Why does a richer model create an integration cost?

