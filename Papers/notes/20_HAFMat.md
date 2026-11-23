# 20 — HAFMat — heterogeneous priors need adaptive fusion

**HAFMat: Hybrid Priors Guided Adaptive Fusion for Single-Image Human Material Estimation; Jiang et al., arXiv 2606.16323, CGI2026 manuscript.** Snapshot: 2026-10-10. [Original PDF](../pdfs/20_HAFMat.pdf), [primary source](https://arxiv.org/abs/2606.16323). Read: §3 guidance and MAFFM; loss Eqs. 9–12; §4 experiments.

## Intuition

Geometry normals, human body structure and pretrained material estimates provide different kinds of information. HAFMat argues that dumping them all into one feature stack is suboptimal. Fine texture cues and semantic cues should influence appropriate decoder stages.

## Architecture and execution

1. Obtain RGB features from a DINOv2 ViT-B/14 distilled backbone.
2. Construct SMPL-derived geometry guidance and human structure/DensePose-related cues.
3. Collect material priors from pretrained material estimators.
4. Encode guidance and fuse it with decoder features at multiple layers using MAFFM.
5. Decode normal, diffuse, roughness, specular and subsurface maps; train with direct material and relighting losses.

## Mathematics and worked example

The reported loss is \(\mathcal L=\sum_{k\in\{n,d,r,s,sss\}}\|M_k-\hat M_k\|_1+\mathcal L_{relit}\). Relighting averages L1/perceptual comparisons over 37 point lights. A teaching fusion form \(F'_j=F_j+\sum_k a_{j,k}G_{j,k}\) illustrates layer-dependent weighting; it is not a replacement for the paper's exact MAFFM equations.

Illustrative: a body-part cue explains which pixels are skin versus clothing, while a normal prior explains the orientation of a local fold. Giving both identical weights at every resolution wastes their different strengths. A wrong SMPL fit can nevertheless contaminate every material estimate.

## Training data and supervision

OpenHumanBRDF contains 147 human samples, with 127 training and twenty testing. It supplies direct PBR targets. Training uses four RTX 3090 24GB GPUs, batch size one, up to 100 epochs and about two days. The small dataset and prior-model training histories must be considered when discussing generalization.

## Evidence and limitations

Ablations test guidance choices and adaptive fusion. Material PSNR and relighting PSNR assess map accuracy and resulting images. Multi-light render loss is already existing work; our integration novelty cannot be merely adding such a loss.

This is full-body human material estimation using SMPL-derived guidance, not a FLAME-native facial eye/ear pipeline. Image-space material outputs do not solve invisible UV regions. Reported compute is beyond the assumed free-GPU training baseline. Public runnable code/weights were not established during this survey.

## Integration and code reading

Close conceptual related work for reliability fusion, with a representation/domain difference that must be explicit. Use the idea of complementary cues, but implement a small face-specific adapter with calibrated confidence rather than importing an unverified full-body model.

## Improvements to test

- Compare visibility/geometry-error-calibrated face fusion against generic learned adaptive weights.
- Ablate which cues help each region and material under lighting, occlusion and camera shift.
- Evaluate confidence calibration rather than treating attention coefficients as probabilities of correctness.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Why do different cues belong at different scales? Is an attention weight a calibrated confidence? Why is a full-body model not a direct facial module?

