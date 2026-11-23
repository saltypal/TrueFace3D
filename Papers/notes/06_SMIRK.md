# 06 — SMIRK — neural synthesis as geometry supervision

**3D Facial Expressions through Analysis-by-Neural-Synthesis; Retsinas et al., CVPR 2024, arXiv 2404.04104.** Snapshot: 2026-10-10. [Original PDF](../pdfs/06_SMIRK.pdf), [primary source](https://arxiv.org/abs/2404.04104). Read: §3 and Figure 2; §4.1 setup; augmentation and rendering ablations.

## Intuition and prerequisites

An overly simple face shader gives poor learning signals because real skin has shadows and complex appearance. SMIRK trains an image generator to turn estimated geometry plus sparse appearance samples into a believable reconstruction. Geometry receives supervision through the generator, whose design limits its ability to ignore the mesh.

## Architecture and execution

1. Encode identity, pose and expression separately; pretrain identity against MICA and pose/expression with landmarks.
2. Freeze identity and pose encoders during the core expression-learning stage.
3. Decode FLAME geometry, including eye-closure blendshapes, and render its geometric image.
4. Feed that render plus sparsely sampled/masked original RGB evidence to a UNet-like translator.
5. Compare synthesized and original images; add synthetic expression augmentation with known geometry and a reconstruction cycle.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

A teaching form is \(\hat I=G(R(FLAME(\beta,\psi,\theta)),S(I))\), where \(S\) supplies sparse image information. Losses on \(\hat I\) propagate through \(G\) and the renderer to expression parameters. If \(S(I)\) supplied the entire image without restriction, copying could minimize image loss while learning wrong geometry.

Illustrative: render two meshes differing only in mouth opening while retaining the same sparse skin colors. The correct geometry should help the translator reconstruct the original lip boundary. Expression augmentation tests whether synthetic geometry is recoverable instead of allowing the translator to bypass it.

## Training data and supervision

Core training uses FFHQ, CelebA, LRS3 and MEAD, sampling frames from the latter video datasets. AffectNet is used for emotion evaluation and associated evaluation classifiers; do not list every AffectNet appearance in the paper as the core reconstruction training source. Shape pretraining inherits MICA's data history.

## Evidence and limits

Evaluations include emotion recognition, reconstruction with a freshly trained translator and a perceptual study. These probe expression fidelity but do not directly measure every local 3D surface point. Results must be complemented by scan geometry in our project.

The learned translator produces images, not a physically defined albedo/roughness material bundle. It can approximate lighting and appearance without being freely relightable. It also adds training complexity and a risk that the generator explains away geometry errors.

## TrueFace3D use and code reading

Use for expression comparison and as the expression estimator in RealDenseFace's published preprocessing. Official code: https://github.com/georgeretsi/smirk. Keep its neural-synthesis branch separate from our final Cycles skin shader; a pretty translated view is not a new editable 3D asset.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Use a neural translator only as an auxiliary training loss, requiring independent scan/landmark constraints.
- Compare sparse sampling levels and held-out expressions to quantify when image generation begins bypassing geometry.

## Explain it back

Why can an image generator improve geometry training? How can it also hide geometry mistakes? What outputs are missing for a physical renderer?

