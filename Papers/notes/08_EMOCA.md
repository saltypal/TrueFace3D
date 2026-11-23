# 08 — EMOCA — expressions that preserve perceived emotion

**EMOCA: Emotion Driven Monocular Face Capture and Animation; Danecek, Black and Bolkart, CVPR 2022, arXiv 2204.11312.** Snapshot: 2026-10-10. [Original PDF](../pdfs/08_EMOCA.pdf), [primary source](https://arxiv.org/abs/2204.11312). Read: Figure 2; §§4–6; emotion-network and dataset ablations.

## Intuition and prerequisites

A small change in lip corners or eyelids can make a reconstructed face look sad instead of angry while changing overall mesh distance very little. EMOCA uses an emotion network to supply a supervision signal sensitive to those important expression differences.

## Architecture and execution

1. Initialize a fixed coarse identity/appearance encoder from DECA.
2. Train a separate expression encoder to predict expressive FLAME parameters.
3. Render the textured mesh with predicted camera and spherical-harmonics light.
4. Pass original and rendered faces through a fixed emotion network and compare feature vectors.
5. Freeze expression estimation and train expression-conditioned detail using DECA-style learning.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

Emotion consistency is schematically \(\mathcal L_{emo}=\|E_{emo}(I)-E_{emo}(\hat I)\|\). It complements photometric, landmark and parameter regularization terms. It is a learned semantic constraint, not a proof that the underlying facial muscles or true depth were recovered.

Illustrative: two meshes can have nearly identical average vertex distance, but only one has raised inner eyebrows and downturned lips. An emotion-sensitive feature can distinguish them. Conversely, a network might learn to exaggerate those cues without matching the person's exact expression.

## Training data and supervision

The coarse EMOCA stage trains on AffectNet for up to twenty epochs with early stopping. The detail stage follows DECA settings; this paper explicitly identifies VGGFace2 and VoxCeleb2 as its DECA-style data sources. DECA's own original corpus additionally includes BUPT-Balancedface, so do not conflate the two lists. The chosen emotion network uses a ResNet-50 architecture after examining alternatives.

## Evidence and limits

Emotion and expression recognition, valence/arousal metrics and ablations assess the contribution of the emotion loss versus dataset changes. The paper finds emotion supervision important; changing to AffectNet alone does not explain all improvement. That is a useful model for separating training novelty from an objective change.

Emotion recognition has domain and population biases. Its features can reward exaggeration, and albedo/lighting can affect features. Jaw optimization is discussed as unstable. Improved emotion scores do not establish correct PBR reflectance or physically exact expression geometry.

## TrueFace3D use and code reading

A FLAME expression baseline or auxiliary expression constraint. Official implementation: https://github.com/radekd91/emoca. Never fuse complete EMOCA and RealDense meshes by naïve vertex averaging; compare or selectively constrain compatible parameter groups after checking model versions.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Evaluate whether emotion loss helps local eye/lip geometry while preserving identity under a fixed scan-based constraint.
- Calibrate emotion supervision by confidence and compare perceived expression, scan error and landmark alignment together.

## Explain it back

Why can global mesh error miss a wrong emotion? Which encoder is frozen? How could the emotion loss create an exaggerated face?

