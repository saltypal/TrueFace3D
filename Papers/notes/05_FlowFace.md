# 05 — FlowFace — refine dense alignment iteratively

**3D Face Tracking from 2D Video through Iterative Dense UV to Image Flow; Taubner et al., CVPR 2024, arXiv 2404.09819.** Snapshot: 2026-10-10. [Original PDF](../pdfs/05_FlowFace.pdf), [primary source](https://arxiv.org/abs/2404.09819). Read: §3; training on pp. 5–6; Appendix B–D; Tables 4–5.

## Intuition and prerequisites

FlowFace repeatedly adjusts guesses about where face vertices lie in an image. Like aligning a transparent face diagram over a photo, it compares features and nudges the diagram until matches improve. Dense alignment avoids relying only on a few sparse landmarks.

## Architecture and execution

1. Extract image features with an ImageNet-pretrained SegFormer encoder.
2. Generate learnable UV feature pyramids for correlation, context and recurrent hidden state.
3. Use a RAFT-style correlation/update module to iteratively refine UV-to-image flow and uncertainty.
4. Convert dense predictions into per-vertex 2D targets.
5. Fit FLAME using alignment, neutral-shape and temporal constraints; optional per-vertex deformation is used in multiview settings.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

A recurrent update is schematically \(f_{k+1}=f_k+\Delta f_k\). Feature correlations indicate plausible matches; a learned update handles ambiguity. Fitting minimizes weighted differences between projected vertices and predicted targets. FlowFace's image-space motion metric compares movement across time horizons, rather than only static vertex errors.

Illustrative: if the nose-tip target is initially ten pixels left of its true location, successive updates may reduce that error. A stable predicted target does not guarantee correct depth: the separate FLAME/identity prior supplies 3D structure. This explains the later RealDenseFace addition of relative depth.

## Training data and supervision

The alignment model trains on FaceScape, Stirling and FaMoS; its appendix describes 360 FaceScape subjects, 140 Stirling subjects and 95 FaMoS subjects before task splitting/selection. It transfers scans into FLAME and projects calibrated targets; Stirling is augmented with synthetic views. SegFormer-b5, image feature dimension 512, UV dimension 64 and three iterations are reported. Training is six epochs. Multiface is the motion evaluation source, not part of this stated alignment training mixture.

## Evidence and limits

Ablations test the backbone, recurrence and MICA fitting prior. Appendix evaluation uses 86 sequences from ten Multiface subjects. Separate single-image and complete-sequence protocols exist; complete-sequence performance uses information unavailable to an isolated photograph.

This two-stage pipeline is not fully differentiable end-to-end. Studio training data limits coverage. Its optional deformation is not an unrestricted monocular detail estimator. Public code availability should not be assumed: RealDenseFace's supplement reports FlowFace unavailable for its reproduced comparisons.

## TrueFace3D use and code reading

Use as methodological reference and benchmark comparator where runnable code becomes available. Retain its insight that dense alignment confidence and identity initialization matter. Do not hold our first milestone waiting for unreleased artifacts.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Compare recurrent refinement against a single decoder under equal latency and model capacity.
- Later evaluate short-video consistency using separate online and offline protocols; never label future-frame fitting causal.

## Explain it back

Why can motion accuracy differ from static mesh accuracy? What does recurrence add? What evidence is exclusive to multiview or full-sequence operation?

