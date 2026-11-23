# 03 — MICA — neutral identity in metric space

**Towards Metrical Reconstruction of Human Faces; Zielonka, Bolkart and Thies, ECCV 2022, arXiv 2204.06607.** Snapshot: 2026-10-10. [Original PDF](../pdfs/03_MICA.pdf), [primary source](https://arxiv.org/abs/2204.06607). Read: §3 and Eq. 1; §5 and Table 1; supplemental dataset and encoder ablations.

## Intuition and prerequisites

Recognition networks learn which faces belong to the same person despite changing smiles and lighting. MICA reuses that identity signal to estimate neutral geometry. It aims to predict actual facial dimensions rather than allowing benchmark alignment to rescale a bad estimate into a seemingly good one.

## Architecture and execution

1. Align the input face for an ArcFace identity encoder pretrained on Glint360K.
2. Map the recognition embedding through three hidden fully connected ReLU layers and a final linear layer to a 300-dimensional shape code.
3. Decode that code with a neutral FLAME geometry model with 5,023 vertices.
4. Train against registered metric scans using region-weighted L1 geometry error.
5. Freeze most ArcFace layers, fine-tune its last three ResNet blocks and select using validation loss.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

Neutral vertices are \(V=\bar V+Bz\), \(z\in\mathbb R^{300}\). Supervision is a region-weighted absolute error \(\sum\|\kappa\odot(V-V_{GT})\|_1\). If both face and camera distance are scaled together, a perspective image can stay nearly unchanged; metric supervision resolves this statistically through learned priors, not through an extra measurement at inference.

Illustrative: an image of a 140mm-wide head far from the camera can resemble a 120mm-wide head closer to it. A similarity-aligned benchmark may scale the prediction and hide the size error. Report rigid-aligned metric error separately from similarity-aligned shape error.

## Training data and supervision

Table 1 unifies 2,315 subjects from Stirling, D3DFACS, Florence 2D/3D, BU-3DFE, LYHM, FaceWarehouse, FRGC and BP4D+. Its counts are 133, 10, 53, 100, 1,211, 150, 531 and 127 respectively. These are unified source counts, not all necessarily training-only counts; Florence is used for validation. Initial landmark/ICP fitting and nonrigid refinement establish common FLAME topology. The paper rejects its analyzed FaceScape data for this metric-training purpose because of scale normalization. ArcFace's large pretraining corpus is a separate data dependency.

## Evidence and limits

The paper compares metric and conventional NoW evaluation and studies dataset choices and identity-encoder fine-tuning. That is a useful precedent for our data-novelty question: topology and unit unification plus an appropriate objective can matter more than merely adding samples.

Neutral identity is the intended output. It does not reconstruct the current expression, full PBR materials or accurate ears/eyes; the supervised loss weights those regions weakly. A recognition embedding also inherits coverage biases and cannot make invisible anatomy observable.

## TrueFace3D use and code reading

Use MICA as an identity initialization or comparator, especially because RealDenseFace's paper uses it. Inspect the actual released optimizer before assuming MICA is active. Official project: https://zielon.github.io/mica/ and https://github.com/Zielon/MICA. Preserve FLAME version and neutral-space alignment.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Test a metric calibration adapter on genuinely calibrated scan data, reporting rigid and similarity alignment separately.
- Balance source identities and evaluate unseen demographics/datasets at a fixed total number of subjects; compare to simply concatenating all sources.

## Explain it back

Why does better face recognition help shape prediction? Which error can similarity alignment hide? Why should a neutral-shape prior not override a smiling expression?

