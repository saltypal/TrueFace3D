# 31 — NPHM: a neural parametric head instead of a fixed mesh basis

**CVPR 2023; arXiv 2212.02761.** Snapshot: 2026-10-10. [PDF](../pdfs/31_NPHM.pdf), [paper](https://arxiv.org/abs/2212.02761). Dataset/registration: PDF pp. 3–4; model/training: pp. 5–6. For image inference extensions, read [MonoNPHM](39_MonoNPHM.md) and [Pix2NPHM](44_Pix2NPHM.md).

## Intuition

FLAME moves a fixed set of connected vertices. NPHM learns a function that tells you whether any queried point is inside, outside, or on a head. The surface is the set of points where that signed distance is zero. This can represent geometry beyond a fixed face topology, including parts of hair.

Instead of asking one network to remember the whole head equally, NPHM uses small local experts around landmarks. Nearby experts blend smoothly. An expression network then deforms a canonical head into a smiling or talking head.

## Architecture and execution

```text
global identity latent -> anchor-position MLP -> facial anchors
point x - local anchor + global/local identity codes -> local SDF MLPs
 -> Gaussian-weighted blend + global background field -> canonical SDF
identity bottleneck + expression latent + canonical point
 -> forward deformation MLP -> expressed position
scans/registered correspondences -> fit latent codes and network parameters
```

Pairs of left/right experts share weights after mirroring coordinates; their latent codes can still express identity variation. One global field handles areas far from anchors. The global identity code predicts anchors, avoiding an assumption that all people have identical landmark positions. Expressions affect multiple regions together, so their deformation field uses global conditioning plus a compressed identity representation.

This is an **auto-decoder**: training jointly learns a code for each subject and the decoding networks. The original paper does not contain a single-RGB encoder. At evaluation, latent codes are optimized to observed depth points.

## Mathematics and worked example

\[
F_{id}(x)=\sum_{k=0}^{K}w_k(x,a_k)f_k(x,z_{glob},z_k),
\qquad\sum_kw_k=1.
\]

Illustrative example: at a cheek point, two experts predict signed distances +0.4 mm and −0.2 mm with weights 0.75 and 0.25. Their blend is +0.25 mm, so the point is outside the predicted surface. Meshing requires sampling and finding zero crossings; the distances are not vertex coordinates.

Surface loss encourages zero distance on scan points; Eikonal regularization encourages \(\|\nabla F\|\approx1\); normals supervise gradient direction. Anchor and latent regularization stabilize the representation. Expression loss supervises displacements from registered correspondences.

## Dataset and experiments

The original dataset has **255 subjects / over 5,200 scans**, with 188 men and 67 women and up to 23 expressions. Two Artec Eva scanners rotate around the head; a capture takes about six seconds. Scans have roughly 1.5 million vertices. An open-mouth expression is canonical to reduce mouth topology problems.

Registration first jointly fits FLAME identity/expression/pose and scale, then refines vertex offsets. Thus even this implicit model needs careful correspondence construction. The paper's reported experiment trains on **87 identities** and evaluates **18 other identities**, using frontal rendered depth samples, typically 5,000 points. The full captured count is not the method's training count. Later dataset releases expand the population; MonoNPHM explicitly uses a 473-identity version.

## Evidence and limitations

Comparisons evaluate identity/expression fitting, Chamfer distance, normal consistency, and F-score at 1.5 mm. Better fitting to depth is not proof of better inference from a lone RGB photograph. Fine scan detail may be missing or inconsistent around hair and ears. Latent fitting and implicit meshing add runtime costs and do not guarantee FLAME-compatible exports.

## TrueFace3D mapping and improvements

Use authorized scans as geometry/detail labels after coordinate alignment. RealDenseFace already uses NPHM-derived data, so test identities must be audited against pretrained exposure. A new implicit backbone is a separately scoped experiment; our baseline remains FLAME.

1. **Hypothesis:** local-confidence detail supervision yields better nose/cheek detail than a global uniform loss. Test region errors at equal training budget.
2. **Hypothesis:** a neural head teacher can supervise a lightweight FLAME-plus-displacement student. Measure representational loss separately from image prediction loss; count teacher supervision honestly.

**Explain it back:** why is an auto-decoder not an image encoder? What advantage and export complication does an SDF introduce?
