# 22 — Parametric eyes — fit a useful eye prior to sparse evidence

**Lightweight Eye Capture Using a Parametric Model; Bérard et al., SIGGRAPH 2016.** Snapshot: 2026-10-10. [Original PDF](../pdfs/22_ParametricEyes.pdf), [primary source](https://cgl.ethz.ch/publications/papers/paperPas16a.php). Read: §§4–8: database, eyeball model, iris synthesis, veins and single-image fitting.

## Intuition

Rather than scan every microscopic eye feature from scratch, this method learns a compact eye shape model from captured eyes. It matches the visible coarse structure and fills fine detail using compatible examples. It supports single-image fitting, but some fine structures are synthesized rather than individually measured.

## Architecture and execution

1. Align a database of thirty captured eyes and construct a PCA eyeball shape model, including limbus-related structure.
2. Fit rigid pose, scale and shape coefficients to geometric/image constraints.
3. Extract a low-resolution iris control map after accounting for eye geometry.
4. Synthesize matching high-frequency iris color and coupled geometry in a polar domain, including pupil-dilation trajectories.
5. Generate/fit scleral veins and combine the surfaces into a parameterized renderable eye.

## Mathematics and worked example

Its model fit balances shape error, limbus error and a normalized coefficient prior. PCA penalties \(\sum_i((c_i-\mu_i)/\sigma_i)^2\) discourage implausible shapes. Iris geometry synthesis stores differential angular/radial/height coordinates and reconstructs them using a constrained Laplacian system; this avoids copying absolute geometry at the wrong iris radius.

Illustrative: a low-resolution blue iris determines overall color and pupil radius, while exemplar patches add fine radial structure. Those patches improve realism but are not evidence of the person's exact iris fingerprint. Enlarging the pupil also changes iris geometry; merely shrinking a texture circle misses that deformation.

## Data and supervision

Thirty high-quality eyes captured in the preceding work provide aligned geometry, sclera/iris texture and dilation examples. Eyes are mirrored under an assumed left/right symmetry. This is small measured-data modeling plus optimization/procedural synthesis, not web-scale neural training. The paper gives multiview scanner and single uncalibrated image scenarios separately.

## Evidence and limitations

The paper demonstrates plausible fitted eyes under varying size, iris color and vein structure. Its detailed synthesis and optical modeling explain quality, but a convincing fitted image is not a complete validation of hidden eye anatomy. Fine iris detail may be transferred from another captured eye.

Database access and executable implementation were not verified. Low-resolution portraits cannot identify all optical and geometric variables. Bilateral symmetry is an assumption and can suppress real differences. Some texture/detail is explicitly procedural or exemplar-based.

## Integration and implementation reading

Best foundational guide for our first parametric eye component. Implement a simpler explicit asset first: eye centers/rotation from FLAME, iris/pupil controls, corneal shell and shared PBR lighting. Preserve high-confidence observed iris color; mark procedural veins/detail as synthetic. Compare with the paper before claiming any eye-specific novelty.

## Improvements to test

- Build a visible-evidence mask that limits fitting when eyelids/glasses hide iris boundaries.
- Jointly fit eye/lid alignment with the head camera while constraining interpenetration and preserving identity shape.
- Compare fitting quality across input eye resolutions, not just face image resolution.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

Why use a coefficient prior? How does a polar iris representation help dilation? Which eye details are preserved and which are invented?

