# 21 — Eye capture — model the sclera, cornea and iris separately

**High-Quality Capture of Eyes; Bérard et al., SIGGRAPH Asia 2014.** Snapshot: 2026-10-10. [Original PDF](../pdfs/21_EyeCapture.pdf), [primary source](https://cgl.ethz.ch/publications/papers/paperPas14a.php). Read: §§3–7: acquisition and separate sclera/cornea/iris reconstruction; results.

## Intuition

The eye is not a white sphere with a flat colored sticker. The white sclera is mostly opaque, the cornea is transparent and refractive, and the iris lies behind it. Each interacts with light differently. A separate eye module has a physical justification, but it must share the face camera and lighting.

## Architecture and execution

1. Use a calibrated specialized capture setup with views, lighting and gaze/dilation variation.
2. Recover visible sclera geometry from image evidence appropriate to an opaque surface.
3. Estimate corneal shape using reflection information rather than pretending it is diffuse skin.
4. Recover iris shape/texture while accounting for refraction through the cornea.
5. Track iris deformation across pupil dilation and assemble the eye's separate surfaces/materials.

## Mathematics and worked example

Refraction follows Snell's law \(n_1\sin\theta_1=n_2\sin\theta_2\). The visible iris is observed through that refracting boundary; directly projecting RGB onto a flat iris ignores this change in rays. Distinct reflection/refraction models are part of the reconstruction, not post-render decoration.

Illustrative: a camera sees a vein on the sclera by reflected surface light but sees an iris feature along a bent ray through the cornea. A single depth-map estimator assuming one opaque surface cannot correctly interpret both observations.

## Data and supervision

This is a capture/optimization paper, not a pretrained single-photo neural predictor. It acquires real calibrated eye observations and validates reconstructed components from those captures. The later parametric-eye paper builds on thirty high-quality eyes from this work. A public executable capture system, complete database or pretrained weights were not verified here.

## Evidence and limitations

Results examine individual eye shape and iris deformation rather than assigning every person one generic sphere. The paper shows that local anatomy and nonrigid pupil response matter. It does not demonstrate unconstrained full fidelity from an ordinary distant portrait.

Specialized acquisition and manual/calibration work exceed our first monocular setup. Transparent surfaces and tiny image regions are underconstrained. Unseen veins and iris depth must be treated as priors in our single-photo implementation.

## Integration and implementation reading

Use as optical/geometry reference. Start with explicit eyeball, cornea and iris assets attached to FLAME eye joints. Fit visible iris size/color/gaze when supported; record generic corneal dimensions as defaults. Read original diagrams and ray/reflection assumptions before authoring eye materials.

## Improvements to test

- Fit only observable eye variables and return uncertainty for hidden/refractive geometry.
- Test whether separate refractive geometry improves held-out gaze/camera renders compared with a flat-eye texture baseline.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

Why does the eye need different surfaces? How does refraction change the correspondence between a photograph and iris texture? Which quantities are measured versus assumed?

