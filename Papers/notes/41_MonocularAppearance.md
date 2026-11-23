# 41 — Monocular facial appearance capture with light visibility

**arXiv 2412.12765; ICCV 2025 version.** Snapshot: 2026-10-10. [PDF](../pdfs/41_MonocularAppearance.pdf), [paper](https://arxiv.org/abs/2412.12765). Method: PDF pp. 2–4; limitations: p. 8; supplemental capture/implementation: p. 12. Monocular here means one camera filming a sequence, not one image.

## Intuition

As you slowly turn your head, a shiny highlight moves while a skin mark stays on the skin. Those changing observations help separate material from lighting. Shadows matter too: if the nose blocks the light, a renderer that ignores that blockage may incorrectly paint a dark stripe into the cheek's albedo.

This work accounts for visibility during inverse rendering, so recovered materials behave more plausibly under a different light.

## Architecture and optimization

```text
neutral-expression head-rotation video + known focal length
 -> landmark/photometric 3DMM tracking -> initial mesh + per-frame pose
 -> optimize mesh, diffuse albedo, specular intensity, roughness, environment
 -> rasterized primary visibility
 -> ray-traced diffuse illumination + visibility-modulated specular split sum
 -> image/mask losses + geometry/light/material regularization
 -> explicit mesh + texture-map asset
```

The initial tracker uses a PCA model with 50 identity and 25 expression components, not simply a drop-in FLAME2023 decoder. Expression remains fixed through the sequence. Mesh gradient preconditioning stabilizes joint geometry/material optimization rather than letting texture updates explain the scene before geometry improves.

Diffuse reflectance is Lambertian; skin specular uses a Beckmann-based microfacet model. Environment lighting is represented as a cubemap. The ordinary split-sum approximation is efficient but misses self-shadowing. The paper adds view-dependent specular visibility and uses OptiX ray tracing with multiple importance sampling for diffuse illumination. This is more physical than a purely learned image renderer, but compute and graphics dependencies are substantial.

## Mathematics and example

The key physical relationship includes visibility:

\[
L_o(x,\omega_o)=\int_\Omega f_r(x,\omega_i,\omega_o)
L_i(\omega_i)V(x,\omega_i)(n\cdot\omega_i)\,d\omega_i.
\]

Illustrative example: illumination predicts a cheek intensity of 0.6, but the nose blocks half the relevant illumination and the observed intensity is 0.3. Without visibility, optimization can halve albedo to explain the image. With visibility, the original albedo remains plausible. Under a moved light, the wrong darkened albedo stays dark, exposing the error.

The actual visibility-modulated specular approximation is most suitable for low-roughness surfaces; diffuse uses ray tracing instead. Do not substitute a binary shadow map everywhere and claim it reproduces this method.

## Data, training, and evidence

The supplement records **Canon EOS 1200D video at 25 fps**, with 35/60 mm lenses and modest 20–30 degree rotations. It captures 500–800 frames per subject and selects about 250. Image and texture maps are solved at 1K; per-subject optimization uses 6,000 iterations and approximately two hours on an RTX3090. This is personalized optimization, not general image-encoder training on a huge dataset.

Evaluation compares reconstructed/relit images and geometry, including a synthetic setting with known shape/appearance. Captured environment/pose accuracy remains important. The retrieved documentation does not establish a complete downloadable identity-level capture manifest, so do not promise those captures as our available training dataset.

Limitations include wrong tracking poses, persistently shadowed regions, missing eye modeling in its template, and ambiguity between skin tone and illumination. A render matching one lighting condition does not uniquely identify material properties.

## TrueFace3D mapping and improvements

Use this as a strong later-video reference and a reason to include light visibility in shared rendering. Our first-photo baseline can use its physical ideas but lacks the sequence evidence that makes separation easier. Changing from Beckmann to GGX or FLAME requires a validated adapter.

1. **Hypothesis:** a modest visibility-aware refinement improves delit skin maps on fixed RealDense geometry. Compare to identical fitting without visibility on held-out real views/lights.
2. **Hypothesis:** geometry confidence determines which self-shadow boundaries should affect optimization. Evaluate registration/calibration failures separately.

**Explain it back:** how can ignoring a shadow create false pigmentation? Why is this monocular method not equivalent to single-image reconstruction?
