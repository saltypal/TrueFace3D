# 42 — Becoming Lit: FLAME-driven Gaussian avatars with hybrid shading

**arXiv 2506.06271, 2025.** Snapshot: 2026-10-10. [PDF](../pdfs/42_BecomingLit.pdf), [paper](https://arxiv.org/abs/2506.06271). Dataset: PDF pp. 3–4; architecture: pp. 4–6; experiments: pp. 7–8. A controlled-light avatar reference, not our one-photo asset backbone.

## Intuition

A smooth parametric face supplies animation. Many small Gaussian primitives add detailed appearance around that face. A neural model learns the complicated soft way skin transports light, while an analytical model handles shiny reflections.

It keeps useful physics without asking a simple material equation to explain every skin effect. However, it learns a personalized avatar from a light stage, so its detail cannot be credited to one input photograph.

## Architecture

```text
multi-view OLAT + fully lit frames -> VHAP FLAME tracking
FLAME expression -> geometry CNN Fg -> UV Gaussian attributes
FLAME UV positions + tangent frame + static/dynamic offsets -> 3D Gaussians
diffuse shading MLP Fd -> learned diffuse light transport
expression features + view direction -> CNN Fv -> specular intensity/normal offsets
analytical specular BRDF + neural diffuse -> gsplat rendering
 -> image + geometric regularization -> personalized avatar
```

Every valid UV texel defines a Gaussian attached to the posed FLAME surface. Static offsets carry most identity detail; expression-dependent offsets are regularized to stay small. Fg also predicts orientation, scale, opacity, and expression features. The appearance system models diffuse transport with a small neural network, including effects not explained by a Lambertian surface. Its specular term uses a two-Blinn-Phong-lobe mixture with analytical Fresnel/masking terms.

The view-conditioned Fv predicts normals/intensities because one ellipsoidal primitive does not have a unique surface normal across views. Such view-conditioned normals should not be blindly exported as a fixed tangent normal map.

## Equations and worked example

Gaussian centers follow

\[
\mu_k=\hat\mu_k+R^{TBN}_k\mu^{local}_k+\delta\mu_k.
\]

Illustrative example: a Gaussian's FLAME UV point is on a cheek. Its static offset is 0.3 mm along the local normal; a smile adds 0.1 mm of dynamic displacement. When the head rotates, the tangent frame rotates the static offset with the cheek. Adding those offsets in an unrotated camera frame would make the detail slide.

The image loss combines L1 and SSIM; normal, opacity-mask, scale, and dynamic-position regularizers stabilize fitting. Neural diffuse and analytical specular are separate explanations of outgoing radiance, not two independently edited screenshots.

## Capture data and experiments

The paper introduces a light-stage dataset with **10 identities, 16 cameras, 40 light configurations**, 72 fps, 2200 × 3208 images. Every third frame is fully lit for tracking; other frames change lights. Access is restricted to approved noncommercial academic use.

Experiments use four subjects, 15 training cameras and one held-out camera, four held-out light patterns, scripted sequences for training, and free motion for testing. Models train for 250K iterations at 1100 × 1604 with about 202K primitives from a 512² UV map, taking roughly 30 hours on an RTXA6000. FLAME2023 is augmented with teeth. These conditions are far beyond an unseen-person one-photo prediction task.

The paper compares hybrid shading to classic PBR and learned precomputed radiance transfer. Improvement in held-out lights supports that training regime, not universal performance from one image.

## TrueFace3D integration and improvements

Use as a later research comparison for neural residual shading and FLAME-bound appearance. Default to explicit materials for portable assets and a physically specified shared renderer. If a Gaussian branch is added later, keep its representation and learned transport distinct from the mesh material export.

1. **Hypothesis:** a bounded neural residual improves difficult skin scattering without corrupting identity geometry. Compare physics-only, unrestricted neural shading, and bounded residuals using unseen lights.
2. **Hypothesis:** one-photo distillation from authorized multi-light teachers helps a lightweight material student. Track teacher access, identity splits, and remaining ambiguity.

**Explain it back:** why are primitive offsets expressed in a tangent frame? Which learned attributes cannot be exported faithfully as a standard fixed material map?
