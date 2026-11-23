# 40 — SFDM: geometry and reflectance help each other under sparse views

**arXiv 2312.06085.** Snapshot: 2026-10-10. [PDF](../pdfs/40_SFDM.pdf), [paper](https://arxiv.org/abs/2312.06085). Method: PDF pp. 3–6; data/experiments: pp. 6–8. Existing integration prior closely related to our proposed joint geometry/material reasoning.

## Intuition

A dark cheek patch might be a shadow, a skin mark, or a dent. Predicting geometry and color independently can pick incompatible explanations. SFDM learns a generic face with shape and reflectance, then lets the shape/material branches exchange information while fitting an individual.

It uses several photographs, not one. Its contribution shows why connecting components can matter; it also means “we connected shape and appearance” is already established prior art.

## Two-stage architecture

```text
Stage 1: multi-person calibrated images
 -> deformation field -> shared SDF template + reflectance template
 -> albedo-gradient predictor -> regularized physical rendering
Stage 2: 3/5/10 views of a new individual
 -> template-conditioned geometry displacement + BRDF offsets
 -> shared geometry/reflectance/gradient features between branches
 -> subsurface-scattering diffuse offset
 -> inverse-render refinement
```

The geometry latent has 128 dimensions. A deformation network maps individual coordinates into template coordinates. Separate reflectance attributes include diffuse albedo, four specular channels, and roughness. Stage 1 additionally learns gradients of fitted albedo, giving boundary/structure cues rather than only raw colors.

Stage 2 adds a displacement network for SDF corrections and a BRDF-offset network. Their inputs include deformation, geometry, reflectance, and albedo-gradient features; they exchange additional detail features. Learned weights control how much of each template attribute to retain. Diffuse albedo varies greatly between people, so its template weight starts low.

The renderer uses a split-sum-style approximation and environment lighting represented by spherical harmonics through order 10. A learned offset approximates subsurface transport using Gaussian-profile features rather than executing a full physical path tracer.

## Equations and worked example

\[
S(x)=T_{geo}(F_{dfm}(x,z_{geo}))+F_{dis}(x),
\qquad
B(x)=W(x)T_{rfl}(F_{dfm}(x,z_{geo}))+F_{ofs}(x).
\]

Illustrative example: a template predicts albedo 0.5, its weight is 0.2, and the learned offset is 0.3; the resulting attribute is 0.4. The weight chooses prior influence; it is not automatically a calibrated confidence. The networks must still obey range and lighting constraints.

Color reconstruction, Eikonal, white-light/specular-energy priors, latent and offset regularization train Stage 1. Stage 2 adds displacement and scattering-offset regularization. These constraints prevent arbitrary colors, huge geometry changes, or an unconstrained scattering term from explaining every pixel.

## Data and experiments

The paper uses approximately **30 authorized FaceScape identities**: ten identities × ten views for template learning; twenty other identities for 3-, 5-, and 10-view refinement, with ten further views for testing. Both stages run 3,000 epochs. H3DS supplies supplementary generalization evaluation. Only four selected FaceScape identities were on the publishable list; eye mosaics protect others in displayed results.

This uses real calibrated images without ground-truth BRDF labels. Photometric decomposition is therefore underdetermined and depends on priors. PSNR/SSIM/LPIPS measure held-out rendered images, and Chamfer measures geometry. Better decomposition images do not establish measured roughness accuracy.

## TrueFace3D use and improvements

Keep SFDM as an integration baseline/reference, not a strict one-photo result. Transferring its ideas to fixed FLAME UV material maps requires explicit coordinate and renderer contracts. A one-image adaptation is a new untested experiment, not the reported method.

1. **Hypothesis:** use geometry uncertainty to limit material-to-geometry updates. Compare to SFDM-style unrestricted feature exchange and independent modules.
2. **Hypothesis:** separate observed versus completed UV regions in the losses. Compare held-out camera/illumination behavior, rather than only the input render.

**Explain it back:** what do the two stages learn, and why can an albedo gradient help geometry? Which reported input settings violate our one-photo inference constraint?
