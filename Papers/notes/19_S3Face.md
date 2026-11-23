# 19 — S3-Face — skin scattering and coupled diffusion maps

**S3-Face: SSS-Compliant Facial Reflectance Estimation via Diffusion Priors; Ren et al., CVPR 2025.** Snapshot: 2026-10-10. [Original PDF](../pdfs/19_S3Face.pdf), [primary source](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_S3-Face_SSS-Compliant_Facial_Reflectance_Estimation_via_Diffusion_Priors_CVPR_2025_paper.html). Read: §3 and Figures 3–5; §4.1, original PDF p.6.

## Intuition

Skin light can enter at one spot and leave nearby. S3-Face predicts not only surface reflectance but pigment-related maps that support a scattering-aware skin model. It couples material outputs so the same face structure is preserved across domains.

## Architecture and execution

1. Encode RGB/texture and material images with a frozen Stable Diffusion V2 VAE.
2. Fine-tune a conditional diffusion UNet for diffuse, specular and detailed normals.
3. Use joint-reflectance attention to share structure across outputs and train with several lightings sharing one diffuse target.
4. Predict hemoglobin and melanin from diffuse albedo with a second diffusion stage.
5. Complete/map texture for single-image inference or fuse per-view estimates for multiview input, then use an SSS-aware shader.

## Mathematics and worked example

The first stage uses v-prediction: \(v_t=\alpha_t\epsilon-\sigma_tz\). Joint attention links domains; the second stage conditions pigment estimates on diffuse albedo to reduce direct illumination contamination. Scattering needs a spatial transport model, not just multiplying a diffuse texture by a constant red tint.

Illustrative: an ear backlit by a bright lamp appears reddish even when its unlit surface color is not bright red. A diffuse-only shader cannot reproduce that transport accurately. However, one frontal photo cannot determine exact internal tissue parameters; predicted pigment maps remain learned estimates.

## Training data and supervision

The authors capture 100 participants with 32 expressions and seven viewpoints, using a ninety/ten subject train/test split. Five domains include diffuse, specular, normals, hemoglobin and melanin. They supplement skin-tone coverage with 142 purchased high-resolution albedo maps. VISIA analysis images from 128 patients are evaluation data, not the primary ninety-person training set. Full-UNet fine-tuning at 768 × 768 takes about eight days on four A6000 GPUs; ten-step inference is reported at around 25 seconds per face on A6000.

## Evidence and limitations

The paper compares reflectance/relighting and pigment-domain analysis, including joint-generation behavior. Clinical analyzer comparison must not be turned into a medical claim about our avatar. The paper's topology-flexible estimator still needs a correct UV projection and normal conversion in an assembled mesh.

Measured training diversity is limited, purchased maps introduce coverage and rights constraints, and output optics need renderer-specific parameter conversion. The official https://github.com/xingyuren/s3face was inspected and is a README-only placeholder; executable weights were not verified. It is not a currently runnable default for our plan.

## Integration and code reading

Reference for physically meaningful skin behavior and cross-map consistency. Begin with explicit conventional SSS settings in Cycles; adopt this predictor only when executable assets and compute are verified. Use the local original PDF: title-only alphaXiv retrieval failed and is marked rejected.

## Improvements to test

- Add uncertainty for pigment/material maps and preserve observed skin features during completion.
- Compare our confidence-gated assembly against joint-reflectance attention; isolate the confidence mechanism with identical material predictors.
- Evaluate both material error and held-out-light ear/skin behavior instead of judging softness alone.

These are proposed experiments, not measured improvements. No model was run while creating this note.

## Explain it back

Why is SSS spatial transport? Why derive pigments from albedo instead of lit RGB? Which parts of this method are actually available to run?

