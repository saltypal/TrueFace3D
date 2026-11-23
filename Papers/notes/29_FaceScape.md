# 29 — FaceScape: capture, parametric shape, and dynamic detail

**CVPR 2020; arXiv 2003.13989.** Snapshot: 2026-10-10. [Original PDF](../pdfs/29_FaceScape.pdf), [paper](https://arxiv.org/abs/2003.13989), [current dataset and access](https://nju-3dv.github.io/projects/FaceScape/). Read PDF pp. 3–6 for capture, bilinear modeling, and detail prediction. This is both a dataset paper and a reconstruction method; the current release differs from the original study.

## Intuition

Take photographs of a person from many directions at almost exactly the same instant. Matching the photographs lets you build a detailed sculpture. Repeat while they smile, frown, and open their mouth. You now have examples of which parts belong to that person and which parts change with expression.

FaceScape compresses each sculpture into a simple mesh plus a displacement map: a smooth toy face plus instructions saying where its surface should rise or sink. Its reconstruction method learns to predict those instructions for expressions beyond the one visible in a photograph.

## Architecture and execution

```text
68 synchronized DSLR cameras -> detailed reconstructed scans
  -> landmark alignment + nonrigid ICP -> common mesh topology
  -> identity/expression tensor -> Tucker decomposition -> bilinear model
single input image -> landmark/photometric fitting -> base shape + UV texture
  -> UV deformation maps + texture -> pix2pixHD -> expression detail maps
  -> expression-dependent detail synthesis -> riggable detailed face
```

The original capture study includes 938 subjects performing 20 expressions. Thirty cameras capture the front at 8K; the other cameras capture side views at 4K. Shutters synchronize within 5 ms. Raw meshes have roughly two million vertices. Most participants are Asian, an important population limitation rather than a universal face distribution.

Registration makes corresponding vertex indices mean the same thing across people. Normal-direction projection from a base mesh to its scan yields signed displacement. The statistical model uses 26,317 vertices and generates 52 expression blendshapes per subject. Tucker decomposition reduces the identity–expression tensor rather than training an image encoder.

For single-image fitting, landmarks, rendered pixel agreement, and Gaussian parameter regularization estimate the base face. Texture and a UV deformation map then condition a pix2pixHD generator. Its displacement outputs cover 20 basic expressions; synthesis combines the learned detail behavior for a requested expression. Adversarial and discriminator-feature matching losses teach detailed map prediction.

## Mathematics and worked example

The bilinear model contracts a core tensor with identity and expression coefficients:

\[
V=C_r\times w_{exp}\times w_{id}.
\]

The displayed mesh is then conceptually refined by \(v'_i=v_i+d_i n_i\). Illustrative example: a forehead vertex at 50.0 mm along its normal receives a +0.2 mm displacement and moves to 50.2 mm. Its identity coefficient does not change when the person raises their eyebrows; the expression coefficients and predicted wrinkle map do. This separation is the reason the resulting face can be animated.

## Data, experiments, and release caveats

The paper compares parametric fitting on 1,000 meshes from 50 test subjects. Its reported compact representation error below 0.3 mm describes representing scans, not guaranteed accuracy from arbitrary photographs.

As checked on the current official page, downloadable TU models cover **847 subjects / 16,940 meshes**; the original 938-subject study is larger. The multi-view package has over 400,000 real photographs with calibration and shapes. The webpage's “359 × 20 = 7120” is arithmetically inconsistent: build counts from the downloaded valid manifest. Some textures for IDs 360–847 have privacy mosaics near the eyes. Access is noncommercial and students may need their supervisor's application; publication portraits/renders must respect its publishable list.

Captured color textures contain capture illumination and residual reflections. Treat them as observed texture, not measured diffuse/specular/roughness maps. FaceScape's topology is not FLAME's, and RealDenseFace already uses this dataset: an arbitrary FaceScape image is not automatically an unseen test example.

## TrueFace3D use and proposed improvements

Use valid camera–image–scan triples for one-photo input with other cameras reserved for evaluation or training supervision. Register detail to our pinned FLAME version before baking maps; retain registration residuals as label confidence.

1. **Hypothesis:** confidence-weighted displacement supervision reduces scan/registration artifacts. Compare identical training with and without residual masks on held-out identities.
2. **Hypothesis:** expression-stratified sampling helps sparse-expression cases. Compare to uniformly sampled frames at equal update count.
3. Test demographic/domain holdouts; do not call extra subjects alone a new algorithm.

**Explain it back:** why do common topology and dynamic detail solve different problems? Why is a photographed UV texture not necessarily albedo?
