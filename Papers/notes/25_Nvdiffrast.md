# 25 — nvdiffrast — differentiable rasterization primitives

**Modular Primitives for High-Performance Differentiable Rendering; Laine et al., SIGGRAPH Asia 2020, arXiv 2011.03277.** Snapshot: 2026-10-10. [Original PDF](../pdfs/25_Nvdiffrast.pdf), [primary source](https://arxiv.org/abs/2011.03277). Read: Rasterization, interpolation, texture filtering and antialiasing sections; official API guide.

## Intuition

Rendering usually turns 3D assets into pixels. Differentiable rendering also tells us how a small change to an asset changes the pixel error, so an optimizer can improve it. nvdiffrast supplies the graphics steps; we must supply cameras, materials and lighting.

## Architecture and execution

1. Transform vertices into homogeneous clip coordinates using our camera/model matrices.
2. Rasterize triangles into a buffer of barycentric coordinates, normalized depth and triangle ID.
3. Interpolate UVs, normals and any other vertex attributes.
4. Sample material textures with bilinear/mipmap filtering and apply our shader.
5. Apply antialiasing to improve pixel coverage and obtain meaningful silhouette-related gradients.

## Mathematics and worked example

For barycentric weights \(b_1+b_2+b_3=1\), interpolate an attribute as \(a=\sum_j b_ja_j\). The rasterizer output has four channels: two barycentric coordinates, z/w and triangle ID. Chain-rule gradients propagate image error through interpolation/texturing/shading to supported parameters; the depth channel itself is not a generic differentiable depth-loss shortcut.

Illustrative: vertex UVs are (0,0), (1,0), (0,1), and barycentric weights are (0.2,0.3,0.5). The pixel UV is (0.3,0.5). Sampling the albedo there is a separate operation from deciding which triangle is visible.

## Data and supervision

This is a rendering library/algorithm paper, not a learned facial model. There is no facial training dataset to report. Experiments benchmark rendering primitives and optimization tasks on graphics assets. Our own training data supplies the face supervision.

## Evidence and limitations

Its benefit is fast modular differentiable operations. Official documentation explicitly states it has no built-in camera, lighting or material models. A renderer running without errors does not prove our shader or coordinate transform is correct.

Visibility is discontinuous; antialiasing supplies boundary gradients using a coverage approximation. Rasterization does not automatically provide full shadows, multiple scattering, transparent corneal refraction or path-traced global illumination. CUDA/toolchain compatibility matters.

## Integration and implementation reading

Use for fitting/learning and fast diagnostic rendering, with an explicit linear-light skin shader. Official implementation: https://github.com/NVlabs/nvdiffrast; guide: https://nvlabs.github.io/nvdiffrast/. Read triangle.py then interpolation/texturing examples before face integration. Check CPU/device fallbacks honestly; this is not a generic TPU renderer.

## Improvements to test

- Compare finite-difference and autograd derivatives for camera, texture and vertex perturbations on a tiny triangle scene.
- Measure the shader gap against Cycles using identical assets/lights, first with diffuse-only material, then GGX.
- Treat a rendering upgrade as an engineering improvement unless its method and experimental contribution are novel.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

Which operations does the library own? Where does skin shading enter? Why do silhouettes need special gradient handling?

