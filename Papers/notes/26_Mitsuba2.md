# 26 — Mitsuba 2 — programmable forward and inverse light transport

**Mitsuba 2: A Retargetable Forward and Inverse Renderer; Nimier-David et al., SIGGRAPH Asia 2019.** Snapshot: 2026-10-10. [Original PDF](../pdfs/26_Mitsuba2.pdf), [primary source](https://tizianzeltner.com/projects/NimierDavidVicini2019Mitsuba2/). Read: System design/type transformations and inverse-rendering demonstrations; current Mitsuba 3 docs separately.

## Intuition

A path tracer simulates light paths through a scene, including reflections and transmission. An inverse renderer exposes derivatives or optimization tools so we can estimate scene parameters from images. Mitsuba is a complete research rendering system, unlike nvdiffrast's smaller set of rasterization primitives.

## Architecture and execution

1. Define scene geometry, camera/sensor, materials, textures and emitters as modular plugins.
2. Select rendering behavior through parameterized numeric/color types.
3. Execute an integrator that samples light transport paths.
4. Instrument computations for vectorization, GPU execution and differentiation where supported.
5. Compare rendered images to targets and optimize selected scene parameters.

## Mathematics and worked example

The rendering equation is \(L_o(x,\omega_o)=L_e+\int f_r(x,\omega_i,\omega_o)L_i(x,\omega_i)|n\cdot\omega_i|d\omega_i\). Monte Carlo estimation samples this integral. For inverse rendering, differentiate an image loss with respect to material/camera/geometry parameters; visibility boundaries require special treatment beyond ordinary arithmetic differentiation.

Illustrative: if a predicted cheek is too shiny, optimization can adjust roughness while rerendering its highlight. If camera or geometry is wrong, the same loss may compensate by changing the material instead. Parameter constraints and independent geometric evidence are still required.

## Data and supervision

There is no pretrained face network or fixed facial training corpus in this systems paper. It evaluates renderer transformations and inverse-light-transport examples, including optical/volume tasks. Our measured images or synthetic controlled scenes become inverse-rendering observations.

## Evidence and limitations

The paper demonstrates one generic renderer supporting multiple execution/differentiation modes. It is evidence for programmable light transport, not a claim that this engine alone reconstructs correct skin from one photo.

Monte Carlo images/gradients are noisy and can be costly. Selecting an appropriate material/integrator matters. Mitsuba 2 uses Enoki-era infrastructure; current Mitsuba 3 uses Dr.Jit and changed APIs. Do not cite the older paper as exact documentation for current setup.

## Integration and implementation reading

Optional research backend if the approximation gap becomes a real limitation. Primary first renderer remains Cycles for final images and nvdiffrast for fitting, avoiding multiple expensive integrations immediately. Current docs: https://mitsuba.readthedocs.io/en/stable/. Check supported SSS, refraction, differentiation and backend combinations before relying on them.

## Improvements to test

- Benchmark inverse material fitting against the simpler rasterized shader at fixed error and compute budgets.
- Choose differentiable path tracing only for the effects that demonstrably alter eye/skin accuracy, rather than replacing the entire pipeline prematurely.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

What does an integrator compute? Why is automatic differentiation insufficient for some visibility changes? Why must current software documentation accompany an old systems paper?

