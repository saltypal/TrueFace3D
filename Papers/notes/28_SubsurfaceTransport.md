# 28 — Subsurface light transport — why skin needs spatial scattering

**A Practical Model for Subsurface Light Transport; Jensen, Marschner, Levoy and Hanrahan, SIGGRAPH 2001.** Snapshot: 2026-10-10. [Original PDF](../pdfs/28_SubsurfaceTransport.pdf), [primary source](https://graphics.stanford.edu/~srm/publications/SG01-subsurf-abstract.html). Read: §2 BSSRDF theory, dipole approximation, measurements and sampling. Original PDF supplied through the documented Cornell course mirror.

## Intuition

Light does not always bounce off the exact point it hits. It may enter skin, scatter beneath the surface and emerge nearby. That softens shadows and makes thin ears glow. This physical effect explains why a skin-color texture and perfect geometry can still look artificial.

## Architecture and execution

1. Replace strictly local BRDF transport with a BSSRDF relating incoming and outgoing positions.
2. Model single scattering and approximate multiple scattering with diffusion.
3. Use real and virtual source dipoles to satisfy boundary behavior.
4. Measure optical properties of translucent materials and validate predicted transport.
5. Sample the scattering contribution within a conventional ray tracer.

## Mathematics and worked example

A BSSRDF \(S(x_i,\omega_i;x_o,\omega_o)\) permits \(x_i\ne x_o\); a local BRDF effectively requires the same surface point. Diffusion parameters involve absorption and reduced scattering. Scene units determine how far a numerical scattering distance travels.

Illustrative: a scattering radius of 1mm should spread light across a small skin neighborhood. Applying the same numeric value to a metre-based renderer as one metre makes the whole face glow. A unit mismatch can masquerade as a bad albedo network.

## Data and supervision

This is a physical rendering/measurement paper, not a deep-learning model. The authors measure translucent materials, including skin, and compare simulated/measured behavior. It does not provide a modern paired monocular face dataset with complete PBR maps.

## Evidence and limitations

Validation supports the approximation's transport behavior for the studied materials. It does not prove all skin layers, thin boundaries and spectral effects are modeled exactly. Modern renderers use evolved SSS methods; understand the principles and consult current engine implementation separately.

Diffusion has assumptions about scattering and geometry. Very thin tissue and detailed layered skin may violate simplified assumptions. A single photo cannot uniquely estimate all absorption/scattering parameters together with geometry and light.

## Integration and implementation reading

Our shared renderer must expose SSS scale/weight with declared units. Begin with a documented generic skin setting and mark it assumed. Compare ear backlighting and shadow transitions using shared lights. Keep scattering separate from pigment estimation and optional S3-Face outputs.

## Improvements to test

- Fit only a low-dimensional regional scattering residual when measured multilight targets support it.
- Test whether SSS improves held-out lighting without washing out identity marks; compare to diffuse/specular-only skin.
- Keep shader physics improvements separate from claims of recovered personal tissue properties.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

How does BSSRDF differ from BRDF? Why can changing scene units alter skin appearance? Which parameters remain unobservable in one image?

