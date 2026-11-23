# 27 — Disney shading — define what material parameters mean

**Physically Based Shading at Disney; Brent Burley, SIGGRAPH 2012 course notes.** Snapshot: 2026-10-10. [Original PDF](../pdfs/27_DisneyBRDF.pdf), [primary source](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf). Read: Material observations, diffuse/specular model design and parameter descriptions. This is technical course material, not a pretrained model.

## Intuition

A renderer needs a rule for how surfaces scatter light. A BRDF gives that rule. Disney's model makes a small set of physically motivated controls practical for artists and production. It lets our modules supply meaningful material parameters instead of independently colored screenshots.

## Architecture and execution

1. Analyze measured material responses and decide which effects must be represented.
2. Define diffuse and microfacet-inspired specular behavior with interpretable controls.
3. Account for roughness, grazing-angle Fresnel and geometric masking/shadowing.
4. Combine lobes for a material and evaluate them for a common light/view direction.
5. Use consistent parameter interpretation in every module and renderer adapter.

## Mathematics and worked example

A microfacet term has the structure \(f_s=D(h)F(\omega_o,h)G(\omega_i,\omega_o)/(4|n\cdot\omega_i||n\cdot\omega_o|)\). Roughness affects the distribution of microscopic orientations. Fresnel increases interface reflection toward grazing angles. These effects couple normals, light and view.

Illustrative: two skin patches can have the same diffuse RGB but different roughness. Under broad frontal light they look similar; under a small side lamp one has a narrow highlight and the other a broad one. A skin-color predictor alone cannot determine both responses.

## Data and supervision

No neural training occurs. The notes analyze measured material data such as MERL and production observations. Measured general material responses are evidence for shader design; they are not a facial albedo/scan training set.

## Evidence and limitations

Material parameterization and lobe behavior are the contribution. Later Blender Principled BSDF revisions are not numerically identical to the original 2012 notes. Use the original for first principles and current Blender documentation for actual socket semantics.

The BRDF assumes local surface scattering. Skin subsurface transport and corneal transmission need additional models. RGB albedo and artist parameters are not full spectral optical measurements. Wrong color management can overwhelm otherwise correct shading.

## Integration and implementation reading

Define a canonical skin material contract and convert to the chosen Cycles version explicitly. Start with dielectric skin and conservative specular/roughness defaults; fitting unsupported maps from one image can increase ambiguity. Ears, skin and eyeballs share lights, while the cornea uses a distinct optical material.

## Improvements to test

- Validate material conversions using a sphere under multiple light/view angles rather than checking one face render.
- Estimate only region-supported residuals and compare against constant-material baselines.
- Do not count selecting a standard production shader as scientific novelty.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

What distinguishes a BRDF from a texture? Why does roughness change highlights? Why does skin need more than local reflection?

