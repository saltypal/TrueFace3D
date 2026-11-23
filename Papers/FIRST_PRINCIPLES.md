# Before the papers: the face reconstruction vocabulary

Read this if a note uses an unfamiliar term. It explains the mechanism first and connects it to TrueFace3D.

## A photo is not a 3D model

A photograph records light arriving at pixels. A 3D asset stores surface shape, material behavior, and often controls for motion. Many 3D scenes can produce a similar photograph, especially when only one view is available. A model learns priors about faces to choose a plausible explanation; that does not make every inferred invisible detail a measurement.

## Mesh, parameters, and camera

A mesh is a list of 3D points (vertices) and connecting triangles. A parametric model describes a family of meshes using a shorter list of controls: identity β, expression ψ, rotations/pose θ. FLAME evaluates those controls to get vertices. A reconstruction network may predict controls directly or predict evidence a solver uses to fit them.

A camera maps those vertices into image pixels. Perspective divides by depth: doubling distance approximately halves apparent size for fixed focal length. This causes scale/focal ambiguity. A close small face and farther large face can appear similar. Cropping/resizing changes pixel coordinates, so retain the transformation between the crop and the original image.

## UV maps and correspondence

A UV chart flattens a surface like unfolding a globe into a map. Texture coordinates identify locations on the face independently of its pose. An image-space coordinate identifies a pixel in the photo. A correspondence links them. RealDenseFace predicts where a UV-indexed point appears in the image. Pixel3DMM predicts UV information for image pixels; reversing that direction is not a harmless naming change.

UV seams allow a chart to cut apart. One 3D vertex can have several UV coordinates on different chart edges. That is why a mesh can require separate triangle indices for positions and UVs. A same-sized texture from another mesh is not automatically aligned to our face.

## Albedo, observed texture, roughness, and reflections

Albedo represents diffuse surface color with illumination removed under a declared material model. A captured texture often includes shadows and highlights. If you paint a shadow into albedo and light the mesh again, the shadow is counted as permanent pigmentation.

Roughness measures microscopic surface variation and broadens specular highlights. Specular reflection is directional surface reflection; skin can be diffuse and specular together. Ordinary skin is a dielectric, not metallic. An HDR environment describes incident light directions/intensities. Camera exposure affects brightness; it is not a skin property.

An illustrative local diffuse relation is \(I\approx\rho\,s\), albedo times shading. If I=0.2, both (ρ=0.4,s=0.5) and (ρ=0.2,s=1) explain it. One photo alone cannot distinguish them without priors or extra evidence. Full rendering additionally includes specular, visibility, and light transport.

## Normals, displacement, and coordinate frames

A normal is a direction perpendicular to the surface. Light response changes when that direction changes. A normal map perturbs the shading direction; displacement physically moves points. Shader-only detail cannot alter a silhouette unless the renderer has additional geometry effects.

Object-space normals refer to the mesh coordinate frame. Tangent-space normals refer to a local UV-dependent basis (tangent, bitangent, normal). Transforming a tangent-space map onto a different UV layout requires decoding to a common frame and rebaking. Normal maps are vector data, not ordinary color photographs.

## BRDF and subsurface scattering

A BRDF is a rule connecting incoming light direction to reflected light direction at the same surface point. It tells the renderer how a material reflects. A BSSRDF allows light to enter one point and leave another: skin's soft translucent appearance needs subsurface transport. A scattering radius has length units; changing head scale without adjusting it changes the look.

The renderer evaluates materials, lights, visibility, and camera together. It does not need a separately trained image model for every anatomical part. Eyes can have a different optical material while belonging to the same scene.

## Eyes are layered optics

The sclera is the outer white region, the iris the colored structure, the pupil its opening, and the cornea the refractive front surface. A painted sphere puts everything on one shell and often paints a highlight into the iris. Layered geometry lets the highlight move with the light and refraction change apparent iris depth. Hidden personal geometry remains inferred unless measured.

## Differentiable rendering and optimization

Forward rendering asks: “What image does this scene make?” Inverse rendering asks: “Which scene parameters explain this image?” Differentiable rendering computes how pixel errors change when scene parameters change, enabling gradient updates.

RealDenseFace's geometry optimizer fits predicted correspondence/depth rather than requiring a full physical material renderer. Its Gauss–Newton step locally linearizes residuals and solves for parameter changes. Damping discourages unstable large steps. This is different from gradient-training a large neural model from scratch.

## Priors, uncertainty, and hallucination

A prior favors plausible shapes/materials when data are missing. A learned generative restorer may replace a blurred eye with a convincing learned pattern. That pattern is a hypothesis, not additional measured evidence. Compare it with raw-image geometry instead of treating visual sharpness as truth.

Log-variance describes a prediction's modeled uncertainty; a larger value should reduce its contribution to fitting. Confidence requires validation against actual errors. A learned attention mask is a feature-combination weight, not automatically a probability of correctness.

## Supervision and evaluation

Measured scan labels, controlled-light evidence, synthetic known-scene labels, and teacher pseudo-labels have different reliability. Training may use many views while test inference gets one; say so. Test scans/views/lights score outputs and must not tune them.

An identity-disjoint split tests new people. A camera-disjoint split tests new views of possibly familiar people. A light-disjoint split tests relighting. Pixel error, geometric error, recognition similarity, and human realism ratings measure different properties. A face can score well on one and poorly on another.

Continue with [RealDenseFace](notes/01_RealDenseFace.md), then follow [the reading order](README.md).
