# 02 — FLAME — a controllable family of faces

**Learning a model of facial shape and expression from 4D scans; Li et al., SIGGRAPH Asia 2017.** Snapshot: 2026-10-10. [Original PDF](../pdfs/02_FLAME.pdf), [primary source](https://flame.is.tue.mpg.de/). Read: §§3–5: registration, model definition, training data; model-fitting experiments.

## Intuition and prerequisites

FLAME is a decoder, not an image reconstruction network. Think of a face sculpture with sliders: one set controls the person's underlying head shape, another controls expression, and joints control jaw, neck and eyes. An image-based method such as RealDenseFace estimates those sliders. Sharing a decoder makes faces comparable and animatable.

## Architecture and execution

1. Register a common template to captured heads so corresponding vertices mean the same anatomical locations.
2. Learn a linear identity space from neutral heads and an expression space from registered expression sequences.
3. Add pose-dependent corrective blendshapes to compensate for the fact that a skin surface does not move like rigid bones.
4. Calculate joint positions from identity and articulate the jaw, neck and eyes.
5. Use linear blend skinning to combine joint transformations for each vertex. Return a mesh with fixed connectivity.

The numbered sequence is a text architecture diagram: each step consumes the previous step's output unless explicitly stated otherwise. Symbolic dimensions are used where exact implementation dimensions were not established.

## Mathematics and worked example

A useful model form is \(M(\beta,\psi,\theta)=W(T(\beta,\psi,\theta),J(\beta),\theta,\mathcal W)\), with \(T=\bar T+B_S\beta+B_E\psi+B_P(\theta)\). Here \(W\) is skinning, not a learned image encoder; \(\mathcal W\) stores bone weights. Pose correctives depend on pose and should not be counted as identity changes.

Illustrative: a lip vertex weighted 0.7 to the jaw and 0.3 to the head follows 70% of the jaw-transformed position and 30% of the head-transformed position. This blending preserves attachment, but can introduce artifacts; learned correctives reduce them. A smile changes expression coefficients while the person's identity coefficients remain fixed.

## Training data and supervision

The original shape model uses 3,800 head scans. Expression and pose learning uses D3DFACS and additional captured 4D sequences; total training exceeds 33,000 scans. The authors capture active-stereo dynamic sequences with 3dMD, align templates and alternate registration/model improvement. Registered scans, not an unaligned collection of photographs, are the necessary learning signal.

## Evidence and limits

Comparisons fit competing face models to static scans and dynamic sequences under the same optimization procedure. They test expressiveness and articulation, not photorealistic texture generation. The original paper predates FLAME2023; cite it for principles and inspect the licensed 2023 asset for current topology, basis and joint definitions.

A low-dimensional linear space cannot represent every identity or skin pore. Texture, hair, teeth, correct corneal optics and individualized ear microgeometry are separate assets. The model's metric units do not remove the single-image camera/scale ambiguity.

## TrueFace3D use and code reading

Canonical geometry for TrueFace3D. Freeze topology and UV versions in every asset manifest. Read the upstream FLAME implementation, shape/expression blendshapes, Rodrigues rotations, joints and skinning. Model files need their own license; Python code availability does not grant redistribution rights to model assets. The primary author's old download redirected to HTML during our fetch; the original paper PDF was obtained from the documented 3dvar mirror.

**Release audit, 2026-10-11:** the [official site](https://flame.is.tue.mpg.de/) now lists **FLAME2023 Open under CC-BY-4.0**, released November2025. Older model versions retain their separate terms. The site links an [expression-basis converter](https://github.com/Suraj-Bhor/flame_to_flame_open_converter). Coefficients need conversion and geometry checks; the Open file is not an automatic drop-in for RealDense's expected model. This update changes asset availability, not the 2017 paper's architecture.

No model training or inference was performed for this note. Claims above describe the authors' method; the changes below are our hypotheses.

## Improvement experiments

- Add bounded residual displacement outside the linear space; test held-out scan distance and self-intersection before/after.
- Condition regional residuals on confidence, while retaining the base FLAME coefficients for animation. Compare against simply adding more basis coefficients.

## Explain it back

Why is a mesh decoder different from an encoder? Why do fixed vertex correspondences matter for transferring textures? Why are pose correctives needed after skinning?

