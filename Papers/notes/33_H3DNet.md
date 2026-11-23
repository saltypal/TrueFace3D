# 33 — H3D-Net and H3DS: few photographs with a learned head prior

**ICCV 2021; arXiv 2107.12512.** Snapshot: 2026-10-10. [PDF](../pdfs/33_H3DNet.pdf), [paper](https://arxiv.org/abs/2107.12512), [official H3DS dataset utilities](https://github.com/CrisalixSA/h3ds). Architecture: PDF pp. 3–5; dataset and experiments: pp. 5–8. Dataset utilities are not automatically the full H3D-Net implementation.

## Intuition

With just three photos, many possible heads could match the visible silhouette. A learned library of plausible head shapes narrows the possibilities. H3D-Net first searches within that library; only later does it let the shape network change to add details.

Imagine finding the closest premade sculpture and then carefully carving it. Carving freely from the beginning can explain one photograph with a nonsensical surface.

## Architecture and fitting schedule

```text
private scan collection -> auto-decoder SDF prior F(z, x)
3+ RGB photographs + camera parameters + masks
 -> geometry prior + implicit differentiable renderer
 -> fit latent z with geometry-network weights frozen
 -> then unfreeze/refine geometry network
 -> reconstructed surface + learned image appearance
```

The SDF decoder takes positional-encoded coordinates and a 256-dimensional shape code. It has eight 512-neuron MLP layers, a skip connection to the fourth layer, and Softplus activations. Surface and Eikonal objectives let it train from incomplete scans without requiring fully watertight geometry.

At inference, sphere tracing finds surface intersections. The rendering model estimates colors using surface/feature information and viewing direction. Photometric and mask evidence drives optimization. The staged freezing schedule uses the prior before opening the larger network's degrees of freedom. Its goal includes full heads, shoulders, and hair; it is not a FLAME parameter predictor.

## Mathematics and worked example

The surface is \(\mathcal S=\{x:F_{z,\theta}(x)=0\}\). A simplified fitting objective is

\[
E(z,\theta,\phi)=E_{photo}+\lambda_mE_{mask}+\lambda_eE_{Eikonal}+\lambda_z\|z\|^2.
\]

First optimize \(z\) with \(\theta=\theta_0\); later optimize \(\theta\) too. Illustrative example: three views suggest a wide nose. A latent update picks a plausible wider-nose prior; later network updates recover a local bridge bump. If the appearance model instead darkens the bridge to fake the bump, input-view agreement alone cannot establish correct geometry.

## Training data and evaluation data

The prior uses an **internal scan dataset of 10,000 individuals**, with a smaller 500-person prior used in ablations. That training resource is not supplied by the public H3DS utilities. The original H3DS benchmark contains 23 subjects with real images and high-quality scans; the later public v2 release lists 59 subjects. Keep these versioned counts separate.

The method assumes **at least three input views**, associated masks, and cameras. Experiments compare few-view and denser-view settings. A paper ablation reports 3-view face mean distance improving from 4.04 mm without prior/schedule to 1.49 mm with its large prior and schedule. These are the paper's aligned dataset results, not an estimate for TrueFace3D or for one image.

H3DS access is an academic/noncommercial process with authorization/download tokens. Never store tokens in the survey or Git. Its real image/scan relationship makes it a valuable external geometry dataset once the allowed one-view evaluation protocol is defined.

## Limitations and integration

This is few-shot optimization with private pretraining, not a ready replacement for our strict single-photo RealDenseFace baseline. Missing prior weights/data prevent exact reproduction. Its view-dependent appearance is not a complete physically separated skin material.

Use H3DS for external one-image evaluation after permission and overlap checks; compare only to methods under the same input/view and calibration assumptions. Borrow the **staged optimization principle**: establish geometry before allowing flexible material/detail models to absorb its errors.

1. **Hypothesis:** freeze–release schedules improve our geometry/material coupling. Compare fixed geometry, unrestricted joint fitting, and bounded staged fitting.
2. **Hypothesis:** a calibrated uncertainty gate decides when geometry refinement is safe. Test against scan errors, not just prettier input renders.

**Explain it back:** which data are public benchmark data and which are private prior-training data? Why does the three-view result not establish single-image performance?
