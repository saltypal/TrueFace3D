# 23 — DeNeRF — eye-aware neural rendering needs multiview data

**High-Fidelity Eye Animatable Neural Radiance Fields for Human Face; Wang et al., arXiv 2308.00773.** Snapshot: 2026-10-10. [Original PDF](../pdfs/23_EyeNeRF.pdf), [primary source](https://arxiv.org/abs/2308.00773). Read: §3 deformation and eye-mask sampling; §4.1 setup. File name retains the initial EyeNeRF category; the actual method is DeNeRF.

## Intuition

A radiance field can learn how a particular person's eyes look across directions. DeNeRF separates rigid eyeball motion from nonrigid changes around the eye. This is a person-specific multiview neural avatar, not a ready-made eye mesh reconstructed from one photograph.

## Architecture and execution

1. Track head, neck, jaw and eyeball poses using a parametric FLAME-based multiview procedure.
2. Map observed-space sample points toward a canonical pose with weighted rigid transformations.
3. Apply a learned MLP residual deformation conditioned on pose.
4. Query canonical NeRF color/density and integrate along camera rays.
5. Bias ray/patch sampling toward eyes so their tiny area is not neglected by whole-face training.

## Mathematics and worked example

The canonical mapping has the form \(x_c=T_{rigid}(x,p)+\Delta x(T_{rigid}(x,p),p)\). Standard NeRF integration accumulates colors with transmittance: \(C(r)=\sum_i T_i(1-e^{-\sigma_i\delta_i})c_i\). Density/color are a radiance representation, not independent diffuse/specular materials.

Illustrative: rotating an eyeball alone moves the iris but also changes nearby tissue appearance. Rigid mapping handles the rotation; the residual field models nonrigid behavior. If only one gaze was observed, predictions at another gaze are poorly constrained.

## Data and supervision

ETH-XGaze provides 110 subjects and eighteen-camera capture. The method selects nine gaze frames and thirteen retained views per frame: 117 training images per person. This is explicitly more than one image. The paper reports four A100 GPUs and a 400K training setting; its wording says epochs, which is ambiguous for iteration accounting and should not be silently rewritten as a verified epoch count. Training renders 512 × 512 using six 32 × 32 patches and 128 samples per ray.

## Evidence and limitations

Evaluation uses image PSNR/SSIM/LPIPS, novel gaze/view rendering and gaze-estimation augmentation. It demonstrates eye-aware sampling and deformation, not one-photo recovery of metric cornea geometry or a directly exportable physical eye material bundle.

Person-specific multiview training and large compute conflict with the first phase. Radiance fields can encode observed lighting, so arbitrary physical relighting is not guaranteed. No direct FLAME-compatible cornea/iris asset interface follows from a rendered image.

## Integration and implementation reading

Later multiview/video reference and a warning against installing an eye NeRF simply because its outputs look realistic. Keep the current eye component explicit and parametric. Check original implementation availability before considering a future neural-rendering extension.

## Improvements to test

- Transfer its eye-focused sampling principle to a small eye-fitting loss with equal pixel budgets.
- Evaluate cross-gaze consistency of explicit eye assets before increasing representation complexity.

No TrueFace3D model or renderer experiment was run for this survey. The experiments above are proposed.

## Explain it back

Why are 117 images per subject incompatible with strict single-photo inference claims? What does density mean? Why can't a radiance image be attached as a mesh component?

