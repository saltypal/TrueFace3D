# 35 — Codec Avatar Studio: Ava-256 and Goliath-4 are different resources

**NeurIPS 2024 Datasets and Benchmarks.** Snapshot: 2026-10-10. [PDF](../pdfs/35_CodecAvatarStudio_AVA256.pdf), [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/9712b78386cebdc3db7f1a48c2d20edb-Paper-Datasets_and_Benchmarks_Track.pdf), [Ava-256 official release](https://github.com/facebookresearch/ava-256). Dataset details: PDF pp. 3–6; model: pp. 6 onward.

## Intuition

A digital face can be driven by a compact expression code rather than storing every video frame. Codec Avatar Studio supplies the recordings and modeling tools for this: photographs from a camera dome, headset views for driving, and derived surface/tracking assets.

It also supplies a smaller dataset with changing lights. Ordinary dome images teach appearance under a fixed setup; changing-light captures teach how appearance responds to illumination. They are related but not interchangeable labels.

## Dataset pipeline and model architecture

```text
Ava-256: dome + headset captures -> calibrated images, reconstruction, UV/tracking assets
neutral identity geometry/texture -> identity-dependent decoder biases
expression geometry/texture minus neutral -> variational expression encoder
 -> expression code -> convolutional decoder
 -> tracked mesh + volumetric primitive poses/opacities + RGB
 -> differentiable volumetric renderer -> real-image losses
```

The multi-identity baseline represents faces as a **mixture of volumetric primitives (MVP)**: small semitransparent cubes carrying color. Geometry and appearance branches decode the shared expression code. Neutral identity features condition decoder biases; subtracting a neutral frame helps the expression encoder avoid encoding identity. A headset encoder can later predict expression codes from headset cameras.

This model outputs radiance-like primitive appearance, not simply a FLAME mesh with measured BRDF maps. Its neural representation needs its own renderer. Exporting an OBJ would discard much of what makes it realistic.

## Mathematics and worked example

Conceptually, \(z_{exp}=E(G-G_0,T-T_0)\), and \((G,P,C)=D(z_{exp};b_{id})\). Here G is geometry, T observed UV texture, P primitive properties, and C color; this is a readable abstraction, not an exact tensor specification.

Illustrative example: two people both raise an eyebrow. A common expression code describes the movement; neutral-identity conditioning retains each person's eyebrow shape and skin tone. A camera ray accumulates colors through partly transparent primitives, so alpha and ordering affect the final pixel. Those RGB values should not be copied into a diffuse-albedo map without a separation model.

## What is captured, how much, and why

**Ava-256 contains 256 subjects**, with a proposed 240/16 identity training/validation split. The physical dome has roughly 170 cameras; released compressed configurations can select fewer views, so “80 selected views” is not the capture rig size. Quest Pro recordings supply five released IR views. The paper offers 4/8/16/32 TB data configurations instead of its multi-petabyte raw volume. Select captures/views explicitly before downloading.

**Goliath-4 contains four subjects** and adds heads, hands, bodies, phone data, and directionally varying illumination for head/hand captures. Group-light and random-light frames are interleaved with fully lit tracking frames. Phone head data include real RGB and depth; some modalities have fewer derived labels. Goliath's four IDs are useful for controlled-light tests but weak evidence of population generalization.

The paper releases code/models under CC-BY-NC 4.0; verify the exact current data terms for each asset. RealDenseFace already uses Ava-derived training. A pretrained model's exposure must be checked before treating an Ava identity as unseen.

## Evidence, integration, and improvements

Codec avatar experiments concern rendering and driveability. Photorealism from many calibrated training views does not demonstrate one-image reconstruction. Derived geometry has tracking error, dome texture retains lighting, and headset images differ strongly from ordinary RGB portraits.

Use selected Ava frames for geometry correspondence training or renderer validation; use authorized Goliath light variations for relighting checks. Keep training input one-photo while additional views remain supervision only.

1. **Hypothesis:** source-balanced sampling avoids Ava dominating smaller scan datasets. Test equal updates and report per-source/identity metrics.
2. **Hypothesis:** light-held-out supervision improves material separation more than extra same-light frames. Compare under matched subject count.

**Explain it back:** why is Goliath's relighting evidence different from Ava's ordinary dome texture? Why can't MVP colors directly become a portable PBR material?
