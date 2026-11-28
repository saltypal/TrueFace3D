"""FaceScape-C: real FaceScape multi-view photos paired with TU-model ground truth.

Modules
-------
constants  expression names, view targets, crop size, subject splits
sources    read params, photos, TU meshes and textures straight from the downloaded zips
mesh       minimal OBJ reader for TU meshes (keeps the shared topology)
geometry   TU (mm) -> camera transforms, view angles, view selection, projection
crop       undistort + square face crop + crop camera
splits     subject table (age, gender, split, publishable)
build      builds the dataset folder (clean stage, degraded stage, manifests)
dataset    read-side API used by notebooks and Kaggle jobs
"""
