# Chapter 1 model asset locations

Paths are relative to the TrueFace3D repository. Vendored code and model files are ignored by Git.

| Method | Source checkout | Model files |
|---|---|---|
| RealDenseFace | third_party/RealDenseFace | weights/realdenseface/vits.pth and vitb.pth, weights/facebox/face_box.pth, weights/flame/flame2023.pkl and flame_assets.npz |
| DECA | third_party/DECA | data/deca_model.tar and data/generic_model.pkl from FLAME2020 |
| Real-ESRGAN | third_party/Real-ESRGAN | weights/RealESRGAN_x4plus.pth |
| Pixel3DMM | third_party/Pixel3DMM | pretrained_weights/uv.ckpt and normals.ckpt |
| MICA | third_party/Pixel3DMM/src/pixel3dmm/preprocessing/MICA | data/pretrained/mica.tar plus data/FLAME2020 and data/FLAME2023 |
| PIPNet | third_party/Pixel3DMM/src/pixel3dmm/preprocessing/PIPNet | snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth |
| InsightFace | Installed by the preprocessing runtime | Archives staged under checkpoints/insightface; unpack into the runtime's model cache when configuring MICA |

Use the original FLAME2023 file for RealDenseFace and the no-jaw variant only where Pixel3DMM requests it. Asset placement and download validation are separate from environment setup and GPU inference. Files ending in .part are transfers in progress or incomplete files. Use the project audit for the RealDenseFace bundle and compare checkpoint hashes with the authors' published checksum list.
