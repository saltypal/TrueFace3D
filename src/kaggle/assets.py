"""Single list of private model files, shared by staging (copy) and setup (symlink).

Paths are project-relative. The dataset stores them under ``weights/<same path>``.
"""
MICA = "third_party/Pixel3DMM/src/pixel3dmm/preprocessing/MICA"
PIP = "third_party/Pixel3DMM/src/pixel3dmm/preprocessing/PIPNet"

REQUIRED_WEIGHTS = [
    # RealDenseFace (ViT-S, ViT-B)
    "third_party/RealDenseFace/weights/realdenseface/vits.pth",
    "third_party/RealDenseFace/weights/realdenseface/vitb.pth",
    "third_party/RealDenseFace/weights/facebox/face_box.pth",
    "third_party/RealDenseFace/weights/flame/flame2023.pkl",
    "third_party/RealDenseFace/weights/flame/flame_assets.npz",
    # DECA (licensed FLAME + released checkpoint)
    "third_party/DECA/data/deca_model.tar",
    "third_party/DECA/data/generic_model.pkl",
    # Pixel3DMM
    "third_party/Pixel3DMM/pretrained_weights/uv.ckpt",
    "third_party/Pixel3DMM/pretrained_weights/normals.ckpt",
    f"{PIP}/FaceBoxesV2/weights/FaceBoxesV2.pth",
    f"{PIP}/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth",
    "checkpoints/pixel3dmm/torch/hub/checkpoints/face_parsing.farl.celebm.main_ema_181500_jit.pt",
    "checkpoints/pixel3dmm/torch/hub/checkpoints/mobilenet0.25_Final.pth",
    # MICA + FLAME files that Pixel3DMM's preprocessing reads
    f"{MICA}/data/pretrained/mica.tar",
    f"{MICA}/data/FLAME2020/generic_model.pkl",
    f"{MICA}/data/FLAME2020/landmark_embedding.npy",
    f"{MICA}/data/FLAME2020/FLAME_masks/FLAME_masks.pkl",
    f"{MICA}/data/FLAME2023/flame2023.pkl",
    f"{MICA}/data/FLAME2023/flame2023_no_jaw.pkl",
    # InsightFace: SCRFD detector for DECA/MICA/Pixel3DMM, buffalo_l for ArcFace identity metric
    "checkpoints/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
    "checkpoints/insightface/models/buffalo_l/det_10g.onnx",
    "checkpoints/insightface/models/buffalo_l/w600k_r50.onnx",
]

# Appearance backends (staged by default). setup.py unpacks the archives into the upstream checkouts.
APPEARANCE_WEIGHTS = [
    "checkpoints/opendelight.tar",
    "checkpoints/BalanceAlb_model.npz",
    "third_party/DECA/data/FLAME_albedo_from_BFM.npz",  # built locally from BFM 2017
    "checkpoints/TRUST_models_BalanceAlb_version.zip",
]
OPTIONAL_WEIGHTS = []

# All appearance weights are now on disk.
MISSING_FOR_APPEARANCE = {}

# Pinned upstreams: name -> (url, sha, dir under third_party)
PINNED = {
    "RealDenseFace": ("https://github.com/LinzhouLi/RealDenseFace.git",
                      "7d855a29931c95cc9016f75d49cead60adbcf0a3", "RealDenseFace"),
    "DECA": ("https://github.com/yfeng95/DECA.git", "a11554ae2a2b0f3998cf1fa94dd4db03babb34a2", "DECA"),
    "Pixel3DMM": ("https://github.com/SimonGiebenhain/pixel3dmm.git",
                  "fcd1fa973c7715b02a8948dfc679dff53cf85924", "Pixel3DMM"),
    "MICA": ("https://github.com/Zielon/MICA.git", "af22e7a5810d474bc28a1433db533723d6bd2b07",
             "Pixel3DMM/src/pixel3dmm/preprocessing/MICA"),
    "nvdiffrast": ("https://github.com/NVlabs/nvdiffrast.git",
                   "253ac4fcea7de5f396371124af597e6cc957bfae", "nvdiffrast"),
    "PyTorch3D": ("https://github.com/facebookresearch/pytorch3d.git",
                  "75ebeeaea0908c5527e7b1e305fbc7681382db47", "PyTorch3D"),
    "now_evaluation": ("https://github.com/soubhiksanyal/now_evaluation.git",
                       "7bd149850a5cad431a488857f1dc774e1eb533c9", "now_evaluation"),
    # Pixel3DMM preprocessing dependencies (same commits as the local tree)
    "facer": ("https://github.com/FacePerceiver/facer.git", "ddd35c76ff840174b8a5403ad1c1255e37b8782b",
              "Pixel3DMM/src/pixel3dmm/preprocessing/facer"),
    "PIPNet": ("https://github.com/jhb86253817/PIPNet.git", "b9eab58816437403a34aa5bc3adeafe5081fd36b",
               "Pixel3DMM/src/pixel3dmm/preprocessing/PIPNet"),
}

# Appearance upstreams, pinned to the commits inspected on 2026-11-23.
APPEARANCE_PINNED = {
    "TRUST": ("https://github.com/HavenFeng/TRUST.git", "f06cdd27d1748a8520d2ff340e4d14f74da847cf", "TRUST"),
    "OpenDelight": ("https://github.com/yxuhan/OpenDelight.git", "431ad736371dcfbf2ce29cfa84b04512df4ea107",
                    "OpenDelight"),
}
UNPINNED = {}
