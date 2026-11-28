"""Generate notebooks/30_dataset_inspection.ipynb (run: python notebooks/make_30_dataset_inspection.py).

The executed notebook shows faces of the publishable subjects only. It still stays local (licensed data).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kaggle.notebook_kit import code, md, save  # noqa: E402

cells = [
    md("""
# 30 - FaceScape-C: inspecting the whole dataset setup

**What you will learn.** Where every file comes from, how a real photo is paired with its true 3D face, and
whether each part of that pairing is correct. Each section ends with a check that prints PASS or FAIL, and the
final table collects them all.

**How the dataset is made, in one picture**
```
FaceScape Multi-View zip:  real photo (4-12 MP) + its camera (K, R, t, lens distortion)
FaceScape TU-Model zip:    3D face mesh in millimetres (+ 4K texture, displacement)
Toolkit Rt_scale_dict:     places the TU mesh into the photo's 3D frame (refined by ICP to the real scan)
                 |
   undistort the photo -> square crop around the face (768 px, face 480 px wide) -> clean.png + camera.json
                 |
   14 degradation profiles x 3 severities  ->  degraded/<profile>_s<k>.png
```
Faces are drawn only for the publishable subjects (340, 344); everyone else appears as numbers only.
"""),
    code("""
import os, sys, json, time
from pathlib import Path
REPO = next(p for p in [Path(os.environ.get("TRUEFACE3D_REPO", Path.cwd())), *Path.cwd().parents,
             Path("/mnt/DataBunker/TrueFace3D")] if (p / "src" / "facescape").is_dir())
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.facescape import constants as C, inspect as I
from src.facescape.dataset import FaceScapeC
from src.facescape.sources import Sources
pd.set_option("display.width", 200, "display.max_columns", 30)
ds, src = FaceScapeC.find(), Sources()
CHECKS = {}
print("dataset folder:", ds.root)
print("build info:", json.dumps(I.read_build_info(ds), indent=1)[:1500])
"""),
    md("""
## 1. Sources: which archives are on disk

Three kinds of FaceScape archive feed the dataset:
- **photos** (`fsmview_trainset_images_*`): real photos and cameras;
- **TU models** (`facescape_trainset_*`): the true 3D faces in millimetres;
- **multi-view meshes** (`fsmview_trainset_shape_*`): the raw 3D scans, used only to refine the alignment.

The second table lists, for each person from 321 to 359, which of the three are present. A person enters the
dataset only when both photos and TU are present.
"""),
    code("""
display(I.sources_table(src))
present = I.expected_vs_present(ds, src)
display(present.set_index("subject").T)
CHECKS["every dataset subject has photos and TU"] = bool((present[present["in dataset"]][["photos", "TU"]]).all().all())
CHECKS["people still waiting for a photo archive"] = int((~present.photos & ~present.invalid).sum())
CHECKS["people still waiting for a scan archive (refinement)"] = int((~present["mview mesh"] & ~present.invalid).sum())
"""),
    md("""
## 2. People and the train/test split

The test people are fixed in `constants.py` (3 women, 3 men, ages 22-66, both publishable subjects). Everyone
else is training data for GARD-Face and the prior corrector. **No person may appear in both.**
"""),
    code("""
summary, split_checks = I.split_summary(ds)
display(summary)
display(pd.DataFrame(ds.subjects))
CHECKS.update(split_checks)
"""),
    md("""
## 3. Cameras: how the three views are chosen

Each expression was photographed by about 58 cameras. This plot shows every camera as seen from the face:
- left-right angle (yaw) on the x-axis, up-down angle (pitch) on the y-axis;
- marker size = camera resolution; crosses = cameras FaceScape marks invalid.

The red rings are the three picked automatically: the valid, roughly level camera closest to 0°, +45° (the
person's left) and -45° (their right). Green bands show the allowed ±15° window.
"""),
    code("""
rig = I.rig_table(src, C.PILOT_SUBJECT, 1)
fig, ax = plt.subplots(figsize=(10, 5)); I.plot_rig(rig, ax); plt.show()
display(rig[rig.selected_as != ""][["view", "selected_as", "yaw", "pitch", "distance_mm", "megapixels"]])
"""),
    code("""
cams = I.camera_stats(ds)
display(cams.groupby("view")[["yaw", "yaw_error", "pitch", "face_px_source", "megapixels", "distance_mm"]]
        .describe().T.round(1))
fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
for v, ax in zip(("right45", "front", "left45"), axs):
    cams[cams.view == v].yaw.hist(ax=ax, bins=30); ax.set_title(f"{v}: actual yaw (deg)")
plt.show()
CHECKS["all views within ±15 deg of target"] = bool((cams.yaw_error.abs() <= C.VIEW_MAX_YAW_ERROR).all())
CHECKS["no crop upscaled (source face >= 480 px)"] = bool((cams.face_px_source >= C.FACE_PX).all())
CHECKS["no black padding in crops"] = int(cams.padded.sum()) == 0
CHECKS["face width in crop 480-560 px"] = bool(cams.face_px.between(C.FACE_PX - 1, 560).all())
"""),
    md("""
**Reading this.** `face_px_source` is the face width in the original photo. It must be at least 480 px, so the
768 px crop is always a *downscale* of a real photo, never an upscale. When a person sat close to a portrait
camera, the 1.6 × jaw-width square does not fit in the photo. The largest square that fits is used instead, so
the face is a little larger in the crop (`face_px` up to about 550). Degradations scale with each image's
`face_px`. The rig's left cameras sit near 50° (a
few near 37°), so `left45` is really "left 37-52°". Camera distance is about 1.6 m.
"""),
    md("""
## 4. Units: is the 3D really in millimetres?

Typical adult values:
- distance between the eye centres (mean of each eye's 6 landmarks): about 60-72 mm;
- landmark 0 to 16 (FaceScape puts them near the ears, so wider than a jaw): about 140-170 mm;
- nose tip to chin: about 60-75 mm.

Each row below is one person's neutral TU mesh.
"""),
    code("""
units = I.units_table(ds)
display(units.describe().round(1))
CHECKS["inter-ocular distance 50-75 mm for everyone"] = bool(units.inter_ocular_mm.between(50, 75).all())
CHECKS["camera distance 1.2-2.5 m"] = bool(units.camera_distance_mm.between(1200, 2500).all())
"""),
    md("""
## 5. Does the true 3D face land on the photo?

Each GT mesh is projected into its own crop:
- green dots: every 25th vertex;
- blue dots: the 68 landmarks.

They should sit on the eyes, nose, mouth and jaw in every view and expression. Subject 344 (publishable) is
shown with all 20 expressions.
"""),
    code("""
grid = I.overlay_grid(ds, C.PILOT_SUBJECT, list(C.EXPRESSIONS), size=220)
for i in range(0, grid.shape[0], 220 * 5):
    plt.figure(figsize=(12, 18)); plt.imshow(grid[i:i + 220 * 5]); plt.axis("off"); plt.show()
"""),
    md("""
### 5b. Measured alignment against the real scan

The toolkit's alignment table is not perfect: it warns of "minor misalignment". We measure it against the raw
multi-view scan and then remove it with rigid ICP (`refine` stage).

The table below is computed live:
- *before* = distance from TU to scan with the toolkit's placement;
- *after* = residual once ICP removes the rigid offset.

After ICP, the remaining difference is the TU registration itself (expected ≈0.2-0.4 mm).
"""),
    code("""
rows = [I.mview_alignment(src, s, e) for s, e in [(340, 1), (340, 3), (325, 1), (339, 19), (344, 1), (350, 2)]]
display(pd.DataFrame(rows))
ref_path = ds.root / "refine.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path); done = ref[ref.status == "refined"]
    display(ref.status.value_counts().to_frame("tuples"))
    if len(done):
        display(done[["median_before_mm", "median_after_mm", "p95_after_mm", "rotation_deg", "translation_mm"]].describe().round(2))
        CHECKS["refined residual median < 0.5 mm"] = bool(done.median_after_mm.median() < 0.5)
    CHECKS["camera.json refined (share of tuples)"] = f"{len(done)}/{len(ref)}"
else:
    CHECKS["camera.json refined (share of tuples)"] = "refine stage not run"
"""),
    md("""
## 6. Integrity: every file present and unchanged

Every image listed in the manifests must exist and match its SHA-256 hash. Every camera and GT file must
exist. A FAIL here means the folder is corrupt; do not upload it.
"""),
    code("""
t0 = time.time()
table, problems = I.integrity(ds, hash_every=1)
display(table)
print(f"checked in {time.time() - t0:.0f}s; problems: {len(problems)}")
if problems: display(pd.DataFrame(problems).head(20))
CHECKS["all files present and hashes match"] = not problems
display(I.sizes(ds))
issues = pd.read_csv(ds.root / "issues.csv"); display(issues.head(20))
"""),
    md("""
## 7. Degradations

The 14 profiles at 3 severities on one publishable photo, with the clean photo first in each row:
- **C01-C10:** real capture problems;
- **R1-R4:** the degradation models that the restorers (GFPGAN/CodeFormer, Real-ESRGAN, DA-CLIP) were trained on.

Small images (low resolution, messaging chain, R1, R2) are shown enlarged with square pixels. That is what the
model actually receives.
"""),
    code("""
sheet = I.degraded_sheet(ds, f"{C.PILOT_SUBJECT}_02_front", size=150)
plt.figure(figsize=(16, 22)); plt.imshow(sheet); plt.axis("off"); plt.show()
"""),
    md("""
**Input quality.** The table gives PSNR (dB) of each degraded image against its clean photo, averaged over
test images. Lower means more damage. Every profile should get worse from S1 to S3.
"""),
    code("""
q = I.input_quality(ds, split="test", max_images=60)
display(q)
CHECKS["PSNR falls S1 > S2 > S3 for every profile"] = bool(((q[1] > q[2]) & (q[2] > q[3])).all())
"""),
    md("""
**Cameras of small images.** When a profile shrinks the image, the camera must shrink with it. The landmarks
projected with the degraded image's camera must match the clean projection rescaled: difference ≈ 0 px.
"""),
    code("""
pc = I.degraded_projection_check(ds, "C01_s1")
display(pc)
CHECKS["degraded-image cameras consistent (< 0.01 px)"] = bool(len(pc) and pc.max_px_diff.max() < 0.01)
"""),
    md("""
## 8. Training mix for GARD-Face

Each training photo gets 3 random (profile, severity) draws, and all three views are mixed, so the model sees
every angle with every kind of damage. This table counts training images per profile and view. The counts
should be roughly even.
"""),
    code("""
mix = I.train_mix(ds); display(mix)
if len(mix):
    CHECKS["every profile appears in every training view"] = bool((mix > 0).all().all())
"""),
    md("## 9. All checks"),
    code("""
display(I.checks_table(CHECKS))
"""),
    md("""
**What this shows.** If every row reads PASS, then three things hold:
- every photo is real, undistorted and correctly cropped;
- its true 3D face is in millimetres and placed in the photo's camera;
- the degradations behave as intended.

Rows that show numbers instead (missing archives, refined share) are status, not failures. They shrink to zero
as the remaining downloads arrive and the build is rerun.
"""),
]

out = save(ROOT / "notebooks" / "30_dataset_inspection.ipynb", cells)
print("wrote", out)
