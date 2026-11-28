"""Fixed choices for FaceScape-C. Change them here only."""

EXPRESSIONS = {
    1: "neutral", 2: "smile", 3: "mouth_stretch", 4: "anger", 5: "jaw_left", 6: "jaw_right",
    7: "jaw_forward", 8: "mouth_left", 9: "mouth_right", 10: "dimpler", 11: "chin_raiser",
    12: "lip_puckerer", 13: "lip_funneler", 14: "sadness", 15: "lip_roll", 16: "grin",
    17: "cheek_blowing", 18: "eye_closed", 19: "brow_raiser", 20: "brow_lower",
}


def expression_dir(index: int) -> str:
    """FaceScape folder/file stem, e.g. 2 -> '2_smile'."""
    return f"{index}_{EXPRESSIONS[index]}"


# Camera yaw targets in degrees, measured in the TU frame (face looks along +Z, +Y up).
# Positive yaw = camera on the subject's left side (+X).
VIEWS = {"front": 0.0, "left45": 45.0, "right45": -45.0}
VIEW_MAX_YAW_ERROR = 15.0      # reject a view if no valid camera is this close to the target
VIEW_MAX_PITCH = 15.0          # ignore cameras looking from far above/below

# Crop: square, CROP_PX wide, with the jaw-to-jaw face width scaled to FACE_PX.
CROP_PX = 768
FACE_PX = 480            # crop = 1.6 x jaw width: fits portrait 12 MP frames without padding
CROP_CENTER_UP = 0.10          # shift the crop centre up by this fraction of face width (forehead room)

# Subjects (photos exist for 1-359; we use 321-359, whose TU models are in trainset 301-400).
PUBLISHABLE = {122, 212, 340, 344, 393, 395, 421, 527, 594, 610}
INVALID = {323}                # info list flags texture and displacement as missing
TEST_SUBJECTS = (340, 344, 350, 325, 339, 352)
PILOT_SUBJECT = 344
TEST_EXPRESSIONS = (1, 2, 3, 19)   # neutral, smile, mouth_stretch, brow_raiser
TRAIN_DEGRADED_PER_IMAGE = 3       # random (profile, severity) draws per clean training photo

BASE_SEED = 20261128
