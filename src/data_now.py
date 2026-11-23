"""Local NoW validation cases and their reference data."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PILOT_IMAGE = "FaMoS_180424_03335_TA/multiview_neutral/IMG_0041.jpg"


@dataclass(frozen=True)
class NoWCase:
    image_key: str
    subject_id: str
    category: str
    image_path: Path
    box_path: Path
    scan_path: Path
    scan_landmark_path: Path | None


def _by_subject(root, suffix):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    index = {}
    for path in sorted(root.rglob("*" + suffix)):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Reference points outside its dataset root: {path}")
        matches = [part for part in path.relative_to(root).parts[:-1] if part.startswith("FaMoS_")]
        if not matches and path.stem.startswith("FaMoS_"):
            matches = [path.stem]
        if len(matches) != 1 or matches[0] in index:
            raise ValueError(f"Ambiguous NoW reference: {path}")
        index[matches[0]] = path
    return index


def _contained_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes NoW dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_cases(photo_root, box_root, list_path, scan_root, landmark_root=None):
    photo_root, box_root = map(lambda value: Path(value).resolve(), (photo_root, box_root))
    scans = _by_subject(scan_root, ".obj")
    landmarks = {} if landmark_root is None else _by_subject(landmark_root, ".pp")
    cases = []
    seen = set()
    for line in Path(list_path).read_text(encoding="utf-8-sig").splitlines():
        key = line.strip()
        if not key:
            continue
        path = PurePosixPath(key)
        if (len(path.parts) != 3 or path.is_absolute() or
                any(part in {".", ".."} or ":" in part or "\\" in part for part in path.parts) or
                path.as_posix() != key or path.suffix.lower() not in {".jpg", ".jpeg"}):
            raise ValueError(f"Invalid NoW image path: {key}")
        if key in seen:
            raise ValueError(f"Duplicate NoW image path: {key}")
        seen.add(key)
        subject, category, _ = path.parts
        if subject not in scans:
            raise FileNotFoundError(f"No reference scan for {subject}")
        cases.append(NoWCase(
            key, subject, category, _contained_file(photo_root, path),
            _contained_file(box_root, path.with_suffix(".npy")), scans[subject], landmarks.get(subject)
        ))
    if not cases:
        raise ValueError(f"Empty NoW list: {list_path}")
    return cases


def installed_cases(data_root=PROJECT / "Data", list_path=None, landmark_root=None):
    data = Path(data_root)
    release = data / "NoWDataset/NoW_Dataset/final_release_version"
    if list_path is None:
        choices = (data / "imagepathsvalidation.txt", Path.home() / "Downloads/imagepathsvalidation.txt")
        list_path = next((path for path in choices if path.is_file()), choices[0])
    if landmark_root is None:
        candidates = [name for name in data.iterdir() if name.is_dir()
                      and ("lmk" in name.name.lower() or "landmark" in name.name.lower())]
        if len(candidates) > 1:
            raise ValueError("Multiple scan landmark roots found; specify the intended directory")
        landmark_root = next(iter(candidates), None)
    return load_cases(release / "iphone_pictures", release / "detected_face",
                      list_path, data / "scans/scans", landmark_root)


def pilot_cases(cases, count=3):
    neutral = sorted((case for case in cases if case.category == "multiview_neutral"),
                     key=lambda case: (case.image_key != PILOT_IMAGE, case.image_key))
    chosen = {}
    for case in neutral:
        chosen.setdefault(case.subject_id, case)
        if len(chosen) == count:
            break
    if not 1 <= count <= len(chosen):
        raise ValueError(f"Requested {count} neutral subjects; found {len(chosen)}")
    return list(chosen.values())


def load_face_box(path, image_wh):
    """Read the trusted official Python 2 crop-dictionary format."""
    width, height = image_wh
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    try:
        values = np.load(path, allow_pickle=True, encoding="latin1").item()
        box = tuple(float(values[key]) for key in ("left", "top", "right", "bottom"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid NoW crop dictionary: {path}") from exc
    if not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"Invalid crop box: {path}")
    if max(box[0], 0) >= min(box[2], width) or max(box[1], 0) >= min(box[3], height):
        raise ValueError(f"Crop box does not intersect image: {path}")
    return box


def load_scan_landmarks(path):
    try:
        points = [item for item in ET.parse(path).getroot().iter("point") if item.get("active", "1") == "1"]
        coordinates = np.asarray([[float(item.attrib[axis]) for axis in "xyz"] for item in points], dtype=float)
    except (ET.ParseError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid NoW scan landmark annotation: {path}") from exc
    if coordinates.shape != (7, 3) or not np.isfinite(coordinates).all():
        raise ValueError(f"Expected seven finite ordered NoW points: {path}")
    return coordinates


REAL_WORLD_CATEGORIES = ("multiview_expressions", "multiview_occlusions", "selfie")


def real_world_cases(cases, categories=REAL_WORLD_CATEGORIES):
    """All validation images in the in-the-wild categories, sorted by image key."""
    wanted = set(categories)
    return sorted((case for case in cases if case.category in wanted), key=lambda case: case.image_key)
