"""Subject table: age, gender, validity, split and publishability."""
from __future__ import annotations

from pathlib import Path

from .constants import INVALID, PILOT_SUBJECT, PUBLISHABLE, TEST_SUBJECTS


def read_info_list(path: Path) -> dict[int, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            sid = int(parts[0])
            out[sid] = {"subject": sid, "gender": parts[1], "age": None if parts[2] == "-" else int(parts[2]),
                        "info_valid": parts[3] == "1"}
    return out


def subject_table(subjects: list[int], info: dict[int, dict]) -> list[dict]:
    rows = []
    for sid in subjects:
        row = dict(info.get(sid, {"subject": sid, "gender": "-", "age": None, "info_valid": False}))
        excluded = sid in INVALID or not row["info_valid"]
        row["split"] = "excluded" if excluded else ("test" if sid in TEST_SUBJECTS else "train")
        row["pilot"] = sid == PILOT_SUBJECT
        row["publishable"] = sid in PUBLISHABLE
        rows.append(row)
    return rows
