"""Render selected original PDF pages for visual architecture inspection."""
from pathlib import Path
import pymupdf

root = Path(__file__).resolve().parents[1]
for name, pages in {
    "01_RealDenseFace": [3, 5],
    "07_DECA": [3, 4],
    "12_OpenDelight": [5],
    "13_MARCUS": [4, 9],
    "21_EyeCapture": [3],
    "22_ParametricEyes": [3, 5],
    "19_S3Face": [3, 4],
    "31_NPHM": [4],
    "40_SFDM": [4],
    "41_MonocularAppearance": [3],
    "43_ReEnFP": [4],
}.items():
    document = pymupdf.open(root / "pdfs" / (name + ".pdf"))
    for number in pages:
        page = document[number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35), alpha=False)
        pixmap.save(root / "figures" / f"{name}_page_{number}.png")
