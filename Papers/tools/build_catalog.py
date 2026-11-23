"""Build the paper index and audit the local research archive without network calls.

Run from any directory. Originals remain unchanged; generated catalog/manifest/report
files describe the local snapshot. A failed title/link/integrity check exits nonzero.
"""
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "2026-10-10"

# Titles were checked against the original PDFs, not title-search retrievals.
TITLES = {
    "01_RealDenseFace": "RealDenseFace: Real-time Monocular 3D Face Reconstruction from Dense UV-space Priors",
    "02_FLAME": "Learning a model of facial shape and expression from 4D scans",
    "03_MICA": "Towards Metrical Reconstruction of Human Faces",
    "04_Pixel3DMM": "Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction",
    "05_FlowFace": "3D Face Tracking from 2D Video through Iterative Dense UV to Image Flow",
    "06_SMIRK": "SMIRK: 3D Facial Expressions through Analysis-by-Neural-Synthesis",
    "07_DECA": "Learning an Animatable Detailed 3D Face Model from In-The-Wild Images",
    "08_EMOCA": "EMOCA: Emotion Driven Monocular Face Capture and Animation",
    "09_HRN": "A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images",
    "10_HiFace": "HiFace: High-Fidelity 3D Face Reconstruction by Learning Static and Dynamic Details",
    "11_MeshLAM": "MeshLAM: Feed-Forward One-Shot Animatable Textured Mesh Avatar Reconstruction",
    "12_OpenDelight": "Learning a Delighting Prior for Facial Appearance Capture in the Wild",
    "13_MARCUS": "Monocular Avatar Reconstruction via Cascaded Diffusion Priors and UV-Space Differentiable Shading",
    "14_MoSAR": "MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading",
    "15_Relightify": "Relightify: Relightable 3D Faces from a Single Image via Diffusion Models",
    "16_AvatarMePlusPlus": "AvatarMe++: Facial Shape and BRDF Inference with Photorealistic Rendering-Aware GANs",
    "17_ID2Reflectance": "Monocular Identity-Conditioned Facial Reflectance Reconstruction",
    "18_ReflectanceMM": "Learning a 3D Morphable Face Reflectance Model from Low-cost Data",
    "19_S3Face": "S3-Face: SSS-Compliant Facial Reflectance Estimation via Diffusion Priors",
    "20_HAFMat": "HAFMat: Hybrid Priors Guided Adaptive Fusion for Single-Image Human Material Estimation",
    "21_EyeCapture": "High-Quality Capture of Eyes",
    "22_ParametricEyes": "Lightweight Eye Capture Using a Parametric Model",
    "23_EyeNeRF": "High-Fidelity Eye Animatable Neural Radiance Fields for Human Face",
    "24_AudioEar": "AudioEar: Single-View Ear Reconstruction for Personalized Spatial Audio",
    "25_Nvdiffrast": "Modular Primitives for High-Performance Differentiable Rendering",
    "26_Mitsuba2": "Mitsuba 2: A Retargetable Forward and Inverse Renderer",
    "27_DisneyBRDF": "Physically Based Shading at Disney",
    "28_SubsurfaceTransport": "A Practical Model for Subsurface Light Transport",
    "29_FaceScape": "FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction",
    "30_NeRSemble": "NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads",
    "31_NPHM": "Learning Neural Parametric Head Models",
    "32_Multiface": "Multiface: A Dataset for Neural Face Rendering",
    "33_H3DNet": "H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction",
    "34_RingNet_NoW": "Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision",
    "35_CodecAvatarStudio_AVA256": "Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars",
    "36_CodeFormer": "Towards Robust Blind Face Restoration with Codebook Lookup Transformer",
    "37_GFPGAN": "Towards Real-World Blind Face Restoration with Generative Facial Prior",
    "38_RealESRGAN": "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data",
    "39_MonoNPHM": "MonoNPHM: Dynamic Head Reconstruction from Monocular Videos",
    "40_SFDM": "SFDM: Robust Decomposition of Geometry and Reflectance for Realistic Face Rendering from Sparse-view Images",
    "41_MonocularAppearance": "Monocular Facial Appearance Capture in the Wild",
    "42_BecomingLit": "BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading",
    "43_ReEnFP": "ReEnFP: Detail-Preserving Face Reconstruction by Encoding Facial Priors",
    "44_Pix2NPHM": "Pix2NPHM: Learning to Regress NPHM Reconstructions From a Single Image",
}

NON_ARXIV_YEARS = {
    "02_FLAME": "2017", "19_S3Face": "2025", "21_EyeCapture": "2014",
    "22_ParametricEyes": "2016", "26_Mitsuba2": "2019",
    "27_DisneyBRDF": "2012 course notes", "28_SubsurfaceTransport": "2001",
    "35_CodecAvatarStudio_AVA256": "2024", "43_ReEnFP": "2023",
}


def normalize_title(value):
    decomposed = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"[^a-z0-9]", "", decomposed)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    corpus = json.loads((ROOT / "evidence/corpus.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "evidence/download_manifest.json").read_text(encoding="utf-8"))
    previous_by_id = {item["id"]: item for item in previous}
    errors = []
    manifest = []
    note_words = 0
    pdf_pages = 0
    pdf_bytes = 0
    index = [
        "# Every paper in the TrueFace3D archive\n",
        f"Snapshot: **{SNAPSHOT}**. **{len(corpus)} original documents; one teaching note per document.**\n",
        "The year column uses initial arXiv submission year where available, not necessarily the final venue year. Venue/version details are in the note and original. Full titles were checked against the PDFs.\n",
        "[Start here](README.md) · [Master plan](TRUEFACE3D_MASTER_PLAN.md) · [Datasets](DATASETS_AND_TRAINING.md) · [Novelty](NOVELTY_ANALYSIS.md)\n",
        "| ID / method | Original title | Initial year | Area | Read | Original | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in corpus:
        identifier = entry["id"]
        entry["title"] = TITLES[identifier]
        if identifier == "02_FLAME":
            entry["original_attempt_url"] = "https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf"
            entry["url"] = "https://3dvar.com/Li2017Learning.pdf"
            entry["primary_paper_url"] = "https://flame.is.tue.mpg.de/"
            entry["retrieval_note"] = "Author PDF endpoint returned HTML; verified original downloaded from mirror."
        if identifier == "23_EyeNeRF":
            entry["method"] = "DeNeRF"
            entry["filename_note"] = "Stable archive ID retained; this is DeNeRF, not the separate EyeNeRF paper."
        pdf_path = ROOT / "pdfs" / (identifier + ".pdf")
        note_path = ROOT / "notes" / (identifier + ".md")
        evidence_path = ROOT / "evidence" / (identifier + ".pages.json")
        if not pdf_path.exists() or not note_path.exists() or not evidence_path.exists():
            errors.append(f"Missing PDF/note/extracted pages: {identifier}")
            continue
        payload = pdf_path.read_bytes()
        if not payload.startswith(b"%PDF-"):
            errors.append(f"Invalid PDF header: {identifier}")
            continue
        with pymupdf.open(pdf_path) as document:
            count = len(document)
            first_page = document[0].get_text()
            # NFKD also expands typographical ligatures in extracted titles.
            title_match = normalize_title(entry["title"]) in normalize_title(first_page)
            if not title_match:
                errors.append(f"Expected title not found on first PDF page: {identifier}")
            actual_hash = hashlib.sha256(payload).hexdigest()
            old = previous_by_id.get(identifier, {})
            if old.get("sha256") and old["sha256"] != actual_hash:
                errors.append(f"Original PDF hash changed: {identifier}")
            pages = json.loads(evidence_path.read_text(encoding="utf-8"))
            if len(pages) != count or [p["page"] for p in pages] != list(range(1, count + 1)):
                errors.append(f"Page evidence mismatch: {identifier}")
            manifest.append({
                **entry,
                "resolved_url": entry["url"] if identifier == "02_FLAME" else old.get("resolved_url", entry["url"]),
                "status": "downloaded",
                "survey_date": SNAPSHOT,
                "pages": count,
                "bytes": len(payload),
                "sha256": actual_hash,
                "pdf_metadata": document.metadata,
                "first_page_excerpt": first_page[:1200],
                "expected_title_found": title_match,
            })
            pdf_pages += count
            pdf_bytes += len(payload)
        note = note_path.read_text(encoding="utf-8")
        word_count = len(note.split())
        note_words += word_count
        if word_count < 400:
            errors.append(f"Paper note unexpectedly short: {identifier}")
        if f"../pdfs/{identifier}.pdf" not in note:
            errors.append(f"Note does not link its original: {identifier}")
        arxiv = entry.get("arxiv")
        year = "20" + arxiv[:2] if arxiv else NON_ARXIV_YEARS[identifier]
        source = "https://arxiv.org/abs/" + arxiv if arxiv else entry.get("primary_paper_url", entry["url"])
        method = entry.get("method", entry["slug"])
        index.append(
            f"| {identifier[:2]} {method} | {entry['title']} | {year} | {entry['area'].replace('_', ' ')} "
            f"| [Note](notes/{identifier}.md) | [PDF](pdfs/{identifier}.pdf) | [Primary/source]({source}) |"
        )
    index.extend([
        "\nThe filename `23_EyeNeRF` is a retained archive identifier: its original is **DeNeRF (2308.00773)**. Disney2012 is course notes. FLAME's PDF came from a verified mirror after an author endpoint returned HTML; citation remains the official project.\n",
        "PDF pages, SHA-256 hashes, resolved download URLs and original title excerpts: [download manifest](evidence/download_manifest.json). Title-retrieval mismatches: [rejected alphaXiv audit](evidence/REJECTED_ALPHA_MATCHES.md).\n",
        "Availability of an original paper is distinct from availability of model weights, training data, implementation, or authorization to use face images. See each note and the master plan.\n",
    ])
    (ROOT / "PAPER_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    write_json(ROOT / "evidence/corpus.json", corpus)
    write_json(ROOT / "evidence/download_manifest.json", manifest)

    # This generated report is itself linked by README. Create its link-bearing
    # skeleton before auditing, so a first invocation works in a fresh archive.
    (ROOT / "VALIDATION_REPORT.md").write_text(
        "# Research archive validation\n\n"
        "[Manifest](evidence/download_manifest.json) · "
        "[Index](PAPER_INDEX.md) · [Master plan](TRUEFACE3D_MASTER_PLAN.md)\n",
        encoding="utf-8",
    )
    checked_links = 0
    for markdown in ROOT.rglob("*.md"):
        content = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", content):
            target = target.strip("<>").split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            checked_links += 1
            if not (markdown.parent / target).exists():
                errors.append(f"Broken local link in {markdown.relative_to(ROOT)}: {target}")
    note_count = len(list((ROOT / "notes").glob("*.md")))
    pdf_count = len(list((ROOT / "pdfs").glob("*.pdf")))
    if note_count != len(corpus) or pdf_count != len(corpus):
        errors.append("Unpaired corpus/PDF/note counts")
    report = {
        "survey_date": SNAPSHOT,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_count": len(corpus), "pdf_count": pdf_count, "note_count": note_count,
        "original_pdf_pages": pdf_pages, "original_pdf_bytes": pdf_bytes,
        "paper_note_words": note_words, "checked_local_links": checked_links,
        "errors": errors,
        "model_inference_run": False, "training_run": False,
        "datasets_downloaded": False, "external_link_health_checked_by_this_script": False,
    }
    write_json(ROOT / "evidence/validation.json", report)
    outcome = "PASS" if not errors else "FAIL"
    lines = [
        "# Research archive validation\n",
        f"**{outcome}** — source snapshot {SNAPSHOT}. Local audit timestamp: {report['validated_at_utc']}.\n",
        f"- {len(corpus)} catalog entries, {pdf_count} original PDFs, {note_count} individual notes.",
        f"- {pdf_pages} pages across original PDFs; {pdf_bytes / (1024 ** 2):.1f} MiB.",
        f"- {note_words:,} words across individual paper notes.",
        f"- {checked_links} local Markdown links checked.",
        "- PDF headers/openability, expected first-page titles, prior SHA-256 integrity, page evidence and one-to-one pairing checked.",
        "- Key architecture/optics diagrams visually inspected from rendered original pages stored in figures/.",
        "- Nine title-based alphaXiv results rejected; intended papers explained from correctly identified originals.",
        "\n## What this does not validate\n",
        "No reconstruction inference, training, GPU compatibility/performance, dataset/model-weight acquisition, automatic scientific novelty, or finished 3D integration has been tested. External URLs can change; this local validator does not probe them. Source-reported results and illustrative examples are not our results.\n",
        "[Manifest](evidence/download_manifest.json) · [Machine-readable audit](evidence/validation.json) · [Index](PAPER_INDEX.md) · [Master plan](TRUEFACE3D_MASTER_PLAN.md)\n",
    ]
    if errors:
        lines.extend(["## Errors\n", *["- " + error for error in errors]])
    (ROOT / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
