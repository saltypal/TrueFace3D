"""Download originals, verify PDF structure, and preserve page-level source evidence."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]

def download(entry):
    destination = ROOT / "pdfs" / (entry["id"] + ".pdf")
    result = dict(entry)
    try:
        if not destination.exists():
            response = requests.get(entry["url"], timeout=(20, 120))
            response.raise_for_status()
            if not response.content.startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF")
            destination.write_bytes(response.content)
            result["resolved_url"] = response.url
        document = pymupdf.open(destination)
        result.update(status="downloaded", pages=len(document),
                      bytes=destination.stat().st_size,
                      sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      pdf_metadata=document.metadata)
        pages = [{"page": number + 1, "text": page.get_text()}
                 for number, page in enumerate(document)]
        (ROOT / "evidence" / (entry["id"] + ".pages.json")).write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        result["first_page_excerpt"] = pages[0]["text"][:1200]
        print(f'{entry["id"]}: {len(document)} pages', flush=True)
        document.close()
    except Exception as error:
        result.update(status="blocked", error=str(error))
        print(f'{entry["id"]}: BLOCKED {error}', flush=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--retry-blocked", action="store_true")
    selection.add_argument("--ids", nargs="+", help="Download/audit only these stable corpus IDs")
    args = parser.parse_args()
    corpus = json.loads((ROOT / "evidence/corpus.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "evidence/download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    previous_by_id = {item["id"]: item for item in previous}
    if args.retry_blocked:
        selected = [entry for entry in corpus
                    if previous_by_id.get(entry["id"], {}).get("status") == "blocked"]
    elif args.ids:
        unknown = set(args.ids) - {entry["id"] for entry in corpus}
        if unknown:
            parser.error("Unknown IDs: " + ", ".join(sorted(unknown)))
        selected = [entry for entry in corpus if entry["id"] in args.ids]
    else:
        selected = corpus
    if not selected:
        print("No blocked entries to retry; existing manifest preserved.", flush=True)
        return
    results_by_id = dict(previous_by_id)
    # arXiv requests are paced; independent publisher endpoints can overlap.
    publisher = [entry for entry in selected if not entry["arxiv"]]
    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(download, publisher):
            results_by_id[result["id"]] = result
    for entry in selected:
        if entry["arxiv"]:
            already_local = (ROOT / "pdfs" / (entry["id"] + ".pdf")).exists()
            results_by_id[entry["id"]] = download(entry)
            if not already_local:
                time.sleep(3)
        manifest_path.write_text(
            json.dumps(sorted(results_by_id.values(), key=lambda item: item["id"]), indent=2,
                       ensure_ascii=False), encoding="utf-8")
    print("Downloaded:", sum(item["status"] == "downloaded" for item in results_by_id.values()),
          "of", len(corpus), flush=True)

if __name__ == "__main__":
    main()
