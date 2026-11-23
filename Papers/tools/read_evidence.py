"""Print small, page-cited windows from already archived paper evidence."""
import argparse
import html
import json
import re
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="+")
parser.add_argument("--terms", default="Training data|Training dataset|Datasets|Dataset|Implementation details|Ablation")
parser.add_argument("--budget", type=int, default=5200)
parser.add_argument("--minpage", type=int, default=1)
parser.add_argument("--pages")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for identifier in args.ids:
    source = root / "evidence" / ("alpha_" + identifier + ".txt")
    if source.exists() and "_" not in identifier:
        raw = source.read_text(encoding="utf-8")
        pages = [(int(number), html.unescape(content)) for number, content in
                 re.findall(r'<page num="(\d+)">(.*?)</page>', raw, re.S)]
    else:
        matches = list((root / "evidence").glob(identifier + "*.pages.json"))
        pages = [(item["page"], item["text"]) for item in
                 json.loads(matches[0].read_text(encoding="utf-8"))] if matches else []
    print("\nPAPER", identifier)
    used = 0
    for number, content in pages:
        if number < args.minpage:
            continue
        if args.pages:
            selected = {int(value) for value in args.pages.split(",")}
            if number in selected:
                print("\nPAGE", number, content[:args.budget])
            continue
        previous_end = 0
        for match in re.finditer(args.terms, content, re.I):
            if match.start() < previous_end:
                continue
            start, end = max(0, match.start()-120), min(len(content), match.start()+1700)
            if used + end - start > args.budget:
                break
            print("\nPAGE", number, content[start:end])
            used += end - start
            previous_end = end
        if used >= args.budget - 1700:
            break
