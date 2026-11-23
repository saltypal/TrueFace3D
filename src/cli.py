"""TrueFace3D commands shared by the terminal and executed notebooks."""

import argparse
import json
from pathlib import Path
import sys

from .geometry.backend import UPSTREAM, audit_backend, run_upstream
from .geometry.artifacts import save_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--model", choices=("vits", "vitb"), default="vits")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    for name in ("native", "reconstruct"):
        action = sub.add_parser(name)
        action.add_argument("--input", type=Path, required=True)
        action.add_argument("--output", type=Path, required=True)
        action.add_argument("--python", default=sys.executable)
        action.add_argument("--device", default="cuda")
        action.add_argument("--precision", choices=("native", "fp32-compat"), default="native")

    inventory = sub.add_parser("inventory-now")
    benchmark = sub.add_parser("benchmark")
    rescore = sub.add_parser("rescore-now")
    for action in (inventory, benchmark, rescore):
        action.add_argument("--data-root", type=Path)
        action.add_argument("--image-list", type=Path)
        action.add_argument("--landmark-root", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--mode", choices=("original", "degraded"), required=True)
    benchmark.add_argument("--models", nargs="+", choices=("vits", "vitb", "deca", "pixel3dmm", "mica"))
    benchmark.add_argument("--subjects", type=int, default=1)
    benchmark.add_argument("--primary-only", action="store_true")
    benchmark.add_argument("--budget-seconds", type=float)
    benchmark.add_argument("--timeout", type=float, default=900)
    rescore.add_argument("--input-run", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    rescore.add_argument("--retry-run", type=Path, action="append", default=[])
    for action in (benchmark, rescore):
        action.add_argument("--protocol", choices=("local_now_mask_similarity_icp",
                                                    "local_now_mask_fixed_scale_icp",
                                                    "official_now_nometrical",
                                                    "official_now_metrical"),
                            default="local_now_mask_similarity_icp")
        action.add_argument("--official-python")
    regenerate = sub.add_parser("regenerate")
    regenerate.add_argument("--bundle", type=Path, required=True)
    regenerate.add_argument("--output", type=Path, required=True)
    regenerate.add_argument("--python", default=sys.executable)
    regenerate.add_argument("--coefficient", type=int)
    regenerate.add_argument("--value", type=float)

    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_backend(args.upstream.resolve(), args.model)
            print(json.dumps(report, indent=2))
            return int(any(item["sha256"] is None for item in report["assets"].values()))
        if args.command in {"inventory-now", "benchmark", "rescore-now"}:
            from .data_now import PROJECT, installed_cases, pilot_cases
            cases = installed_cases(args.data_root or PROJECT / "Data", args.image_list, args.landmark_root)
            if args.command == "inventory-now":
                print(json.dumps({
                    "images": len(cases), "subjects": len({case.subject_id for case in cases}),
                    "images_with_scan_landmarks": sum(case.scan_landmark_path is not None for case in cases),
                    "pilot": [case.image_key for case in pilot_cases(cases)],
                }, indent=2))
                return 0
            from .benchmark import DEGRADED, MODEL_IDS, ORIGINAL, model_specs, rescore_run, run_matrix, summarize_runs
            from .evaluation import Evaluator
            scorer = None
            if args.protocol.startswith("official"):
                if not args.official_python:
                    raise ValueError("Pass the interpreter in the qualified official evaluator environment")
                from .now_official import OfficialNoW
                scorer = OfficialNoW(args.official_python, PROJECT / "third_party/now_evaluation",
                                     PROJECT / "experiments/geometry/runs/official-now-tmp")
            if args.command == "rescore-now":
                rows = rescore_run(args.input_run, args.output, cases, evaluator=Evaluator(scorer),
                                   protocol=args.protocol, retry_dirs=args.retry_run)
                print(json.dumps(summarize_runs(rows, args.protocol), indent=2))
                return int(not any(row.score_status == "success" for row in rows))
            variants = (ORIGINAL,) if args.mode == "original" else DEGRADED[:1] if args.primary_only else DEGRADED
            rows = run_matrix(pilot_cases(cases, args.subjects), model_specs(args.models or MODEL_IDS),
                              variants, args.output, evaluator=Evaluator(scorer), protocol=args.protocol,
                              budget_seconds=args.budget_seconds, timeout=args.timeout)
            print(json.dumps(summarize_runs(rows, args.protocol), indent=2))
            return int(not any(row.fit_status == "success" for row in rows))
        if args.command == "regenerate":
            import subprocess
            worker = Path(__file__).parent / "geometry/worker.py"
            command = [args.python, str(worker), "--regenerate", str(args.bundle.resolve()), "--output",
                       str(args.output.resolve())]
            if (args.coefficient is None) != (args.value is None):
                raise ValueError("Provide both the expression coefficient and its value")
            if args.coefficient is not None:
                command += ["--coefficient", str(args.coefficient), "--value", str(args.value)]
            return subprocess.run(command, cwd=args.upstream.resolve(), check=False).returncode

        capture = args.command == "reconstruct"
        output = args.output.resolve()
        # Keep the existing native invocation path, even for callers using older worker signatures.
        precision = {"precision": args.precision} if args.precision != "native" else {}
        run_upstream(args.python, args.upstream.resolve(), args.input.resolve(), output,
                     args.model, args.device, capture, **precision)
        if capture:
            save_bundle(output)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
