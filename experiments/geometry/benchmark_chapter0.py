"""Complete the five-model NoW geometry benchmark from the frozen pilot."""

from collections import Counter
from pathlib import Path
import json
from statistics import median

from src.benchmark import (
    DEGRADED, MODEL_IDS, ORIGINAL, model_specs, pair_variants, result_records,
    run_matrix, write_table,
)
from src.data_now import installed_cases, pilot_cases
from src.reporting import accuracy_rows, cost_rows


ROOT = Path("experiments/geometry/runs")
OUTPUT = ROOT / "chapter0-geometry"
CONDITIONS = ("face-32", "blur-heavy")


def require_compatible(pairs):
    incompatible = {"no_original", "model_config_mismatch", "reference_mismatch",
                    "source_request_mismatch", "protocol_mismatch"}
    bad = [row for row in pairs if row["pair_status"] in incompatible]
    if bad:
        raise ValueError(f"Incompatible retained baseline for {bad[0]}")


def summarize_pairs(pairs):
    """Show paired geometry error and all-request coverage by model and condition."""
    rows = []
    for model in MODEL_IDS:
        for condition in dict.fromkeys(row["condition"] for row in pairs):
            matches = [row for row in pairs if row["model"] == model
                       and row["condition"] == condition]
            if not matches:
                continue
            scored = [row for row in matches if row["pair_status"] == "scored"]
            means = {key: sum(row[key] for row in scored) / len(scored) if scored else None
                     for key in ("original_mm", "degraded_mm", "delta_mm")}
            rows.append({"model": model, "condition": condition, **means,
                         "scored": len(scored), "requested": len(matches),
                         "worse": sum(row["delta_mm"] > 0 for row in scored)})
    return rows


def common_scored_pairs(pairs, models):
    """Keep each photo/condition only when every named baseline has both scores."""
    scores = {}
    for row in pairs:
        if row["pair_status"] == "scored":
            scores.setdefault((row["image"], row["condition"]), set()).add(row["model"])
    required = set(models)
    return [row for row in pairs
            if row["model"] in required
            and required <= scores.get((row["image"], row["condition"]), set())]


def main():
    cases = installed_cases()
    development = pilot_cases(cases)
    specs = model_specs()
    variants = tuple(next(v for v in DEGRADED if v.name == name) for name in CONDITIONS)
    original = result_records(ROOT / "chapter1-original-scan-local-01")
    original += result_records(ROOT / "chapter1-development-original-scan-local-01")
    degraded = result_records(ROOT / "chapter1-degraded-scan-local-01")
    degraded += result_records(ROOT / "chapter1-development-degraded-scan-local-01")
    current = {spec.model_id: spec.config_id for spec in specs}
    if any(current[row.model_id] != row.model_config_id for row in original[5:]):
        raise ValueError("Run with the same geometry worker configuration as the two-subject pilot")

    extra = run_matrix(development[1:], specs, variants, OUTPUT / "development", timeout=1500)
    development_pairs = pair_variants(degraded + extra, original)
    require_compatible(development_pairs)
    write_table(OUTPUT / "development-pairs.csv", development_pairs)

    # Freeze the decision before starting held-out inference. Prefer the
    # stronger ViT-S signal because its prior interface supports correction.
    decision = OUTPUT / "decision.json"
    if decision.is_file():
        choice = json.loads(decision.read_text(encoding="utf-8"))
    else:
        signal = {}
        for name in CONDITIONS:
            deltas = [p["delta_mm"] for p in development_pairs
                      if p["model"] == "vits" and p["condition"] == name
                      and p["delta_mm"] is not None]
            if len(deltas) == len(development):
                signal[name] = median(deltas)
        if not signal:
            raise ValueError("No complete three-person ViT-S comparison for condition selection")
        primary = max(signal, key=signal.get)
        choice = {"primary": primary, "vits_median_delta_mm": signal}
        decision.write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    print("Frozen primary:", choice, flush=True)
    variant = next(v for v in variants if v.name == choice["primary"])

    used = {case.subject_id for case in development}
    neutral = [case for case in cases if case.subject_id not in used
               and case.scan_landmark_path is not None and case.category == "multiview_neutral"]
    covered = pilot_cases(neutral, len({case.subject_id for case in neutral}))
    shared = run_matrix(covered[:6], specs, (ORIGINAL, variant), OUTPUT / "holdout-shared", timeout=1500)
    core = tuple(spec for spec in specs if spec.model_id != "pixel3dmm")
    remaining = run_matrix(covered[6:], core, (ORIGINAL, variant), OUTPUT / "holdout-remaining", timeout=900)
    holdout = shared + remaining
    pairs = pair_variants([r for r in holdout if r.variant_id != "original"], holdout)
    require_compatible(pairs)
    write_table(OUTPUT / "holdout-pairs.csv", pairs)
    write_table(OUTPUT / "shared-five-model-pairs.csv", pair_variants(
        [r for r in shared if r.variant_id != "original"], shared))
    write_table(OUTPUT / "scores.csv", accuracy_rows(original + degraded + extra + holdout))
    write_table(OUTPUT / "cost.csv", cost_rows(original + degraded + extra + holdout))
    report = {
        "decision": choice,
        "development": dict(Counter(p["pair_status"] for p in development_pairs)),
        "holdout": dict(Counter(p["pair_status"] for p in pairs)),
        "holdout_identities": len(covered),
        "shared_identities": len(covered[:6]),
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
