"""Run the five geometry models on one original portrait and its five variants."""

from pathlib import Path

from experiments.geometry.one_photo_geometry import CONDITIONS, RUN
from src.benchmark import DEGRADED, ORIGINAL, model_specs, run_matrix
from src.data_now import installed_cases, pilot_cases


def main():
    variants = {item.name: item for item in (ORIGINAL, *DEGRADED)}
    target = Path("experiments/geometry/runs") / RUN
    cases = pilot_cases(installed_cases(), 1)
    rows = run_matrix(cases, model_specs(), tuple(variants[name] for name in CONDITIONS),
                      target, timeout=1500)
    fitted = sum(row.fit_status == "success" for row in rows)
    scored = sum(row.score_status == "success" for row in rows)
    print(f"Fresh results: {fitted}/{len(rows)} fits, {scored}/{len(rows)} local scan scores.",
          flush=True)


if __name__ == "__main__":
    main()
