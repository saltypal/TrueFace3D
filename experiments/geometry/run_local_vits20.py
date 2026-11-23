"""Local (MX330) RealDenseFace degradation benchmark: 20 NoW people x 5 conditions. Usage: python -m experiments... MODEL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark import KAGGLE_VARIANTS, model_specs, run_matrix, summarize_runs
from src.data_now import installed_cases, pilot_cases

model = sys.argv[1] if len(sys.argv) > 1 else "vits"
cases = pilot_cases(installed_cases(), count=20)
out = Path(f"experiments/geometry/runs/local-{model}-20")
rows = run_matrix(cases, model_specs([model]), KAGGLE_VARIANTS, out, timeout=300)
for row in summarize_runs(rows):
    print(row)
