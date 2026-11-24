"""Build notebooks/kaggle/13_kaggle_prior_corrector.ipynb (GARD-inspired ViT-S prior corrector demo)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# 13 - GARD-inspired prior corrector for ViT-S (NoW, 20 people, cross-validated)

Train a small residual network that maps ViT-S's degraded per-vertex priors to the clean photo's priors
(GARD's idea: correct the intermediate representation, conditioned on the degradation, instead of the pixels).
Folds are person-disjoint. Then refit ViT-S with corrected priors (all 5 conditions) and with clean priors
(oracle upper bound), and report the % of the clean->degraded scan-error gap recovered."""),
    code("""REPO_URL = "https://github.com/saltypal/TrueFace3D.git"
import os, subprocess, sys
REPO = "/kaggle/working/TrueFace3D"
if os.path.isdir(REPO + "/.git"):
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
print("commit", subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
os.chdir(REPO); sys.path.insert(0, REPO)"""),
    code("""from src.kaggle import setup
report = setup.main(['--skip-appearance'])
assert all(v in ('installed from wheel', 'already importable', 'built') for v in report['builds'].values()), report['builds']"""),
    code("""import glob, json
from pathlib import Path
from src.correction import prior_corrector as pc
ROOT = Path(next(p for p in glob.glob('/kaggle/input/**/out_A/runs', recursive=True)))
OUT = Path('/kaggle/working/out_PC'); OUT.mkdir(exist_ok=True)
data = pc.load_vits_priors(ROOT)
subjects = sorted({s for s, v in data if v == 'original'})
print(len(data), 'ViT-S priors;', len(subjects), 'people')
assert len(subjects) >= 15, 'priors missing from the attached dataset'"""),
    md("## 1. Train (5 person-disjoint folds) and check the prior error"),
    code("""import collections, numpy as np, time
t = time.time()
rows = pc.cross_validate(data, folds=5, out_dir=OUT / 'corrected', epochs=1500)
print(f'trained 5 folds in {time.time() - t:.0f}s')
(OUT / 'prior_errors.json').write_text(json.dumps(rows, indent=1))
g = collections.defaultdict(list)
for r in rows: g[r['variant']].append(r)
for v in pc.VARIANTS:
    rr = g[v]; cd = np.mean([r['coord_err_degraded'] for r in rr]); cc = np.mean([r['coord_err_corrected'] for r in rr])
    dd = np.mean([r['depth_err_degraded'] for r in rr]); dc = np.mean([r['depth_err_corrected'] for r in rr])
    print(f"{v:9} coord {cd:.4f} -> {cc:.4f}   depth {dd:.4f} -> {dc:.4f}")"""),
    md("## 2. Refit ViT-S with corrected priors and with clean priors (oracle)"),
    code("""from src.kaggle import runner
jobs = pc.refit_jobs(data, OUT / 'corrected', OUT)
print(len(jobs), 'refit jobs:', collections.Counter(j['arm'] for j in jobs))
# smoke test: one corrected job must produce a scored fit before launching the rest
pc.run_refit_job(jobs[0])
smoke = pc.collect_scores(OUT / 'runs', arm_from_run=True)
assert smoke, 'smoke refit produced no score'
print('smoke ok:', smoke)"""),
    code("""codes = runner.launch('src.correction.prior_corrector:run_refit_job', jobs[1:], OUT / 'logs', cwd=REPO)
print('exit codes', codes)"""),
    md("## 3. Result: % of the scan-error gap recovered"),
    code("""import pandas as pd
baseline = pc.collect_scores(ROOT)
arms = pc.collect_scores(OUT / 'runs', arm_from_run=True)
table = pc.gap_table(baseline, arms)
(OUT / 'gap_table.json').write_text(json.dumps(table, indent=1))
df = pd.DataFrame(table).set_index('variant')
print('corrected fits:', sum(1 for k in arms if k[0] == 'corrected'), ' oracle fits:', sum(1 for k in arms if k[0] == 'oracle'))
assert sum(1 for k in arms if k[0] == 'corrected') > 50, 'too few corrected fits scored'
df[['n', 'clean', 'baseline', 'corrected', 'oracle', 'corrected_gap_recovered_pct', 'oracle_gap_recovered_pct']].round(3)"""),
    code("""import shutil
shutil.make_archive('/kaggle/working/out_PC', 'zip', OUT)
print('zip written')"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "notebooks/kaggle/13_kaggle_prior_corrector.ipynb")
print("written")
