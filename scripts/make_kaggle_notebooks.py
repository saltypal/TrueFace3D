"""Generate the Kaggle and analysis notebooks WITHOUT outputs (run: python scripts/make_kaggle_notebooks.py).

Notebook cells only call modules under src/. Keeping the generator in the repo makes the notebooks reproducible.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CLONE = '''REPO_URL = "https://github.com/saltypal/TrueFace3D.git"   # <- set before running
import os, subprocess, sys
REPO = "/kaggle/working/TrueFace3D"
if os.path.isdir(REPO + "/.git"):
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
print("commit", subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
os.chdir(REPO); sys.path.insert(0, REPO)'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells, folder):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.validate(nb)
    path = ROOT / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path)


readiness = [
    md("# 10 - Kaggle readiness (run on account A and on account B)\nChecks the GPUs, builds the environment, and runs a one-photo smoke test of every mesh model and appearance backend on GPU 0 and GPU 1. The result is `readiness.json` plus the built wheels, saved as notebook output so later sessions can attach them."),
    code(CLONE),
    code("!nvidia-smi"),
    code("from src.kaggle import readiness\nenv = readiness.environment()\nprint(env['torch'])"),
    md("Setup clones the pinned upstream repositories, links weights from the private dataset, and builds nvdiffrast / flame_solver / PyTorch3D for the T4 (sm_75). The first run takes a while; wheels land in `/kaggle/working/wheels`."),
    code("from src.kaggle import setup\nsetup_report = setup.main([])"),
    code("OUT = '/kaggle/working/readiness'\nrows = readiness.run_smoke(OUT, gpus=(0, 1))\ntable = readiness.readiness_table(rows)\nfrom src.reporting import table_markdown\nfrom IPython.display import Markdown\nMarkdown(table_markdown(table))"),
    md("`ready` means the item passed on both GPUs. `not_run` means its code or weights are not available (see the detail column); that is expected for appearance methods whose weights are not in the dataset. `NOT READY` needs attention before the real run."),
    code("readiness.save(OUT, env, rows)\nprint('saved', OUT)"),
]

mesh = [
    md("# 11 - Account A: mesh benchmark\n5 mesh models x 20 subjects x 5 conditions (original, face-64, face-32, face-16, compound), the real-world set (original only), the ViT-S oracle diagnostic, raw projection on the ViT-S meshes, a GLB for every mesh, and the cost table. Everything is written under `/kaggle/working/out_A` and zipped. The run is resumable: re-running skips fits that already succeeded."),
    code(CLONE),
    code("from src.kaggle import setup\nsetup.main([])"),
    code("from pathlib import Path\nfrom src.kaggle import runner, mesh_jobs\nOUT = Path('/kaggle/working/out_A'); OUT.mkdir(exist_ok=True)\nsets = mesh_jobs.case_sets()\nprint({k: len(v) for k, v in sets.items()})\njobs = mesh_jobs.matrix_jobs(OUT, sets)\nprint(len(jobs), 'matrix jobs')"),
    md("### Phase 1: models x conditions on both GPUs\nTwo worker processes, one per T4 (`CUDA_VISIBLE_DEVICES`). Logs are in `out_A/logs`."),
    code("codes = runner.launch('src.kaggle.mesh_jobs:run_job', jobs, OUT / 'logs', cwd=REPO)\nprint('exit codes', codes)"),
    md("### Phase 2: ViT-S oracle\nFit each degraded image again while substituting the clean image's priors: correspondence only, depth only, or both. This shows how much of the error could be removed by perfect priors."),
    code("oracle = mesh_jobs.oracle_jobs(OUT, sets)\nprint(len(oracle), 'oracle jobs')\ncodes = runner.launch('src.kaggle.mesh_jobs:run_job', oracle, OUT / 'logs_oracle', cwd=REPO)\nprint('exit codes', codes)"),
    md("### Phase 3: raw projection, GLBs and cost"),
    code("from src.kaggle import appearance_jobs, export, cost\ntagged = mesh_jobs.collect_records(OUT)\nraw = appearance_jobs.raw_projection_from_vits(tagged, OUT / 'appearance')\nappearance_jobs.score_pairs(OUT / 'appearance', methods=('raw_projection',))\nindex = export.export_glbs(tagged, OUT / 'glb')\nprint(len(raw), 'raw projections;', len(index), 'GLB entries')"),
    code("import json\nrows = cost.cost_rows(tagged)\nvit = next((Path(r.case_dir) / 'fit' / 'priors.npz' for _, r in tagged if r.model_id == 'vits' and r.fit_status == 'success'), None)\nimage = next((c.image_path for c in sets['clean']), None)\nprofiles = cost.profile_models(OUT / 'flops', vit, image) if vit else {}\nsummary = cost.summarize_cost(rows, profiles)\n(OUT / 'cost_rows.json').write_text(json.dumps(rows, indent=1, default=str))\n(OUT / 'cost_summary.json').write_text(json.dumps(summary, indent=1, default=str))\nfrom src.reporting import table_markdown\nfrom IPython.display import Markdown\nMarkdown(table_markdown(summary))"),
    code("import shutil\nshutil.make_archive('/kaggle/working/out_A', 'zip', OUT)\nprint('zip written')"),
]

appearance = [
    md("# 12 - Account B: appearance benchmark\nAppearance backends (deca, opendelight, trust) x 20 subjects x 5 conditions. Each attempt saves its UV texture, render and metrics; a backend whose code or weights are missing is recorded as `not_run` with the reason. Output goes to `/kaggle/working/out_B` and is zipped."),
    code(CLONE),
    code("from src.kaggle import setup\nsetup.main([])"),
    code("from pathlib import Path\nfrom src.kaggle import runner, appearance_jobs\nOUT = Path('/kaggle/working/out_B'); OUT.mkdir(exist_ok=True)\nfrom src.appearance import backends\nprint(backends.availability())\njobs = appearance_jobs.jobs(OUT)\nprint(len(jobs), 'jobs')"),
    code("codes = runner.launch('src.kaggle.appearance_jobs:run_job', jobs, OUT / 'logs', cwd=REPO)\nprint('exit codes', codes)"),
    md("Degraded versus clean scores use the same method's clean result as the reference (UV maps on the common visible mask, and renders)."),
    code("pairs = appearance_jobs.score_pairs(OUT)\nglbs = appearance_jobs.export_obj_glbs(OUT)\nprint(len(pairs), 'pair scores;', len(glbs), 'GLBs')"),
    code("import shutil\nshutil.make_archive('/kaggle/working/out_B', 'zip', OUT)\nprint('zip written')"),
]

analysis = [
    md("# 20 - Results analysis (local)\nLoads the downloaded `out_A` and `out_B` folders and builds the tables, plots and galleries. Set the two paths below. Each section starts with a plain-words note on what it shows and how to read it."),
    code("OUT_A = r'D:/Kaggle/out_A'   # unzipped output of notebook 11\nOUT_B = r'D:/Kaggle/out_B'   # unzipped output of notebook 12\nNOW_DATA = r'Data'            # local NoW data (only needed for scan-error heatmaps)"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nfrom IPython.display import Markdown\nfrom src import analysis as an\nfrom src.reporting import table_markdown\nshow = lambda rows: Markdown(table_markdown(rows))\ntagged = an.load_mesh_records(OUT_A)\nrows = an.score_rows(tagged, OUT_A)\nprint(len(tagged), 'attempts;', sum(r['mean_mm'] is not None for r in rows), 'scored')"),
    md("## B1. Clean mesh error against the scan\nEach row is one model on the original photos of the 20 people. Error is the distance from each scan point to the fitted face surface after the local alignment, in millimetres; lower is better. `within 2.5 mm` is the share of scan points that are close. Statistics are taken over people (each person counts once)."),
    code("show(an.clean_mesh_table(rows))"),
    md("## B2. Degraded images: against the scan and against the clean result\nFor every model and condition, the mean error against the scan, then the paired change from the same person's clean fit (positive means worse). The Wilcoxon p-value tests whether the per-person changes are centred on zero; with 20 people, small p-values are suggestive rather than proof."),
    code("deg = an.degradation_table(rows)\nshow(deg)"),
    code("an.plot_error_vs_severity(deg)"),
    md("The curve shows error as the face gets smaller in the image (right to left means more severe). The compound condition is labelled; it is a different mix (blur, noise, JPEG) at 48 px width."),
    code("deltas = an.per_person_deltas(rows)\nan.plot_paired_lines(deltas, 'vits')"),
    md("Each line is one person: how their error changes from clean to each degraded condition. Lines that mostly go up mean the damage is consistent; a tangle around zero means the effect is small compared with person-to-person scatter."),
    code("an.plot_ced_by_condition(rows, 'vits')"),
    md("Cumulative error curves pool every scored scan point. A curve further right is worse; the gap between curves shows how much the whole error distribution shifts."),
    md("## Real-world photos\nExpressions, occlusions and selfies (originals only, no degradation) for all five models. Compare categories to see which kinds of photo hurt which model."),
    code("show(an.real_world_table(rows))"),
    md("## Oracle: how much could perfect priors recover?\nViT-S was re-run on degraded photos with the clean photo's priors swapped in. `gap recovered` = (degraded - oracle) / (degraded - clean). Near 1 means the priors are the whole problem; near 0 means a prior correction cannot help for that condition. When the clean-versus-degraded gap is tiny the ratio is unstable - read the gap column first."),
    code("show(an.oracle_table(rows))"),
    md("## B3. Clean appearance against the photo\nFor each method, the re-rendered textured face is compared with the input photo inside the face mask. PSNR and SSIM: higher is better. LPIPS, CIEDE2000 colour difference and ITA skin-tone angle difference: lower is better. ArcFace cosine: closer to 1 means the identity is kept. `unavailable` entries mean that metric's package or model was missing."),
    code("results, pairs = an.load_appearance(OUT_B)\nshow(an.appearance_clean_table(results))"),
    md("## B4. Degraded appearance against the same method's clean result\nThis asks how much each method's output moves when the photo gets worse. Render columns compare the images; UV columns compare the texture atlases on the region visible in both."),
    code("app = an.appearance_degraded_table(pairs)\nshow(app)"),
    code("an.plot_appearance_curves(app, 'render ciede2000')"),
    md("The curve tracks colour shift (CIEDE2000, lower is better) as the photo degrades; a flat line means the method is stable."),
    md("## Cost\nOne row per model: first-call time on a GPU (cold), median later calls (warm), peak GPU memory, counted GFLOPs and parameters where profiling worked. Each ViT fit runs in a fresh process, so even 'warm' includes model loading."),
    code("import json\ncost_rows = json.load(open(os.path.join(OUT_A, 'cost_summary.json')))\nshow(cost_rows)"),
    code("an.plot_cost_bars(cost_rows)"),
    md("## Image galleries\nInput strip and clay renders for one person across conditions, with scan-error heatmaps. Brighter heatmap colour means farther from the scan (clipped at 5 mm)."),
    code("import numpy as np\nfrom PIL import Image\nsubject = sorted({r['subject'] for r in rows if r['kind'] == 'clean'})[0]\nsel = [r for r in rows if r['kind'] == 'clean' and r['subject'] == subject and r['model'] == 'vits' and r['fit'] == 'success']\nimgs, titles = [], []\nfor r in sorted(sel, key=lambda r: an.CONDITIONS.index(r['condition'])):\n    obj = os.path.join(r['folder'], 'fit', 'neutral.obj')\n    if os.path.isfile(obj):\n        imgs.append(an.clay_render(obj)); titles.append(r['condition'])\nan.show_image_grid(imgs, titles)"),
    code("# Scan-error heatmaps (needs the local NoW data to re-read the scan).\nfrom src.data_now import installed_cases\nimport matplotlib.pyplot as plt\ncases = {c.image_key: c for c in installed_cases(NOW_DATA)}\nsel = [r for r in rows if r['kind'] == 'clean' and r['subject'] == subject and r['model'] == 'vits' and r['score_status'] == 'success']\nsel = sorted(sel, key=lambda r: an.CONDITIONS.index(r['condition']))\nfig, axes = plt.subplots(1, max(len(sel), 1), figsize=(3 * max(len(sel), 1), 3.2), squeeze=False)\nfor ax, r in zip(axes[0], sel):\n    case = cases[r['image']]\n    d = np.load(os.path.join(r['folder'], 'distances.npz'))['distances_mm']\n    an.scan_error_heatmap(case.scan_path, case.scan_landmark_path, d, ax)\n    ax.set_title(r['condition'], fontsize=8)"),
    code("# UV textures and clean-vs-degraded renders for the appearance methods.\nimport glob\nmethod = 'raw_projection'\nfolders = sorted(glob.glob(os.path.join(OUT_A, 'appearance', method, subject, '*')))\nimgs = [np.asarray(Image.open(os.path.join(f, 'uv_texture.png'))) for f in folders if os.path.isfile(os.path.join(f, 'uv_texture.png'))]\ntitles = [os.path.basename(f) for f in folders if os.path.isfile(os.path.join(f, 'uv_texture.png'))]\nan.show_image_grid(imgs, titles) if imgs else print('no UV textures found')"),
    code("imgs = [np.asarray(Image.open(os.path.join(f, 'render.png'))) for f in folders if os.path.isfile(os.path.join(f, 'render.png'))]\ntitles = [os.path.basename(f) for f in folders if os.path.isfile(os.path.join(f, 'render.png'))]\nan.show_image_grid(imgs, titles) if imgs else print('no renders found')"),
    md("## GLB index\nEvery exported mesh. Open a file in any glTF viewer (for example by dragging it into https://gltf-viewer.donmccurdy.com, or into the viewer under `ui/`)."),
    code("glb = an.glb_index_table(OUT_A, OUT_B)\nprint(len(glb), 'GLB files')\nshow(glb[:200])"),
]

if __name__ == "__main__":
    save("10_kaggle_readiness.ipynb", readiness, "notebooks/kaggle")
    save("11_kaggle_A_mesh.ipynb", mesh, "notebooks/kaggle")
    save("12_kaggle_B_appearance.ipynb", appearance, "notebooks/kaggle")
    save("20_results_analysis.ipynb", analysis, "notebooks")
