"""Build notebooks/20_faculty_showcase.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# TrueFace3D: degradation benchmark showcase

**Question.** How much does a small or degraded face photo hurt monocular 3D face reconstruction?

**Setup.** 20 NoW validation people, one neutral photo each. Every photo is given to 5 models in 5 versions:
the original, the face shrunk to 64 / 32 / 16 px wide, and a compound version (48 px + blur + noise + JPEG).
Each mesh is scored against that person's 3D scan (mean point-to-surface distance in mm after alignment; lower
is better). Appearance has no ground truth on NoW, so each degraded texture is compared with the same method's
result on the original photo."""),
    code("""import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from src import showcase as sc

# Later sources override earlier ones (add Kaggle B's out_A.zip when it is downloaded).
MESH_SOURCES = [ROOT / 'kaggle_outputs/mesh_A_runs', ROOT / 'kaggle_outputs/mesh_B_runs']
APPEARANCE = [ROOT / 'kaggle_outputs/app_dt', ROOT / 'kaggle_outputs/app_od2', ROOT / 'kaggle_outputs/app_trust']
rows = sc.load_mesh_results(*[p for p in MESH_SOURCES if p.exists()])
pairs, app_images = sc.load_appearance(*[p for p in APPEARANCE if p.exists()])
print(len(rows), 'scored meshes;', len(pairs), 'appearance pairs')"""),
    md("## 1. Scan error by model and condition"),
    code("""fig, ax = plt.subplots(figsize=(8, 4.5))
sc.plot_scan_error(rows, ax)
ax.set_title('Mean scan error, 20 NoW people (lower is better)')
plt.show()
tab = sc.summary_table(rows)
tab['mean'][sc.VARIANTS].round(3).rename(columns=sc.VARIANT_LABELS)"""),
    code("""tab['count'][sc.VARIANTS].rename(columns=sc.VARIANT_LABELS)  # people scored per cell (coverage)"""),
    md("""## 2. One person, every model, every condition

Top row: the photo each model received (all shown at the same display size, so small inputs look blocky).
Below: each model's reconstructed neutral face, rendered from the same camera. Under each mesh: scan error in mm
and the change from that model's original-photo result."""),
    code("""by = {(r['subject'], r['model'], r['variant']): r for r in rows}
subjects = sorted({r['subject'] for r in rows})
complete = [s for s in subjects if all((s, m, v) in by for m in ['vits', 'vitb', 'deca'] for v in sc.VARIANTS)]
# Show the person whose ViT-S result degrades the most (clearest example); change SUBJECT to see others.
SUBJECT = max(complete, key=lambda s: by[(s, 'vits', 'face-16')]['mean_mm'] - by[(s, 'vits', 'original')]['mean_mm'])
print('subject:', SUBJECT, '| complete subjects:', len(complete))"""),
    code("""models = [m for m in sc.MODELS if any((SUBJECT, m, v) in by for v in sc.VARIANTS)]
fig, axes = plt.subplots(len(models) + 1, 5, figsize=(13, 2.7 * (len(models) + 1)))
from src.data_now import installed_cases
orig_path = next((c.image_path for c in installed_cases() if c.subject_id == SUBJECT
                  and c.image_key == by[(SUBJECT, 'vits', 'original')]['image_key']), None)
for j, v in enumerate(sc.VARIANTS):
    inp = orig_path if v == 'original' else ROOT / f'kaggle_outputs/app_dt/deca/inputs/{SUBJECT}/{v}.png'
    ax = axes[0, j]; ax.axis('off'); ax.set_title(sc.VARIANT_LABELS[v], fontsize=12)
    if inp.exists():
        im = Image.open(inp); ax.imshow(im.resize((200, int(200 * im.height / im.width)), Image.NEAREST))
        ax.text(0.5, -0.06, f'{im.width}x{im.height} px', transform=ax.transAxes, ha='center', fontsize=9)
for i, m in enumerate(models, start=1):
    base = by.get((SUBJECT, m, 'original'))
    for j, v in enumerate(sc.VARIANTS):
        ax = axes[i, j]; ax.axis('off')
        r = by.get((SUBJECT, m, v))
        if j == 0:
            ax.text(-0.15, 0.5, sc.MODEL_LABELS[m].replace(' (', '\\n('), transform=ax.transAxes,
                    ha='right', va='center', fontsize=10, color=sc.MODEL_COLORS[m], weight='bold')
        if not r:
            ax.text(0.5, 0.5, 'fit failed', ha='center', va='center', color='gray'); continue
        V, F = sc.read_obj(r['obj'])
        if V is not None:
            sc.render_mesh(V, F, ax)
        delta = '' if (v == 'original' or not base) else f"  ({r['mean_mm'] - base['mean_mm']:+.2f})"
        ax.text(0.5, -0.04, f"{r['mean_mm']:.2f} mm{delta}", transform=ax.transAxes, ha='center', fontsize=10,
                color='#b8452f' if delta and r['mean_mm'] - base['mean_mm'] > 0.2 else 'black')
plt.tight_layout(); plt.show()"""),
    md("""## 3. Appearance: one person, every method, every condition

Under each image: similarity to the **same method's result on the original photo** (PSNR / SSIM higher is
better, LPIPS lower is better, ΔE2000 = colour difference, lower is better, ArcFace = identity similarity,
higher is better)."""),
    code("""# Representative person for appearance: median DECA colour difference at 32 px (not a best or worst case).
d32 = sorted((p['render']['ciede2000'], p['subject']) for p in pairs
             if p['method'] == 'deca' and p['condition'] == 'face-32' and p.get('status') == 'ok')
APP_SUBJECT = d32[len(d32) // 2][1]
print('appearance subject:', APP_SUBJECT)

def face_crop(im, pad=0.15):
    a = np.asarray(im.convert('RGB')).astype(int).sum(-1) > 15
    if not a.any(): return im
    ys, xs = np.where(a); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1; p = int(pad * max(h, w))
    return im.crop((max(xs.min() - p, 0), max(ys.min() - p, 0), xs.max() + p, ys.max() + p))"""),
    code("""methods = [m for m in ['deca', 'trust', 'opendelight'] if any(k[0] == m for k in app_images)]
SUBJECT_A = APP_SUBJECT
fig, axes = plt.subplots(len(methods), 5, figsize=(13, 3.2 * len(methods)), squeeze=False)
for i, meth in enumerate(methods):
    for j, v in enumerate(sc.VARIANTS):
        ax = axes[i, j]; ax.axis('off')
        if i == 0: ax.set_title(sc.VARIANT_LABELS[v], fontsize=12)
        if j == 0:
            ax.text(-0.1, 0.5, meth.upper() if meth != 'opendelight' else 'OpenDelight', transform=ax.transAxes,
                    ha='right', va='center', fontsize=11, weight='bold')
        p = app_images.get((meth, SUBJECT_A, v))
        if not p:
            ax.text(0.5, 0.5, 'failed', ha='center', va='center', color='gray'); continue
        im = face_crop(Image.open(p)).convert('RGB'); ax.imshow(im.resize((200, int(200 * im.height / im.width))))
        mt = sc.pair_metrics(pairs, meth, SUBJECT_A, v)
        if mt:
            ax.text(0.5, -0.05, f"PSNR {mt['psnr']:.1f}  SSIM {mt['ssim']:.2f}\\nLPIPS {mt['lpips']:.3f}  "
                    f"ΔE {mt['ciede2000']:.1f}  ID {mt['arcface_cos']:.2f}", transform=ax.transAxes,
                    ha='center', va='top', fontsize=8.5)
        elif v == 'original':
            ax.text(0.5, -0.05, 'reference', transform=ax.transAxes, ha='center', va='top', fontsize=9)
plt.tight_layout(); plt.show()"""),
    md("## 4. Appearance across all 20 people"),
    code("""recs = [{'method': p['method'], 'variant': p['condition'], **p['render']} for p in pairs
        if p.get('status') == 'ok' and isinstance(p.get('render'), dict)]
adf = pd.DataFrame(recs)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, metric, label in zip(axes, ['arcface_cos', 'ciede2000', 'lpips'],
                             ['Identity similarity (ArcFace, higher = better)', 'Colour difference ΔE2000 (lower = better)',
                              'LPIPS (lower = better)']):
    for meth, g in adf.groupby('method'):
        s = g.groupby('variant')[metric].mean().reindex(sc.VARIANTS[1:])
        ax.plot(range(4), s.values, '-o', lw=2, label=meth)
    ax.set_xticks(range(4), [sc.VARIANT_LABELS[v] for v in sc.VARIANTS[1:]]); ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
adf.groupby(['method', 'variant'])[['psnr', 'ssim', 'lpips', 'ciede2000', 'arcface_cos']].mean().round(3)"""),
    md("""## 5. What this shows

- **RealDenseFace (ViT-S, ViT-B) reads the face at 512 px** and is accurate on good photos, but its error
  rises sharply at 16 px (ViT-S: every one of the 20 people gets worse).
- **DECA and MICA squeeze the face to 224 px** (MICA's identity encoder to 112 px), so a small photo removes little
  that they use: they stay stable, but they are coarse.
- **Appearance follows geometry:** DECA's photo-based texture holds to 32 px and breaks at 16 px. OpenDelight keeps
  its structure but loses identity first.
- **Next step:** correct the dense model's intermediate representation (GARD's idea) so it keeps its accuracy on bad
  photos, and compare with pixel restoration (Real-ESRGAN, GFPGAN, CodeFormer)."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "notebooks/20_faculty_showcase.ipynb")
print("written")
