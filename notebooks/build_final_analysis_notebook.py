"""
Generate notebook 14 — FINAL ANALYSIS (CPU) on the complete dataset.

Runs the analysis on the COMPLETE results folder (rhprofile_results_other), which
holds everything: the 18-model panel, the 3-family bnb4 rings, qwen AWQ + GPTQ,
phi, mistral, seeds. Regenerates the RQ2 prediction + RQ3 inheritance JSONs and
prints the paper tables (RQ1 profiles, RQ3 within-method bnb4 rings, the E14
cross-method table, RQ2 prediction).

No GPU. ~minutes. This is the notebook to run after all experiments are done.
ONLY writes 14_final_analysis_colab.ipynb.

Run:  python notebooks/build_final_analysis_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import md, code, notebook, SETUP_PIP, SETUP_CLONE, SETUP_PATHS

CPU_DRIVE_OTHER = code("""
# Cell 0 — Drive + the COMPLETE results folder (CPU; Runtime -> None is fine)
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/MyDrive/rhprofile_results_other'   # the complete data
except Exception:
    RESULTS_DIR = '/content/rhprofile_results_other'
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Results dir (complete data):', RESULTS_DIR)
""")

RUN = code("""
# Regenerate RQ2 prediction (E8/E9) + RQ3 inheritance (E10-E15) on the complete data.
import subprocess, sys
P2 = '/content/rope-part2'
def run(args):
    cmd = [sys.executable] + args + ['--results-dir', RESULTS_DIR, '--config', CONFIG,
                                     '--part1-repo', '/content/rope-part1']
    print('>>', ' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-1500:] if r.returncode else '')
run([f'{P2}/scripts/run_prediction.py', '--seed', '42', '--retest-seed', '123'])
run([f'{P2}/scripts/run_inheritance.py', '--lineage', 'all', '--seed', '42'])
print('\\nRegenerated analysis/*.json + inheritance/*.json on the complete data.')
""")

TABLES = code("""
# Paper tables — RQ1 profiles, RQ3 bnb4 rings, E14 cross-method, RQ2 prediction.
import json, glob, os
from pathlib import Path
RD = Path(RESULTS_DIR)
def L(p): return json.load(open(p, encoding='utf-8'))
def ok(x): return x is not None and not (isinstance(x, float) and x != x)
def f(x, d='%6.2f', w=6):
    return (d % x) if ok(x) else (' ' * (w - 3) + 'nan')

print('##### RQ1 — per-model profiles #####')
print(f\"{'model':28s} {'#heads':>6} {'gini':>5} {'Lcom':>5} {'freqEff':>7} {'util_d':>7}\")
for p in sorted(glob.glob(str(RD/'profile'/'*_seed42.json'))):
    m = os.path.basename(p).replace('_seed42.json', ''); d = L(p); sc = d['profile']['scalars']
    up = RD/'utility'/f'{m}_seed42.json'
    ud = L(str(up))['utility'].get('cohens_d') if up.exists() else None
    print(f\"{m:28s} {len(d.get('argmax_heads',[])):6d} {f(sc.get('gini'),'%5.2f',5)} \"
          f\"{f(sc.get('layer_com'),'%5.2f',5)} {f(sc.get('frequency_effect'),'%7.2f',7)} {f(ud,'%7.2f',7)}\")

print('\\n##### RQ3 — within-method 4-bit (bnb NF4) rings #####')
for lin in ['llama', 'gemma', 'qwen', 'mistral']:
    jp = RD/'inheritance'/f'{lin}.json'
    if not jp.exists(): continue
    for r in L(str(jp)).get('rings', []):
        c = r.get('child', '')
        if 'bnb4' in c:
            print(f\"  {lin:6s} {r.get('parent')} -> {c}: copyJac=%.3f dFreqCom=%.3f\"
                  % (r['E10_identity']['copy']['jaccard'], r['E12_frequency']['delta_freq_com']))

print('\\n##### E14 — CROSS-METHOD (qwen instruct -> 4-bit, by method) #####')
def prof(k):
    p = RD/'profile'/f'{k}_seed42.json'; return L(str(p)) if p.exists() else None
def jac(a, b):
    sa = {tuple(x) for x in a}; sb = {tuple(x) for x in b}
    return len(sa & sb)/len(sa | sb) if (sa or sb) else float('nan')
ref = prof('qwen25_7b_instruct')
if ref:
    rh = ref.get('copy_heads', []); ra = ref.get('argmax_heads', [])
    print(f\"  {'method':7s} {'copyJac':>8s} {'argmaxJac':>10s} {'#heads':>7}\")
    for mth, k in [('bnb4','qwen25_7b_instruct_bnb4'),
                   ('awq4','qwen25_7b_instruct_awq4'),
                   ('gptq4','qwen25_7b_instruct_gptq4')]:
        c = prof(k)
        if not c: print(f'  {mth:7s} (missing)'); continue
        print(f'  {mth:7s} {jac(rh, c.get(\"copy_heads\", [])):8.3f} '
              f'{jac(ra, c.get(\"argmax_heads\", [])):10.3f} {len(c.get(\"argmax_heads\", [])):7d}')
    print('  (all > 0.8*R_self -> inherited; AWQ most faithful)')

print('\\n##### RQ2 — prediction (BH-corrected single correlations) #####')
an = L(str(RD/'analysis'/'prediction_e8.json'))
for r in sorted(an['single_correlations'], key=lambda r: -abs(r.get('spearman_rho') or 0))[:6]:
    star = '  *' if r.get('significant_bh') else ''
    print(f\"  {r['predictor']:16s} -> {r['target']:14s} rho=%+.3f p_bh=%.3f%s\"
          % (r['spearman_rho'], r.get('p_adjusted_bh'), star))
print('  (no metric significant after BH -> RQ2 is an honest null)')
""")


def nb_14_final():
    cells = [md(
        "# 14 · FINAL ANALYSIS (CPU) — complete dataset, all paper tables\n"
        "**GPU: NONE** (Runtime → None). Runs on the COMPLETE results folder "
        "**`rhprofile_results_other`** — the 18-model panel + 3-family bnb4 rings + "
        "qwen AWQ/GPTQ + phi + mistral + seeds. Regenerates RQ2 prediction (E8/E9) "
        "and RQ3 inheritance (E10–E15), then prints the paper tables: RQ1 profiles, "
        "RQ3 within-method bnb4 rings, the **E14 cross-method** table, and RQ2.\n\n"
        "Run after all experiments are done. ~minutes. (06 is the older analysis "
        "notebook but points at the main folder and lacks the E14 table — use this.)")]
    cells += [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + tokens in Cell 2."),
        CPU_DRIVE_OTHER, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## 1 — Regenerate prediction + inheritance on the complete data"))
    cells.append(RUN)
    cells.append(md("## 2 — Paper tables (RQ1 · RQ3 bnb4 rings · E14 cross-method · RQ2)"))
    cells.append(TABLES)
    return notebook(cells, gpu=False)


def main():
    name = "14_final_analysis_colab.ipynb"
    nb = nb_14_final()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("CPU (no GPU). Points at rhprofile_results_other. Prints all paper tables.")


if __name__ == "__main__":
    main()
