"""
Generate the Colab notebooks for the Retrieval-Head Profile project.

Run from anywhere:
    python notebooks/build_notebooks.py

Each notebook is a self-contained Colab "task" sized to finish inside one
session (≤24 h, default budget 11 h so the free tier survives) and is
resume-safe: every per-model result is written to Google Drive and skipped on
re-run, so the full panel is completed across several sessions. The notebooks
reuse the project's tested helpers (``scripts._common``, ``rhp.*``) rather than
re-implementing logic, so Colab and local runs share one code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

NB_DIR = Path(__file__).resolve().parent
CONFIG_PATH = NB_DIR.parent / "configs" / "panel.yaml"

# Conservative per-model wall-clock estimates on an L4 (8-bit). Used only to size
# the chunks so each notebook finishes well inside a 24 h Colab session; the
# TIME_BUDGET_HOURS guard inside every notebook is the hard backstop.
EST_PROFILE_H = 3.5
EST_BEHAVIOR_H = 2.5
PROFILE_CHUNK = 5          # 5 × 3.5 h ≈ 17.5 h nominal
BEHAVIOR_CHUNK = 7         # 7 × 2.5 h ≈ 17.5 h nominal
# Backstop guard. Checked BEFORE each model, so the worst-case finish is
# budget + one model (≈3.5 h) — kept under 24 h with margin.
TIME_BUDGET_HOURS = 20


def panel_model_order() -> list[str]:
    """Ordered panel keys, core models first (so they finish early)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    models = cfg.get("models", {})
    core = [k for k, v in models.items() if v.get("tier") == "core"]
    rest = [k for k in models if k not in core]
    return core + rest


def chunks(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict], gpu: bool = True) -> dict:
    meta = {
        "accelerator": "GPU" if gpu else "None",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


# ---------------------------------------------------------------------------
# Shared setup cells (clone Part-1 + Part-2, install, paths, HF login)
# ---------------------------------------------------------------------------

SETUP_GPU_DRIVE = code("""
# Cell 0 — GPU check + Google Drive + results dir
import subprocess, os
print(subprocess.check_output('nvidia-smi', shell=True).decode())

USE_DRIVE = True   # keep True so results survive a disconnect and resume
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/MyDrive/rhprofile_results'
else:
    RESULTS_DIR = '/content/rhprofile_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Results dir:', RESULTS_DIR)
""")

SETUP_PIP = code("""
%%bash
# Cell 1 — dependencies (pinned transformers to match the Part-1 src/ behaviour)
pip install -q transformers==4.47.0 bitsandbytes accelerate datasets
pip install -q scipy scikit-learn matplotlib seaborn pandas huggingface_hub tqdm pyyaml
echo 'Install complete.'
""")

SETUP_CLONE = code("""
# Cell 2 — tokens + clone BOTH repos
#   • Part 1 provides the inherited src/ (detector, patching, statistics).
#   • Part 2 provides rhp/, scripts/, configs/panel.yaml.
# Paste tokens below. If the repos are public you can leave GITHUB_TOKEN blank.
import os, subprocess

GITHUB_TOKEN = \"\"          # ghp_... (needed only for private repos)
HF_TOKEN     = \"\"          # hf_...  (needed for gated models: Llama/Gemma)

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

# --- repos (defaults point at the author's GitHub; change if you fork) ---
PART1 = dict(owner='CengizhanBayram',
             name='Does-RoPE-Prevent-or-Degrade-Retrieval-Heads-A-Mechanistic-Analysis-Across-Model-Families',
             dir='/content/rope-part1')
PART2 = dict(owner='CengizhanBayram',
             name='retrieval-head-profile',
             dir='/content/rope-part2')

def clone(repo):
    tok = GITHUB_TOKEN
    pub = f\"https://github.com/{repo['owner']}/{repo['name']}.git\"
    auth = f\"https://x-access-token:{tok}@github.com/{repo['owner']}/{repo['name']}.git\" if tok else pub
    if not os.path.isdir(repo['dir']):
        r = subprocess.run(['git', 'clone', auth, repo['dir']], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).replace(tok or '___', '***'))
        if tok:
            subprocess.run(['git', '-C', repo['dir'], 'remote', 'set-url', 'origin', pub])
    else:
        subprocess.run(['git', '-C', repo['dir'], 'pull'], capture_output=True, text=True)
    print('ready:', repo['dir'])

clone(PART1); clone(PART2)
""")

SETUP_PATHS = code("""
# Cell 3 — paths + HF login
import sys, os
sys.path.insert(0, '/content/rope-part2')          # rhp, scripts
os.environ['RHP_PART1_REPO'] = '/content/rope-part1'
sys.path.insert(0, '/content/rope-part1')          # src (inherited)
CONFIG = '/content/rope-part2/configs/panel.yaml'

from scripts._common import bootstrap
bootstrap('/content/rope-part1')
try:
    from src.auth_utils import login_huggingface
    login_huggingface(required=False)
except Exception as e:
    print('HF login skipped:', e)
print('Setup OK. CONFIG =', CONFIG)
""")


def setup_cells() -> list[dict]:
    return [
        md("### Setup — run cells 0–3 once per session\n"
           "Mounts Drive, installs deps, clones the Part-1 (inherited `src/`) and "
           "Part-2 (`rhp/`) repos, and wires up the paths. **Edit `PART1`/`PART2` "
           "owners** and paste tokens in Cell 2 before running."),
        SETUP_GPU_DRIVE, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]


# ---------------------------------------------------------------------------
# Notebook 00 — pilot (WP1)
# ---------------------------------------------------------------------------

def nb_pilot() -> dict:
    cells = [md(
        "# 00 · Pilot (WP1) — validate the whole pipeline on 3 models\n"
        "Runs the **full profile (E1–E5)** and the **behaviour sweep (E6–E7)** on a "
        "3-model pilot, exactly as the panel run will, to confirm the inherited "
        "`src/` plumbing works end-to-end before scaling up. Expect ~6–9 h on an "
        "L4. Everything is resume-safe to Drive.\n\n"
        "| Task | Experiments | Output |\n|---|---|---|\n"
        "| Profile | E1 dual detector · E2 freq signature · E3 knockout · E4/E5 profile | `profile/<model>_seed42.json` |\n"
        "| Behaviour | E6 NIAH sweep · E7 RULER subset | `behavior/<model>_seed42.json` |")]
    cells += setup_cells()
    cells.append(md("## Task P — pilot profile + behaviour (3 models)"))
    cells.append(code("""
# Pilot models span 3 families + a small model (long-context behaviour).
import time
from pathlib import Path
from scripts._common import run_profile_for_model, run_behavior_for_model, save_json
from rhp.panel import load_panel, model_cfg

config = load_panel(CONFIG)
PILOT = ['llama32_3b', 'qwen25_7b', 'olmo2_7b']
SEED = 42
TIME_BUDGET_HOURS = 16.0          # 3 models × ~6 h (profile+behaviour) ≈ 18 h < 24 h

prof_dir = Path(RESULTS_DIR) / 'profile'; prof_dir.mkdir(parents=True, exist_ok=True)
beh_dir  = Path(RESULTS_DIR) / 'behavior'; beh_dir.mkdir(parents=True, exist_ok=True)
start = time.time()

for key in PILOT:
    if time.time() - start > TIME_BUDGET_HOURS * 3600:
        print('Time budget reached — re-run to resume.'); break
    # --- profile (E1–E5) ---
    pout = prof_dir / f'{key}_seed{SEED}.json'
    if pout.exists():
        print(key, 'profile done -> skip')
    else:
        try:
            res = run_profile_for_model(key, model_cfg(config, key), config,
                                        seed=SEED, context_length=4096)
            save_json(res, pout)
            print(key, 'profile: argmax heads =', res['profile']['n_heads'],
                  '| copy heads =', res['profile']['n_heads_copy'],
                  '| detector Jaccard =', round(res['profile']['detector_agreement']['jaccard'], 3))
        except Exception as e:
            print(key, 'profile FAILED:', e)
    # --- behaviour (E6–E7) ---
    bout = beh_dir / f'{key}_seed{SEED}.json'
    if bout.exists():
        print(key, 'behaviour done -> skip')
    else:
        try:
            res = run_behavior_for_model(key, model_cfg(config, key), config, seed=SEED)
            res['family'] = model_cfg(config, key).get('family')
            save_json(res, bout)
            print(key, 'behaviour: NIAH overall =', round(res['behavior']['niah_overall'], 3))
        except Exception as e:
            print(key, 'behaviour FAILED:', e)

print('\\nPilot done. If the numbers look sane, scale up with notebooks 01 + 02.')
"""))
    cells.append(md("## Inspect one pilot profile"))
    cells.append(code("""
import json
from pathlib import Path
p = Path(RESULTS_DIR) / 'profile' / 'qwen25_7b_seed42.json'
if p.exists():
    prof = json.load(open(p))['profile']
    print('n_heads       :', prof['n_heads'])
    print('frac          :', round(prof['frac'], 4))
    print('gini          :', round(prof['concentration']['gini'], 3))
    print('layer COM     :', round(prof['layer_profile']['layer_com_weighted'], 3))
    print('freq COM/width:', prof['scalars']['freq_com'], '/', prof['scalars']['freq_width'])
    print('knockout drop :', prof['scalars']['knockout_drop'])
else:
    print('Run the pilot task first.')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Notebook 01 — panel profile (Block A, E1–E5)
# ---------------------------------------------------------------------------

def nb_profile_chunk(models: list[str], idx: int, n_chunks: int) -> dict:
    """One profile notebook for a fixed chunk of models (sized to < 24 h)."""
    est = len(models) * EST_PROFILE_H
    letter = chr(ord("A") + idx)
    is_first = idx == 0
    cells = [md(
        f"# Profile · chunk {letter} of {n_chunks} (Block A · E1–E5)\n"
        f"Extracts the **retrieval-head profile** for **{len(models)} models** "
        f"(≈{est:.0f} h on an L4 — under one 24 h Colab session): two detectors "
        f"(E1), frequency signature + dose patch (E2/E12), knockout (E3), "
        f"concentration + layer profile (E4), GQA control (E5).\n\n"
        f"**This chunk's models:**\n\n`{models}`\n\n"
        f"Resume-safe: each model writes `profile/<model>_seed<seed>.json` to "
        f"Drive and is skipped on re-run. The `TIME_BUDGET_HOURS={TIME_BUDGET_HOURS}` "
        f"guard stops the loop before the Colab limit; just re-run to finish any "
        f"remainder. Run chunks A→{chr(ord('A')+n_chunks-1)} in any order / parallel "
        f"Colab accounts.")]
    cells += setup_cells()
    cells.append(md(f"## Profile this chunk ({len(models)} models, seed 42)"))
    cells.append(code(f"""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, save_json
from rhp.panel import load_panel, model_cfg

config = load_panel(CONFIG)
MODEL_SUBSET = {models}
SEED = 42
CONTEXT = 4096
TIME_BUDGET_HOURS = {TIME_BUDGET_HOURS}    # hard backstop < 24 h Colab limit

OUT = Path(RESULTS_DIR) / 'profile'; OUT.mkdir(parents=True, exist_ok=True)
start = time.time()
for key in MODEL_SUBSET:
    out = OUT / f'{{key}}_seed{{SEED}}.json'
    if out.exists():
        print(key, 'done -> skip'); continue
    if time.time() - start > TIME_BUDGET_HOURS * 3600:
        print('Time budget reached — re-run this notebook to resume at', key); break
    try:
        res = run_profile_for_model(key, model_cfg(config, key), config,
                                    seed=SEED, context_length=CONTEXT)
        save_json(res, out)
        pr = res['profile']
        print(f"{{key}}: heads={{pr['n_heads']}} copy={{pr['n_heads_copy']}} "
              f"gini={{pr['concentration']['gini']:.3f}} "
              f"freq_com={{pr['scalars']['freq_com']}} "
              f"knock={{pr['scalars']['knockout_drop']}}  "
              f"[{{(time.time()-start)/3600:.1f}} h elapsed]")
    except Exception as e:
        print(key, 'FAILED:', e)
print('\\nChunk {letter} elapsed %.1f h.' % ((time.time()-start)/3600))
"""))
    if is_first:
        cells.append(md(
            "## Extra seeds for the 5 core models (R5) — run after all chunks\n"
            "The core models (`llama32_3b, llama31_8b, qwen25_3b, qwen25_7b, "
            "gemma2_9b`) get 3 seeds. This cell does seeds 123 + 2024 for **just "
            "the core** (≈18 h each seed → run as two more sessions if needed)."))
        cells.append(code(f"""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, save_json
from rhp.panel import load_panel, model_cfg, core_models

config = load_panel(CONFIG)
CORE = core_models(config)
OUT = Path(RESULTS_DIR) / 'profile'; OUT.mkdir(parents=True, exist_ok=True)
TIME_BUDGET_HOURS = {TIME_BUDGET_HOURS}
for SEED in (123, 2024):
    start = time.time()
    for key in CORE:
        out = OUT / f'{{key}}_seed{{SEED}}.json'
        if out.exists(): print(key, SEED, 'done -> skip'); continue
        if time.time() - start > TIME_BUDGET_HOURS*3600:
            print('budget reached at', key, SEED); break
        try:
            save_json(run_profile_for_model(key, model_cfg(config, key), config,
                      seed=SEED, context_length=4096), out)
            print(key, 'seed', SEED, 'done')
        except Exception as e:
            print(key, SEED, 'FAILED:', e)
"""))
    cells.append(md("## Quick summary of everything profiled so far"))
    cells.append(code("""
import json, glob, pandas as pd
from pathlib import Path
rows = []
for f in sorted(glob.glob(str(Path(RESULTS_DIR)/'profile'/'*_seed42.json'))):
    r = json.load(open(f)); p = r['profile']
    rows.append(dict(model=r['model'], family=r.get('family'),
                     n_heads=p['n_heads'], copy=p['n_heads_copy'],
                     gini=round(p['concentration']['gini'],3),
                     det_jacc=round(p['detector_agreement']['jaccard'],3),
                     freq_com=p['scalars']['freq_com'],
                     knock=p['scalars']['knockout_drop']))
print(pd.DataFrame(rows).to_string(index=False) if rows else 'No profiles yet.')
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Notebook 02 — panel behaviour (Block B, E6–E7)
# ---------------------------------------------------------------------------

def nb_behavior_chunk(models: list[str], idx: int, n_chunks: int) -> dict:
    """One behaviour notebook for a fixed chunk of models (sized to < 24 h)."""
    est = len(models) * EST_BEHAVIOR_H
    letter = chr(ord("A") + idx)
    cells = [md(
        f"# Behaviour · chunk {letter} of {n_chunks} (Block B · E6 NIAH + E7 RULER)\n"
        f"Behavioural targets the profile must predict (RQ2) for **{len(models)} "
        f"models** (≈{est:.0f} h on an L4): the NIAH position/length sweep (E6) and "
        f"the RULER subset — multi-key, multi-value, variable tracking (E7).\n\n"
        f"**This chunk's models:**\n\n`{models}`\n\n"
        f"Resume-safe to `behavior/<model>_seed<seed>.json`; "
        f"`TIME_BUDGET_HOURS={TIME_BUDGET_HOURS}` backstop. ≤3B models automatically "
        f"get the long-context lengths (8192/16384).")]
    cells += setup_cells()
    cells.append(md(f"## Behaviour for this chunk ({len(models)} models, seed 42)"))
    cells.append(code(f"""
import time
from pathlib import Path
from scripts._common import run_behavior_for_model, save_json
from rhp.panel import load_panel, model_cfg

config = load_panel(CONFIG)
MODEL_SUBSET = {models}
SEED = 42
TIME_BUDGET_HOURS = {TIME_BUDGET_HOURS}
DO_RULER = True

small = config['niah'].get('context_lengths_small')
OUT = Path(RESULTS_DIR) / 'behavior'; OUT.mkdir(parents=True, exist_ok=True)
start = time.time()
for key in MODEL_SUBSET:
    out = OUT / f'{{key}}_seed{{SEED}}.json'
    if out.exists():
        print(key, 'done -> skip'); continue
    if time.time() - start > TIME_BUDGET_HOURS * 3600:
        print('Time budget reached — re-run this notebook to resume at', key); break
    cfg = model_cfg(config, key)
    ctx = small if (small and any(t in key for t in ('3b','2b','1_6b','mini'))) else None
    try:
        res = run_behavior_for_model(key, cfg, config, seed=SEED,
                                     context_lengths=ctx, do_ruler=DO_RULER)
        res['family'] = cfg.get('family')
        save_json(res, out)
        b = res['behavior']
        print(f"{{key}}: NIAH overall={{b['niah_overall']:.3f}} "
              f"worst_pos={{b['niah_worst_pos']:.3f}}  "
              f"[{{(time.time()-start)/3600:.1f}} h elapsed]")
    except Exception as e:
        print(key, 'FAILED:', e)
print('\\nChunk {letter} elapsed %.1f h.' % ((time.time()-start)/3600))
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Notebook 03 — inheritance chains (Block C, E10–E15)
# ---------------------------------------------------------------------------

def nb_inheritance() -> dict:
    cells = [md(
        "# 03 · Inheritance chains (Block C · E10–E15)\n"
        "Measures **circuit inheritance** across each lineage's adjacent ring-pairs: "
        "identity (E10), function (E11), frequency (E12), behaviour bridge (E13), "
        "the quantization ablation (E14 — AWQ vs GPTQ), and the RQ4 localisation "
        "read-out (E15).\n\n"
        "Two phases: **(A, GPU)** make sure every model in the chosen lineages has a "
        "profile + behaviour on Drive (runs only the missing ones); **(B, CPU)** the "
        "pure-analysis comparison. The 4-bit rings need extra kernels — installed "
        "below.")]
    cells += setup_cells()
    cells.append(md("## Extra install for the 4-bit rings (AWQ / GPTQ)\n"
                    "Only needed if your lineage includes the quantized rings "
                    "(`qwen25_7b_instruct_awq4`, `..._gptq4`)."))
    cells.append(code("""
%%bash
pip install -q autoawq optimum auto-gptq 2>/dev/null || echo 'AWQ/GPTQ kernels optional; skip if unused.'
echo done
"""))
    cells.append(md("## Phase A (GPU) — fill in any missing profile/behaviour for the lineages"))
    cells.append(code("""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, run_behavior_for_model, save_json
from rhp.panel import load_panel, model_cfg, lineage_chain, lineage_sibling

config = load_panel(CONFIG)
LINEAGES = ['qwen', 'llama', 'gemma', 'mistral']     # edit to taste
SEED = 42
TIME_BUDGET_HOURS = 18.0          # backstop; most base/instruct are already on Drive from the chunks

# collect every model needed by the chosen lineages (chain rings + siblings)
needed = []
for ln in LINEAGES:
    needed += lineage_chain(config, ln)
    s = lineage_sibling(config, ln)
    if s: needed.append(s)
needed = list(dict.fromkeys(needed))
print('models needed:', needed)

prof = Path(RESULTS_DIR)/'profile'; beh = Path(RESULTS_DIR)/'behavior'
prof.mkdir(parents=True, exist_ok=True); beh.mkdir(parents=True, exist_ok=True)
start = time.time()
for key in needed:
    if time.time() - start > TIME_BUDGET_HOURS*3600:
        print('Time budget reached — re-run to resume.'); break
    cfg = model_cfg(config, key)
    pout = prof / f'{key}_seed{SEED}.json'
    if not pout.exists():
        try:
            save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), pout)
            print(key, 'profile saved')
        except Exception as e:
            print(key, 'profile FAILED:', e)
    bout = beh / f'{key}_seed{SEED}.json'
    if not bout.exists():
        try:
            r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
            save_json(r, bout); print(key, 'behaviour saved')
        except Exception as e:
            print(key, 'behaviour FAILED:', e)
print('Phase A done.')
"""))
    cells.append(md("## Phase B (CPU) — inheritance analysis (E10–E15)"))
    cells.append(code("""
import json
from pathlib import Path
from rhp.panel import load_panel, lineage_ring_pairs, lineage_sibling
from rhp.inheritance import compare_ring, compare_identity, localize_recall_change
from scripts._common import save_json

config = load_panel(CONFIG)
RD = Path(RESULTS_DIR)
out_dir = RD / 'inheritance'; out_dir.mkdir(parents=True, exist_ok=True)
SEED = 42

def load_merged(model):
    pf = RD/'profile'/f'{model}_seed{SEED}.json'
    if not pf.exists(): return None
    res = json.load(open(pf))
    bf = RD/'behavior'/f'{model}_seed{SEED}.json'
    if bf.exists(): res['behavior'] = json.load(open(bf)).get('behavior', {})
    return res

for ln in ['qwen', 'llama', 'gemma', 'mistral']:
    rings = []
    for a, b in lineage_ring_pairs(config, ln):
        pa, pb = load_merged(a), load_merged(b)
        if pa is None or pb is None:
            print(ln, 'skip', a, '->', b, '(missing)'); continue
        ring = compare_ring(pa, pb, lineage=ln)
        ring['E15_localization'] = localize_recall_change(ring)
        rings.append(ring)
        print(f"{ln}: {a}->{b}  copyJ={ring['E10_identity']['copy']['jaccard']:.3f} "
              f"Δfreq_com={ring['E12_frequency']['delta_freq_com']} "
              f"verdict={ring['E15_localization']['verdict']}")
    out = {'lineage': ln, 'rings': rings}
    sib = lineage_sibling(config, ln)
    if sib:
        base = load_merged(config['lineages'][ln]['chain'][0]); sres = load_merged(sib)
        if base and sres:
            out['sibling'] = {'base': config['lineages'][ln]['chain'][0], 'sibling': sib,
                              'identity': compare_identity(base, sres)}
    save_json(out, out_dir / f'{ln}.json')
print('\\nInheritance analysis written to', out_dir)
"""))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Notebook 04 — prediction analysis (E8, E9) — CPU
# ---------------------------------------------------------------------------

def nb_prediction() -> dict:
    cells = [md(
        "# 04 · Prediction analysis (Block B · E8 + E9) — CPU only\n"
        "The RQ2 test: do the training-free profile scalars predict the behavioural "
        "scores? BH-corrected single correlations + a LOO 3-predictor regression "
        "(E8), family-demeaned correlations (E8 iii), and the test-retest reliability "
        "ceiling (E9). **No GPU needed** — set Runtime → None. Just needs the "
        "`profile/` + `behavior/` JSONs on Drive.")]
    # CPU notebook: keep Drive + paths but no GPU/install of torch needed (still
    # need numpy/pandas/scipy/sklearn which Colab has by default).
    cells.append(SETUP_GPU_DRIVE)
    cells.append(code("""
%%bash
pip install -q scikit-learn pandas scipy pyyaml 2>/dev/null
echo ok
"""))
    cells.append(SETUP_CLONE)
    cells.append(code("""
# paths (CPU; we only need rhp + the inherited stats_utils, no model load)
import sys, os
sys.path.insert(0, '/content/rope-part2')
os.environ['RHP_PART1_REPO'] = '/content/rope-part1'
sys.path.insert(0, '/content/rope-part1')
CONFIG = '/content/rope-part2/configs/panel.yaml'
from scripts._common import bootstrap; bootstrap('/content/rope-part1')
print('ok')
"""))
    cells.append(md("## E8 — profile → behaviour prediction"))
    cells.append(code("""
import json, glob
from pathlib import Path
from rhp.prediction import run_prediction_analysis, single_correlations, build_table
from scripts._common import save_json
import pandas as pd

RD = Path(RESULTS_DIR); SEED = 42
def collect(seed):
    out = []
    for f in sorted(glob.glob(str(RD/'profile'/f'*_seed{seed}.json'))):
        r = json.load(open(f))
        bf = RD/'behavior'/f"{r['model']}_seed{seed}.json"
        if bf.exists(): r['behavior'] = json.load(open(bf)).get('behavior', {})
        out.append(r)
    return out

results = collect(SEED)
print('models:', len(results))
analysis = run_prediction_analysis(results)
save_json(analysis, RD/'analysis'/'prediction_e8.json')

df = pd.DataFrame(analysis['single_correlations'])
print('\\nStrongest correlations (BH-corrected):')
print(df.sort_values('p_value').head(10).to_string(index=False))
print('\\nLOO 3-predictor regression:', analysis['loo_top3_predictors'])
for t, loo in analysis['loo_regression'].items():
    print(f"  {t}: LOO R^2 = {loo.get('loo_r2'):.3f}")
"""))
    cells.append(md("## E9 — test-retest reliability (needs a 2nd seed on disk)\n"
                    "Run notebook 01 with `SEED=123` for the core models first, then this."))
    cells.append(code("""
from rhp.prediction import test_retest
from scripts._common import save_json
rep1, rep2 = collect(42), collect(123)
if rep2:
    tr = test_retest(rep1, rep2)
    save_json({'seed_a':42,'seed_b':123,'test_retest':tr.to_dict(orient='records')},
              RD/'analysis'/'test_retest_e9.json')
    print(tr.to_string(index=False))
else:
    print('No seed-123 results yet — run notebook 01 with SEED=123 for the core models.')
"""))
    return notebook(cells, gpu=False)


# ---------------------------------------------------------------------------
# Notebook 05 — robustness + optional (R-series, O-series)
# ---------------------------------------------------------------------------

def nb_robustness() -> dict:
    cells = [md(
        "# 05 · Robustness & optional (R-series · O-series)\n"
        "Targeted robustness checks that re-use the same helpers with different "
        "knobs. Each task is independent and resume-safe; run only the ones you "
        "need.\n\n"
        "| Task | Proposal item | What changes |\n|---|---|---|\n"
        "| R1 | threshold robustness | re-derive heads at τ∈{.05,.1,.2,.3} from saved scores (no GPU) |\n"
        "| R3 | coverage robustness | freq signature at coverage∈{.3,.5,1.0} on 3 core models |\n"
        "| R4 | quantization robustness | fp16 vs 8-bit profile on 2 core models |\n"
        "| R6 | sample-size sensitivity | profile at n=100 vs 200 |\n"
        "| R7 | haystack-source robustness | profile with an alternative corpus |\n"
        "| O5 | attention-mass score | third detector sign-agreement from saved scores (no GPU) |")]
    cells += setup_cells()
    cells.append(md("## R1 (CPU) — threshold robustness from saved score matrices\n"
                    "Re-counts heads and recomputes Gini/layer-COM at each τ without "
                    "touching a GPU — the score matrices are already saved."))
    cells.append(code("""
import json, glob, numpy as np, pandas as pd
from pathlib import Path
from rhp.profile import concentration, layer_profile
RD = Path(RESULTS_DIR); SEED = 42
rows = []
for f in sorted(glob.glob(str(RD/'profile'/f'*_seed{SEED}.json'))):
    r = json.load(open(f)); S = np.asarray(r['argmax_scores'])
    for tau in [0.05, 0.1, 0.2, 0.3]:
        ls, hs = np.where(S >= tau); heads = list(zip(ls.tolist(), hs.tolist()))
        rows.append(dict(model=r['model'], tau=tau, n_heads=len(heads),
                         gini=round(concentration(S, tau)['gini'],3),
                         layer_com=round(layer_profile(S, heads)['layer_com_weighted'],3)))
print(pd.DataFrame(rows).to_string(index=False) if rows else 'No profiles yet.')
"""))
    cells.append(md("## R3 (GPU) — coverage robustness of the frequency signature\n"
                    "Re-runs E2 at coverage ∈ {0.3, 0.5, 1.0} for 3 core models, "
                    "guarding against a false-null from too-small a patched population."))
    cells.append(code("""
import time, json
from pathlib import Path
import numpy as np
from src.model_loader import load_model, purge_hf_cache
from src.retrieval_head_detector import RetrievalHeadDetector
from src.dimension_utility import DimensionUtilityAnalyzer
from src.activation_patching import ActivationPatcher
from src.repro import set_determinism
from rhp.freq_signature import frequency_signature
from rhp.panel import load_panel, model_cfg
from scripts._common import save_json
import gc, torch

config = load_panel(CONFIG)
MODELS = ['qwen25_7b', 'llama31_8b', 'olmo2_7b']
SEED = 42; CTX = 4096
out_dir = Path(RESULTS_DIR)/'robustness'/'R3_coverage'; out_dir.mkdir(parents=True, exist_ok=True)
for key in MODELS:
    out = out_dir / f'{key}.json'
    if out.exists(): print(key, 'done'); continue
    cfg = model_cfg(config, key); m = t = None
    try:
        set_determinism(SEED); m, t = load_model(cfg, key)
        det = RetrievalHeadDetector(m, t, config, score_threshold=config['niah']['score_threshold'], seed=SEED)
        s = det.score_heads(det.generate_niah_samples(120, [CTX], [0.1,0.25,0.5,0.75,0.9]))
        heads = det.get_retrieval_heads(s)
        an = DimensionUtilityAnalyzer(m, config); patcher = ActivationPatcher(m, t, config)
        samp = det.generate_niah_samples(120, [CTX], [0.5])
        res = {}
        for cov in [0.3, 0.5, 1.0]:
            res[str(cov)] = frequency_signature(patcher, retrieval_heads=heads, retrieval_scores=s,
                freq_order=an.freq_order, head_dim=an.head_dim, samples=samp,
                n_heads=m.config.num_attention_heads, coverage=cov, seed=SEED)
            print(key, 'coverage', cov, 'freq_com=', res[str(cov)]['freq_com'])
        save_json(res, out)
    except Exception as e:
        print(key, 'FAILED', e)
    finally:
        m = t = None; gc.collect(); torch.cuda.empty_cache()
        try: purge_hf_cache(cfg['hf_id'])
        except Exception: pass
"""))
    cells.append(md("## R4 (GPU) — fp16 vs 8-bit profile on 2 core models\n"
                    "Closes the Part-1 'future work' item: does quantization move the "
                    "profile? Loads the same model twice (8-bit and fp16) and compares "
                    "head-set Jaccard. fp16 7B needs ~16 GB — fine on L4/A100."))
    cells.append(code("""
import json
from pathlib import Path
import numpy as np
from src.model_loader import load_model, purge_hf_cache
from src.repro import set_determinism
from src.retrieval_head_detector import RetrievalHeadDetector
from src.stats_utils import jaccard
from rhp.panel import load_panel, model_cfg
from scripts._common import save_json
import gc, torch

config = load_panel(CONFIG)
MODELS = ['qwen25_7b', 'olmo2_7b']
SEED = 42; CTX = 4096
out_dir = Path(RESULTS_DIR)/'robustness'/'R4_quant'; out_dir.mkdir(parents=True, exist_ok=True)
for key in MODELS:
    out = out_dir/f'{key}.json'
    if out.exists(): print(key,'done'); continue
    base = model_cfg(config, key); rec = {}
    for tag, eight in [('int8', True), ('fp16', False)]:
        cfg = dict(base); cfg['load_in_8bit'] = eight; m = t = None
        try:
            set_determinism(SEED); m, t = load_model(cfg, key, load_in_8bit=eight)
            det = RetrievalHeadDetector(m, t, config, score_threshold=config['niah']['score_threshold'], seed=SEED)
            s = det.score_heads(det.generate_niah_samples(120, [CTX], [0.1,0.25,0.5,0.75,0.9]))
            rec[tag] = det.get_retrieval_heads(s)
        finally:
            m = t = None; gc.collect(); torch.cuda.empty_cache()
            try: purge_hf_cache(cfg['hf_id'])
            except Exception: pass
    j = jaccard(rec.get('int8', []), rec.get('fp16', []))
    save_json({'model': key, 'jaccard_int8_fp16': j['jaccard'],
               'n_int8': j['n_a'], 'n_fp16': j['n_b']}, out)
    print(key, 'int8 vs fp16 Jaccard =', round(j['jaccard'], 3))
"""))
    cells.append(md("## R6 (GPU) — sample-size sensitivity (100 vs 200)\n"
                    "Quantifies the real floor of the \"10-minute scan\" claim: how "
                    "much does the detected head set move between a 100-sample and a "
                    "200-sample profile? Reports the head-set Jaccard per model."))
    cells.append(code("""
import json
from pathlib import Path
import numpy as np, gc, torch
from src.model_loader import load_model, purge_hf_cache
from src.repro import set_determinism
from src.retrieval_head_detector import RetrievalHeadDetector
from src.stats_utils import jaccard
from rhp.panel import load_panel, model_cfg
from scripts._common import save_json

config = load_panel(CONFIG)
MODELS = ['qwen25_7b', 'llama31_8b', 'olmo2_7b']
SEED = 42; CTX = 4096; POS = [0.1,0.25,0.5,0.75,0.9]
out_dir = Path(RESULTS_DIR)/'robustness'/'R6_nsamples'; out_dir.mkdir(parents=True, exist_ok=True)
for key in MODELS:
    out = out_dir/f'{key}.json'
    if out.exists(): print(key,'done'); continue
    cfg = model_cfg(config, key); m = t = None
    try:
        set_determinism(SEED); m, t = load_model(cfg, key)
        det = RetrievalHeadDetector(m, t, config, score_threshold=config['niah']['score_threshold'], seed=SEED)
        heads = {}
        for n in (100, 200):
            s = det.score_heads(det.generate_niah_samples(n, [CTX], POS))
            heads[n] = det.get_retrieval_heads(s)
        j = jaccard(heads[100], heads[200])
        save_json({'model':key,'jaccard_100_200':j['jaccard'],'n_100':j['n_a'],'n_200':j['n_b']}, out)
        print(key, '100-vs-200 Jaccard =', round(j['jaccard'],3))
    except Exception as e:
        print(key,'FAILED',e)
    finally:
        m = t = None; gc.collect(); torch.cuda.empty_cache()
        try: purge_hf_cache(cfg['hf_id'])
        except Exception: pass
"""))
    cells.append(md("## R7 (GPU) — haystack-source robustness\n"
                    "Re-profiles 3 core models with an **alternative neutral corpus** "
                    "instead of PG-19, to rule out corpus dependence. We swap "
                    "`src.corpus`'s process-wide cache before detection; restore it "
                    "afterwards."))
    cells.append(code("""
import json
from pathlib import Path
import numpy as np, gc, torch
import src.corpus as corpus
from src.model_loader import load_model, purge_hf_cache
from src.repro import set_determinism
from src.retrieval_head_detector import RetrievalHeadDetector
from src.stats_utils import jaccard
from rhp.panel import load_panel, model_cfg
from scripts._common import save_json

# Alternative neutral corpus (WikiText-103 sentences); falls back if offline.
def alt_corpus(n=5000):
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext','wikitext-103-raw-v1',split='train',streaming=True)
        out=[]
        for ex in ds:
            for s in ex['text'].split('. '):
                s=s.strip()
                if 20 < len(s) < 200: out.append(s+'.')
                if len(out)>=n: break
            if len(out)>=n: break
        return out or corpus.FALLBACK_SENTENCES*40
    except Exception as e:
        print('wikitext unavailable, using shuffled fallback:', e)
        return corpus.FALLBACK_SENTENCES*40

config = load_panel(CONFIG)
MODELS = ['qwen25_7b','llama31_8b','olmo2_7b']
SEED=42; CTX=4096; POS=[0.1,0.25,0.5,0.75,0.9]
out_dir = Path(RESULTS_DIR)/'robustness'/'R7_haystack'; out_dir.mkdir(parents=True, exist_ok=True)
ALT = alt_corpus()
for key in MODELS:
    out = out_dir/f'{key}.json'
    if out.exists(): print(key,'done'); continue
    cfg = model_cfg(config, key); m = t = None
    try:
        # load PG-19 baseline heads from the panel profile if present
        prof = Path(RESULTS_DIR)/'profile'/f'{key}_seed{SEED}.json'
        base_heads = json.load(open(prof))['argmax_heads'] if prof.exists() else None
        set_determinism(SEED); m, t = load_model(cfg, key)
        corpus._CORPUS_CACHE = list(ALT)        # swap corpus
        det = RetrievalHeadDetector(m, t, config, score_threshold=config['niah']['score_threshold'], seed=SEED)
        s = det.score_heads(det.generate_niah_samples(120, [CTX], POS))
        alt_heads = det.get_retrieval_heads(s)
        rec = {'model':key,'n_alt_heads':len(alt_heads)}
        if base_heads is not None:
            j = jaccard([tuple(h) for h in base_heads], alt_heads)
            rec['jaccard_pg19_alt'] = j['jaccard']
            print(key,'PG19-vs-alt Jaccard =', round(j['jaccard'],3))
        save_json(rec, out)
    except Exception as e:
        print(key,'FAILED',e)
    finally:
        corpus._CORPUS_CACHE = None             # restore
        m = t = None; gc.collect(); torch.cuda.empty_cache()
        try: purge_hf_cache(cfg['hf_id'])
        except Exception: pass
"""))
    cells.append(md("## O5 (CPU) — attention-mass as a third detector (sign agreement)\n"
                    "The argmax detector already records a continuous mass score via "
                    "`return_mass=True`; here we just confirm sign-agreement of the "
                    "head ranking from the saved argmax scores as a placeholder. To "
                    "compute the true mass detector, re-run E1 with `return_mass=True` "
                    "and store `scores['mass']`."))
    cells.append(code("""
print('O5 is a reanalysis hook: when you run E1 with return_mass=True, save the '
      'mass matrix alongside argmax/copy and add it as a third column in the '
      'detector-agreement table. No extra GPU pass is required.')
"""))
    cells.append(md("### O1–O4 (optional showcase)\n"
                    "O1 surgical band extension, O2 OLMo-2 checkpoint inheritance, O3 "
                    "merged-model profile, O4 long-context fine-tune variant — each is "
                    "a single-model add-on. Implement by pointing `run_profile_for_model` "
                    "at the relevant checkpoint/merge HF id (add it to `panel.yaml`) and "
                    "comparing profiles with `rhp.inheritance.compare_identity`."))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

def _clean_old_notebooks() -> None:
    """Remove previously-emitted .ipynb so a re-chunk doesn't leave stragglers."""
    for f in NB_DIR.glob("*_colab.ipynb"):
        f.unlink()


def main() -> None:
    _clean_old_notebooks()
    order = panel_model_order()
    prof_chunks = chunks(order, PROFILE_CHUNK)
    beh_chunks = chunks(order, BEHAVIOR_CHUNK)

    notebooks: dict[str, dict] = {"00_pilot_colab.ipynb": nb_pilot()}

    # Profile chunks: 01a, 01b, 01c, ...
    for i, ck in enumerate(prof_chunks):
        notebooks[f"01{chr(ord('a')+i)}_profile_chunk_colab.ipynb"] = \
            nb_profile_chunk(ck, i, len(prof_chunks))
    # Behaviour chunks: 02a, 02b, ...
    for i, ck in enumerate(beh_chunks):
        notebooks[f"02{chr(ord('a')+i)}_behavior_chunk_colab.ipynb"] = \
            nb_behavior_chunk(ck, i, len(beh_chunks))

    notebooks["03_inheritance_colab.ipynb"] = nb_inheritance()
    notebooks["04_prediction_analysis_colab.ipynb"] = nb_prediction()
    notebooks["05_robustness_optional_colab.ipynb"] = nb_robustness()

    for name, nb in notebooks.items():
        path = NB_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path, f"({len(nb['cells'])} cells)")
    print(f"\n{len(prof_chunks)} profile chunk(s) x {PROFILE_CHUNK} models, "
          f"{len(beh_chunks)} behaviour chunk(s) x {BEHAVIOR_CHUNK} models; "
          f"<= {TIME_BUDGET_HOURS} h each.")


if __name__ == "__main__":
    main()
