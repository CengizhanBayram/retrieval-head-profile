"""
Generate the FIX / GAP-FILL Colab notebooks (06, 07, 08) for the problems found
in the first result sweep. These are NEW notebooks; this generator never touches
00–05 (it only writes 06_/07_/08_). Reuses the cell + setup helpers from
build_notebooks.py so clone/paths/tokens stay identical.

Run:  python notebooks/build_fix_notebooks.py

GPU per notebook (also stated in each notebook's header):
  06_reanalysis           -> NONE (CPU): re-run E8/E9/E10 analysis with the fixes
  07_fill_gaps_L4         -> L4 24 GB : M7 utility, qwen 4-bit rings, mistral base,
                                        extra seeds, qwen-3b-instruct E2 re-run
  08_gemma_longctx_A100   -> A100 40 GB (your 96 GB ideal): gemma 32k via sdpa
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import (  # reuse identical helpers
    md, code, notebook, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
)

# CPU-safe drive cell (nvidia-smi may be absent on a CPU runtime).
CPU_DRIVE = code("""
# Cell 0 — Drive + results dir (CPU runtime; no GPU needed)
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/MyDrive/rhprofile_results'
except Exception as e:
    RESULTS_DIR = '/content/rhprofile_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Results dir:', RESULTS_DIR)
""")

GPU_DRIVE = code("""
# Cell 0 — GPU + Drive + results dir
import subprocess, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print(subprocess.check_output('nvidia-smi', shell=True).decode())
from google.colab import drive
drive.mount('/content/drive')
RESULTS_DIR = '/content/drive/MyDrive/rhprofile_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Results dir:', RESULTS_DIR)
""")


def setup(gpu: bool):
    drive = GPU_DRIVE if gpu else CPU_DRIVE
    return [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + tokens in Cell 2."),
        drive, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]


# ---------------------------------------------------------------------------
# 06 — CPU re-analysis (E8 prediction + E9 reliability + E10 inheritance) with fixes
# ---------------------------------------------------------------------------

def nb_06_reanalysis():
    cells = [md(
        "# 06 · Re-analysis with fixes (CPU) — E8 prediction · E9 reliability · E10–E15 inheritance\n"
        "**GPU: NONE** (Runtime → Change runtime type → None). Pure CPU re-analysis\n"
        "of the results already on Drive, with the corrected code:\n"
        "- prediction: `freq_width` dropped (collinear), null-frequency gate on "
        "`freq_com`, **BH on the family-demeaned** correlations;\n"
        "- E9: saves **`reliability_e9.json`** (per-model R_self — the real ceiling), "
        "alongside the (still-NaN-until-≥4-models) `test_retest_e9.json`;\n"
        "- inheritance: cross-architecture distillation siblings now report "
        "architecture-invariant axes, **not** the meaningless head-set Jaccard.\n\n"
        "No models are run here — it just rebuilds the analysis JSONs. Re-run any "
        "time the profile/behaviour/utility files change.")]
    cells += setup(gpu=False)
    cells.append(md("## Re-run the two analysis scripts (they contain all the fixes)"))
    cells.append(code("""
import subprocess, sys
P2 = '/content/rope-part2'
def run(args):
    cmd = [sys.executable] + args + ['--results-dir', RESULTS_DIR,
           '--config', CONFIG, '--part1-repo', '/content/rope-part1']
    print('>>', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:]);  print(r.stderr[-1500:] if r.returncode else '')

# E8 prediction + E9 reliability (R_self needs the 2nd seed: --retest-seed 123)
run([f'{P2}/scripts/run_prediction.py', '--seed', '42', '--retest-seed', '123'])
# E10–E15 inheritance (sibling fix)
run([f'{P2}/scripts/run_inheritance.py', '--lineage', 'all', '--seed', '42'])
print('\\nWrote analysis/prediction_e8.json, analysis/reliability_e9.json, inheritance/*.json')
"""))
    cells.append(md("## Quick read of the cleaned outputs"))
    cells.append(code("""
import json
from pathlib import Path
RD = Path(RESULTS_DIR)
rel = RD/'analysis'/'reliability_e9.json'
if rel.exists():
    d = json.load(open(rel))
    print('E9 R_self (copy-Jaccard) per model:')
    for x in d['reliability']:
        print(f\"   {x['model']:16s} copyJ={x['copy_jaccard']:.3f}\")
an = json.load(open(RD/'analysis'/'prediction_e8.json'))
print('\\nFamily-demeaned (BH-corrected) — top by |rho|:')
fam = sorted([f for f in an['family_demeaned'] if f.get('n',0)>0],
             key=lambda f: -abs(f.get('within_family_spearman') or 0))
for f in fam[:6]:
    print(f\"   {f['predictor']:18s} rho={f['within_family_spearman']:+.3f} \"
          f\"p={f['p_value']:.3f} p_bh={f.get('p_adjusted_bh'):.3f}\"
          f\"{'  *' if f.get('significant_bh') else ''}\")
"""))
    return notebook(cells, gpu=False)


# ---------------------------------------------------------------------------
# 07 — L4 gap-fill: M7 utility, qwen 4-bit rings, mistral base, extra seeds, qwen-3b-inst E2
# ---------------------------------------------------------------------------

def nb_07_fillgaps():
    cells = [md(
        "# 07 · Fill the gaps (L4) — M7 utility · qwen 4-bit rings · mistral base · extra seeds\n"
        "**GPU: L4 24 GB.** Every task here is ≤9 B in 8-bit or 4-bit and fits 24 GB.\n"
        "Adaptive 23 h guard + Drive resume on each task — re-run to continue.\n\n"
        "Closes: M7 all-NaN (run utility), the missing **qwen AWQ/GPTQ** quant rings "
        "(RQ3 frequency-under-quant — the key test), **mistral base** profile (its "
        "ring), extra **seed-123** profiles for ≥4-model E9, and the "
        "**qwen2.5-3B-instruct** E2 re-run at full coverage (its signature read "
        "zero-drop despite a strong dose effect).")]
    cells += setup(gpu=True)

    cells.append(md("## 7a — M7 utility for every profiled model (cheap; weight-space, no generation)"))
    cells.append(code("""
import subprocess, sys
cmd = [sys.executable, '/content/rope-part2/scripts/run_utility.py', '--models', 'all',
       '--results-dir', RESULTS_DIR, '--config', CONFIG, '--part1-repo', '/content/rope-part1', '--seed', '42']
print('>>', ' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-1500:] if r.returncode else '')
"""))

    cells.append(md("## 7b — qwen 4-bit rings (AWQ + GPTQ): profile + behaviour  [needs `autoawq`/`optimum`]"))
    cells.append(code("""
%%bash
pip install -q autoawq optimum auto-gptq 2>/dev/null || echo 'AWQ/GPTQ kernels: install issues are OK to retry'
echo done
"""))
    cells.append(code("""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, run_behavior_for_model, save_json, time_guard
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42
RINGS = ['qwen25_7b_instruct_awq4', 'qwen25_7b_instruct_gptq4']
prof = Path(RESULTS_DIR)/'profile'; beh = Path(RESULTS_DIR)/'behavior'
prof.mkdir(parents=True, exist_ok=True); beh.mkdir(parents=True, exist_ok=True)
start = time.time(); times = []
for key in RINGS:
    cfg = model_cfg(config, key)
    pout = prof/f'{key}_seed{SEED}.json'; bout = beh/f'{key}_seed{SEED}.json'
    if pout.exists() and bout.exists(): print(key, 'done -> skip'); continue
    ok, el, est = time_guard(start, times, first_est_h=8.0)
    if not ok: print(f'STOP {key}: {el:.1f}h+{est:.1f}h>23h. Re-run.'); break
    t0 = time.time()
    try:
        if not pout.exists():
            save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), pout)
            print(key, 'profile saved')
        if not bout.exists():
            r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
            save_json(r, bout); print(key, 'behaviour saved')
        times.append((time.time()-t0)/3600)
    except Exception as e:
        print(key, 'FAILED:', e)
"""))

    cells.append(md("## 7c — mistral base profile + behaviour (completes the mistral ring)"))
    cells.append(code("""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, run_behavior_for_model, save_json, time_guard
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'mistral7b_v03'
cfg = model_cfg(config, key)
prof = Path(RESULTS_DIR)/'profile'/f'{key}_seed{SEED}.json'
beh  = Path(RESULTS_DIR)/'behavior'/f'{key}_seed{SEED}.json'
start = time.time()
try:
    if not prof.exists():
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print('mistral base profile saved')
    if not beh.exists():
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh); print('mistral base behaviour saved')
except Exception as e:
    print('FAILED:', e)
print('elapsed %.1f h' % ((time.time()-start)/3600))
"""))

    cells.append(md("## 7d — extra seed-123 profiles (so E9 has ≥4 paired models)\n"
                    "Adds qwen25_7b + gemma2_9b at seed 123 (llama31_8b/llama32_3b/qwen25_3b "
                    "already have it). gemma profile detection is at 4096 → fits L4."))
    cells.append(code("""
import time
from pathlib import Path
from scripts._common import run_profile_for_model, save_json, time_guard
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 123
MODELS = ['qwen25_7b', 'gemma2_9b']
OUT = Path(RESULTS_DIR)/'profile'
start = time.time(); times = []
for key in MODELS:
    out = OUT/f'{key}_seed{SEED}.json'
    if out.exists(): print(key, 'seed123 done -> skip'); continue
    ok, el, est = time_guard(start, times, first_est_h=6.0)
    if not ok: print('STOP; re-run to resume.'); break
    t0 = time.time()
    try:
        save_json(run_profile_for_model(key, model_cfg(config, key), config, seed=SEED, context_length=4096), out)
        times.append((time.time()-t0)/3600); print(key, 'seed123 saved')
    except Exception as e:
        print(key, 'FAILED:', e)
"""))

    cells.append(md("## 7e — qwen2.5-3B-instruct E2 re-run at full coverage (fix the zero-drop signature)\n"
                    "Its 8-window signature read zero drop (freq_com=NaN) though its dose "
                    "effect is −1.0; re-profiling with `freq_coverage=1.0` matches the "
                    "dose's head population. Overwrites that one profile."))
    cells.append(code("""
from pathlib import Path
from scripts._common import run_profile_for_model, save_json
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'qwen25_3b_instruct'
out = Path(RESULTS_DIR)/'profile'/f'{key}_seed{SEED}.json'
try:
    res = run_profile_for_model(key, model_cfg(config, key), config, seed=SEED,
                                context_length=4096, freq_coverage=1.0)
    save_json(res, out)
    print(key, 'freq_com now =', res['profile']['scalars']['freq_com'],
          '(was NaN); re-run notebook 06 to refresh the prediction/inheritance.')
except Exception as e:
    print('FAILED:', e)
"""))
    return notebook(cells, gpu=True)


# ---------------------------------------------------------------------------
# 08 — gemma long-context on A100/96GB via sdpa attention
# ---------------------------------------------------------------------------

def nb_08_gemma():
    cells = [md(
        "# 08 · Gemma long-context (A100) — 32k via sdpa attention + gemma seed-123\n"
        "**GPU: A100 40 GB minimum; your 96 GB card is ideal.** Gemma-2 defaults to "
        "*eager* attention (logit soft-capping), which materialises the full "
        "O(n²) attention matrix and OOMs at 16k–32k even on big GPUs. We force "
        "`attn_implementation='sdpa'` so gemma reaches 32k — turning its 16k/32k "
        "**NaN (unmeasured)** into a real recall number (it is 8k-native, so likely "
        "low, which is the honest result, not a missing one).\n\n"
        "> Select an A100 runtime (or your 96 GB). Resume-safe; overwrites only the "
        "gemma behaviour files.")]
    cells += setup(gpu=True)
    cells.append(md("## Re-run gemma behaviour at full 32k with sdpa"))
    cells.append(code("""
import time
from pathlib import Path
from scripts._common import run_behavior_for_model, save_json, time_guard
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42
GEMMA = ['gemma2_9b', 'gemma2_9b_it', 'gemma2_2b']   # 256-dim heads
OUT = Path(RESULTS_DIR)/'behavior'
start = time.time(); times = []
for key in GEMMA:
    out = OUT/f'{key}_seed{SEED}.json'
    ok, el, est = time_guard(start, times, first_est_h=6.0)
    if not ok: print('STOP; re-run to resume.'); break
    cfg = model_cfg(config, key)
    cfg['attn_implementation'] = 'sdpa'   # avoid eager O(n^2) OOM at long context
    t0 = time.time()
    try:
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, out)   # overwrite: 16k/32k now measured, not NaN
        b = r['behavior']
        print(f\"{key}: per_ctx={b.get('niah_per_context')} maxlen={b.get('niah_maxlen')}\")
        times.append((time.time()-t0)/3600)
    except Exception as e:
        print(key, 'FAILED:', e)
"""))
    cells.append(md("## (optional) gemma2_9b seed-123 profile — its own R_self denominator"))
    cells.append(code("""
from pathlib import Path
from scripts._common import run_profile_for_model, save_json
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG)
out = Path(RESULTS_DIR)/'profile'/'gemma2_9b_seed123.json'
if not out.exists():
    cfg = model_cfg(config, 'gemma2_9b'); cfg['attn_implementation'] = 'sdpa'
    save_json(run_profile_for_model('gemma2_9b', cfg, config, seed=123, context_length=4096), out)
    print('gemma2_9b seed123 profile saved (for its own E9 R_self)')
else:
    print('already done')
"""))
    return notebook(cells, gpu=True)


def main():
    out = {
        "06_reanalysis_cpu_colab.ipynb": nb_06_reanalysis(),
        "07_fill_gaps_L4_colab.ipynb": nb_07_fillgaps(),
        "08_gemma_longctx_A100_colab.ipynb": nb_08_gemma(),
    }
    for name, nb in out.items():
        with open(NB_DIR / name, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", name, f"({len(nb['cells'])} cells)")
    print("\n00-05 untouched. GPU: 06=NONE(CPU), 07=L4 24GB, 08=A100 40GB (96GB ideal).")


if __name__ == "__main__":
    main()
