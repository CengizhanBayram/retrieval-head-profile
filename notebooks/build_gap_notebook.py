"""
Generate the SINGLE all-in-one gap-fill notebook (09) that closes every
remaining hole found in the result sweep, in one runnable file.

This generator ONLY writes ``09_complete_gaps_colab.ipynb`` — it can never touch
00–08. It reuses the canonical setup cells from build_notebooks.py so the clone /
deps / paths are byte-identical to the other notebooks.

Run:  python notebooks/build_gap_notebook.py

What 09 closes (all of it, resume-safe, one GPU):
  • qwen AWQ + GPTQ rings  -> profile + behaviour + utility  (E14 was {awq:null,gptq:null})
  • gemma2_9b_it_bnb4 ring  -> profile + behaviour + utility  (gemma had no 4-bit ring)
  • phi35_mini             -> profile + utility  (only behaviour existed; 8th family was absent)
  • mistral base           -> utility  (7a ran before 7c made its profile, so it was skipped)
  • qwen2.5-3B-instruct    -> E2 re-run at freq_coverage=1.0 (its freq_com read NaN)
  • re-run prediction + inheritance + a coverage/E14 verification read-out

GPU: A100 40 GB (80 GB ideal). Reason: AWQ/GPTQ kernels are mature on A100 but may
be missing on sm_120 (Blackwell); gemma's 256-dim KV needs >=40 GB. One A100 does
all five tasks. (L4 can do everything EXCEPT gemma2_9b_it_bnb4.)

OUTPUT FOLDER: writes to a SEPARATE 'rhprofile_results_other' test folder, seeded
no-clobber from the main 'rhprofile_results'. The main folder is never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import (  # reuse identical helpers + setup cells
    md, code, notebook, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
)

# ---------------------------------------------------------------------------
# Cell 0 — GPU + Drive + the SEPARATE test results dir (main stays read-only)
# ---------------------------------------------------------------------------
GPU_DRIVE_TEST = code("""
# Cell 0 — GPU + Drive + TEST results dir
import subprocess, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print(subprocess.check_output('nvidia-smi', shell=True).decode())
from google.colab import drive
drive.mount('/content/drive')

# This notebook writes to a SEPARATE 'other/test' folder so the main results are
# never touched. It is seeded (no-clobber) from the main folder below, so utility
# and the final analysis still see the FULL panel + the new rings.
RESULTS_DIR = '/content/drive/MyDrive/rhprofile_results_other'   # <- write target (test)
MAIN_DIR    = '/content/drive/MyDrive/rhprofile_results'         # <- read-only source
os.makedirs(RESULTS_DIR, exist_ok=True)
print('TEST results dir :', RESULTS_DIR)
print('MAIN (read-only) :', MAIN_DIR)
""")

# ---------------------------------------------------------------------------
# AWQ / GPTQ kernels (best-effort; failures are isolated to those two rings)
# ---------------------------------------------------------------------------
AWQ_GPTQ_INSTALL = code("""
%%bash
# AWQ/GPTQ kernels for the qwen 4-bit rings. **CRITICAL: --no-deps.** A normal
# install pulls transformers 5.x + a mismatched numpy, which breaks scipy
# ('cannot import name _center') and kills EVERY task (even phi/mistral/gemma).
# --no-deps installs ONLY the kernel wheels, leaving numpy/scipy/transformers
# (4.47) untouched. Best-effort: a missing kernel only skips the 2 qwen rings.
echo '== installing AWQ/GPTQ kernels (--no-deps; env stays pinned) =='
pip install -q --no-deps autoawq    2>&1 | tail -1 || echo 'autoawq failed (AWQ ring will skip)'
pip install -q --no-deps gptqmodel  2>&1 | tail -1 || echo 'gptqmodel failed (GPTQ ring will skip)'
echo '== integrity: numpy/scipy/transformers MUST still import =='
python - <<'PYEOF'
import importlib
def ok(m):
    try: importlib.import_module(m); return True
    except Exception as e: print('BROKEN', m, '->', str(e)[:90]); return False
core_ok = ok('numpy') and ok('scipy.stats') and ok('transformers')
if not core_ok:
    print('CORE BROKEN -> Runtime > Restart session, then Run all again. '
          '(--no-deps should prevent this; report if it persists.)')
else:
    import transformers, numpy
    print('core OK: transformers', transformers.__version__, '| numpy', numpy.__version__)
for m in ['awq', 'gptqmodel']:
    try: importlib.import_module(m); print('OK  ', m)
    except Exception as e: print('MISS', m, '->', str(e)[:90])
print('NOTE: prebuilt kernels may be missing for very new torch/CUDA or '
      'Blackwell/sm_120 -> the qwen rings then skip (the rest still completes).')
PYEOF
""")

# ---------------------------------------------------------------------------
# Seed the test folder from main (no-clobber) so analysis sees the full panel
# ---------------------------------------------------------------------------
SEED_FROM_MAIN = code("""
# Seed the TEST folder from the MAIN one (NO-CLOBBER): copies the existing panel
# into the test folder WITHOUT overwriting anything already produced here, so the
# final analysis + mistral utility see the complete set. Main is only ever READ.
import os, shutil
SEED = True   # set False to keep ONLY this notebook's new artifacts in the test folder
if SEED and os.path.isdir(MAIN_DIR):
    n = 0
    for root, _dirs, files in os.walk(MAIN_DIR):
        rel = os.path.relpath(root, MAIN_DIR)
        dst = os.path.join(RESULTS_DIR, rel); os.makedirs(dst, exist_ok=True)
        for fn in files:
            d = os.path.join(dst, fn)
            if not os.path.exists(d):           # never clobber test-folder work (resume-safe)
                shutil.copy2(os.path.join(root, fn), d); n += 1
    print(f'Seeded {n} new files from main -> test (no-clobber). Main untouched.')
else:
    print('Main folder not found or seeding disabled; test folder holds only new artifacts.')
""")

# ---------------------------------------------------------------------------
# The unified resume-safe driver over every gap task
# ---------------------------------------------------------------------------
DRIVER = code("""
# Unified gap-fill driver — built for "Runtime -> Run all", unattended. ALL heavy
# work is under ONE adaptive 23 h guard, so if time runs short it stops cleanly
# and nothing heavy runs after it. Resume-safe (skip-if-exists): if it stops
# early, just press Run all AGAIN -> finished work is skipped and it resumes.
import time, json, math
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json, time_guard)
from rhp.panel import load_panel, model_cfg

config = load_panel(CONFIG); SEED = 42
RD = Path(RESULTS_DIR)
prof = RD/'profile'; prof.mkdir(parents=True, exist_ok=True)
beh  = RD/'behavior'; beh.mkdir(parents=True, exist_ok=True)
util = RD/'utility';  util.mkdir(parents=True, exist_ok=True)

# (key, do_profile, do_behavior, do_utility, attn_impl, first_est_h)
# Order: the RELIABLE base-env tasks first (no special kernels) so they always
# complete; the fragile AWQ/GPTQ rings (prebuilt-kernel dependent) run LAST, so a
# kernel problem never blocks the rest.
TASKS = [
    ('phi35_mini',               True,  False, True,  None,   4.0),  # behaviour already on Drive
    ('mistral7b_v03',            False, False, True,  None,   1.5),  # profile+behaviour exist; only M7
    ('gemma2_9b_it_bnb4',        True,  True,  True,  'sdpa', 8.0),  # heaviest: 256-dim KV -> A100 (bnb)
    ('qwen25_7b_instruct_awq4',  True,  True,  True,  None,   8.0),  # CRITICAL: E14 AWQ (needs autoawq)
    ('qwen25_7b_instruct_gptq4', True,  True,  True,  None,   8.0),  # CRITICAL: E14 GPTQ (needs gptqmodel)
]

def heads_from_profile(key):
    d = json.load(open(prof/f'{key}_seed{SEED}.json', encoding='utf-8'))
    return d['argmax_heads'], d['argmax_scores']

start = time.time(); times = []
for key, dp, db, du, attn, est in TASKS:
    pout = prof/f'{key}_seed{SEED}.json'
    bout = beh/f'{key}_seed{SEED}.json'
    uout = util/f'{key}_seed{SEED}.json'
    need = (dp and not pout.exists()) or (db and not bout.exists()) or (du and not uout.exists())
    if not need:
        print(key, 'all done -> skip'); continue
    ok, el, eh = time_guard(start, times, first_est_h=est)
    if not ok:
        print(f'STOP before {key}: {el:.1f}h + est {eh:.1f}h would cross 23h. Press Run all again to resume.')
        break
    t0 = time.time()
    cfg = dict(model_cfg(config, key))
    if attn: cfg['attn_implementation'] = attn
    try:
        if dp and not pout.exists():
            save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), pout)
            print(key, 'profile saved')
        if db and not bout.exists():
            r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
            save_json(r, bout); print(key, 'behaviour saved')
        if du and not uout.exists():
            ah, asc = heads_from_profile(key)        # from the profile just saved / pre-existing
            save_json(run_utility_for_model(key, cfg, config, argmax_heads=ah,
                                            argmax_scores=asc, seed=SEED), uout)
            print(key, 'utility saved')
        times.append((time.time()-t0)/3600)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(key, 'FAILED ->', e, '| continuing with the next task.')
else:
    # for-else: only runs if the loop did NOT break (all main tasks done this
    # session) -> the qwen2.5-3B-instruct E2 fix, under the SAME guard.
    key = 'qwen25_3b_instruct'; out = prof/f'{key}_seed{SEED}.json'
    cur = (json.load(open(out, encoding='utf-8'))['profile']['scalars'].get('freq_com')
           if out.exists() else None)
    if cur is not None and not (isinstance(cur, float) and math.isnan(cur)):
        print('qwen-3B freq_com already real (', cur, ') -> skip')
    else:
        ok, el, eh = time_guard(start, times, first_est_h=4.0)
        if not ok:
            print('STOP before qwen-3B fix; press Run all again to finish it.')
        else:
            try:
                res = run_profile_for_model(key, model_cfg(config, key), config, seed=SEED,
                                            context_length=4096, freq_coverage=1.0)
                save_json(res, out)
                print('qwen-3B freq_com now =', res['profile']['scalars'].get('freq_com'))
            except Exception as e:
                import traceback; traceback.print_exc(); print('qwen-3B fix FAILED:', e)
print('Driver pass complete.')
""")

# ---------------------------------------------------------------------------
# Re-run analysis on the TEST folder + verify coverage / E14
# ---------------------------------------------------------------------------
ANALYZE_VERIFY = code("""
# Re-run E8 prediction + E9 reliability + E10-E15 inheritance on the TEST folder,
# then print a coverage + E14 verification read-out. (CPU work; fine on the GPU
# runtime.) After this, download rhprofile_results_other to your local datas/.
import subprocess, sys, json
from pathlib import Path
P2 = '/content/rope-part2'
def run(args):
    cmd = [sys.executable] + args + ['--results-dir', RESULTS_DIR, '--config', CONFIG,
                                     '--part1-repo', '/content/rope-part1']
    print('>>', ' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-1200:] if r.returncode else '')

run([f'{P2}/scripts/run_prediction.py', '--seed', '42', '--retest-seed', '123'])
run([f'{P2}/scripts/run_inheritance.py', '--lineage', 'all', '--seed', '42'])

RD = Path(RESULTS_DIR)
def mark(sub, key): return 'Y' if (RD/sub/f'{key}_seed42.json').exists() else '.'
print('\\n==== COVERAGE (test folder) ====')
for key in ['phi35_mini', 'mistral7b_v03', 'qwen25_7b_instruct_awq4',
            'qwen25_7b_instruct_gptq4', 'gemma2_9b_it_bnb4']:
    print(f'  {key:28s} prof={mark(\"profile\",key)}  beh={mark(\"behavior\",key)}  util={mark(\"utility\",key)}')

q = json.load(open(RD/'inheritance'/'qwen.json', encoding='utf-8'))
e14 = q.get('E14_quant_ablation', {})
print('\\nE14 quant ablation:  awq4=%s  gptq4=%s'
      % ('set' if e14.get('awq4') else 'NULL', 'set' if e14.get('gptq4') else 'NULL'))
print('qwen rings :', [(r.get('parent'), r.get('child')) for r in q.get('rings', [])])
g = json.load(open(RD/'inheritance'/'gemma.json', encoding='utf-8'))
print('gemma rings:', [(r.get('parent'), r.get('child')) for r in g.get('rings', [])])
sc = json.load(open(RD/'profile'/'qwen25_3b_instruct_seed42.json', encoding='utf-8'))['profile']['scalars']
print('qwen-3b-inst freq_com:', sc.get('freq_com'))
print('\\nDone. Coverage above should be all-Y (AWQ/GPTQ only if their kernels installed).')
""")


def nb_09_complete_gaps():
    cells = [md(
        "# 09 · Complete the gaps (one runnable file) — qwen AWQ/GPTQ + gemma 4-bit "
        "+ phi + mistral utility + qwen-3B fix\n"
        "**GPU: A100 40 GB (80 GB ideal).** AWQ/GPTQ kernels are mature on A100 but "
        "may be missing on Blackwell/sm_120; gemma's 256-dim KV needs ≥40 GB. One "
        "A100 runs all five tasks. *(L4 can do everything except `gemma2_9b_it_bnb4`.)*\n\n"
        "**Writes to a SEPARATE `rhprofile_results_other` (test) folder** — the main "
        "`rhprofile_results` is never touched. The test folder is seeded no-clobber "
        "from main so the final analysis sees the full panel.\n\n"
        "### How to run it: `Runtime → Run all`, then walk away.\n"
        "One-time before you press it: **(1)** paste your **HF token** in Cell 2 "
        "(needed for gated gemma); **(2)** at the very start, click the Drive "
        "permission popup once. After that it runs top-to-bottom unattended.\n\n"
        "Resume-safe & idempotent: every sub-step skips if its file exists, and an "
        "adaptive 23 h guard stops before Colab's 24 h limit. **If it doesn't finish "
        "in one session, just press `Run all` again** — done work is skipped and it "
        "resumes.\n\n"
        "| Task | What it closes | GPU |\n|---|---|---|\n"
        "| phi35_mini profile+utility | 8th family was absent from RQ1/RQ2 | A100/L4 |\n"
        "| mistral base utility | M7 was skipped (ran before its profile) | A100/L4 |\n"
        "| qwen AWQ ring (prof+beh+util) | **E14 AWQ (was null)** | A100/L4 |\n"
        "| qwen GPTQ ring (prof+beh+util) | **E14 GPTQ (was null)** | A100/L4 |\n"
        "| gemma2_9b_it_bnb4 ring | gemma had no 4-bit ring | **A100** |\n"
        "| qwen-3B-inst E2 re-run | freq_com read NaN | A100/L4 |")]
    cells += [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + paste tokens in Cell 2."),
        GPU_DRIVE_TEST, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## 1 — AWQ/GPTQ kernels (for the qwen 4-bit rings; best-effort)"))
    cells.append(AWQ_GPTQ_INSTALL)
    cells.append(md("## 2 — Seed the test folder from main (no-clobber; main stays read-only)"))
    cells.append(SEED_FROM_MAIN)
    cells.append(md("## 3 — Gap-fill driver (all heavy work; resume-safe; includes the qwen-3B fix)"))
    cells.append(DRIVER)
    cells.append(md("## 4 — Re-run analysis on the test folder + verify coverage/E14"))
    cells.append(ANALYZE_VERIFY)
    return notebook(cells, gpu=True)


def main():
    name = "09_complete_gaps_colab.ipynb"
    nb = nb_09_complete_gaps()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-08 untouched. GPU: A100 40GB (80GB ideal). Writes to rhprofile_results_other.")


if __name__ == "__main__":
    main()
