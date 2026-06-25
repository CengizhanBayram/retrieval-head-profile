"""
Generate notebook 12 — E14 cross-method quant (GPTQ via Triton, AWQ best-effort)
using the RESTART approach (the user's instinct).

Why restart: the '_center' numpy crash in runs 09/12-v1 is the classic "package
upgraded mid-session, old C-extension still loaded" stale-module problem. The fix
is to install the stack, then RESTART the runtime so the new numpy/transformers
load cleanly. (A fresh interpreter has no stale numpy.)

Kernel reality on the base torch 2.11 stack:
  • GPTQ via `gptqmodel` uses TRITON kernels (compiled at runtime) -> works on
    torch 2.11. So restart + transformers 4.51 + gptqmodel  ==> GPTQ is feasible.
  • AWQ via `autoawq` needs PREBUILT CUDA kernels that don't exist for torch 2.11,
    so AWQ is best-effort here (use the venv/torch-2.6 path if AWQ is required).

So this notebook realistically yields the **bnb4 vs GPTQ** cross-method E14 result
(a real finding); AWQ is attempted but may skip.

FLOW (NOT a single Run-all — one manual restart in the middle):
  Cell A: install (transformers 4.51.3 + gptqmodel + autoawq) -> then RESTART.
  After restart: mount/clone/paths -> SRC-COMPAT smoke test -> rings -> E14 table.

ONLY writes 12_e14_awq_gptq_colab.ipynb. Writes to rhprofile_results_other.
GPU: A100. qwen2.5 is ungated (no HF token needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import md, code, notebook, SETUP_CLONE, SETUP_PATHS
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN

# Cell A — install the stack, then the user RESTARTS. Kernels first, then pin
# transformers + a consistent numpy LAST so they win the resolution.
INSTALL_THEN_RESTART = code("""
%%bash
# STEP 1 of 2. Install the AWQ/GPTQ stack, then RESTART THE RUNTIME (next cell
# tells you). gptqmodel = Triton kernels (work on torch 2.11). Kernels are
# installed first; transformers 4.51.3 + a consistent numpy are pinned LAST so the
# resolution ends on known-good versions. The restart clears the stale numpy that
# causes the '_center' crash.
echo '== kernels =='
pip install -q gptqmodel 2>&1 | tail -1 || echo 'gptqmodel failed'
pip install -q autoawq optimum 2>&1 | tail -1 || echo 'autoawq best-effort (needs torch 2.6 kernels)'
echo '== pin transformers 4.51.3 + consistent numpy/scipy LAST =='
pip install -q transformers==4.51.3 2>&1 | tail -1
pip install -q "numpy<2.2" scipy scikit-learn 2>&1 | tail -1
echo
echo '##################################################################'
echo '#  DONE. NOW: Runtime > Restart session (NOT Disconnect).        #'
echo '#  Then run every cell BELOW this one (do NOT re-run this cell).  #'
echo '##################################################################'
""")

RESTART_NOTE = md(
    "## ⛔ RESTART NOW — `Runtime → Restart session`\n"
    "Then run the cells **below** (skip the install cell above). The restart is what "
    "clears the stale numpy (`_center` fix). Re-running the install cell would "
    "re-trigger it — don't.")

# After restart: verify the fresh env imports cleanly + kernels present.
VERIFY = code("""
# Post-restart sanity: numpy/scipy/transformers import cleanly + kernels present.
import importlib
for m in ['numpy', 'scipy.stats', 'transformers', 'gptqmodel', 'awq', 'optimum']:
    try:
        importlib.import_module(m); print('OK  ', m)
    except Exception as e:
        print('MISS', m, '->', str(e)[:90])
import transformers, numpy, torch
print('transformers', transformers.__version__, '| numpy', numpy.__version__,
      '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
""")

# SRC-COMPAT smoke test under transformers 4.51 (stop early if src breaks).
SMOKE = code("""
# SRC-COMPAT smoke test under transformers 4.51. If this crashes, src/ is not
# compatible with 4.51 -> STOP and keep the bnb4-only quant story (notebook 11).
import gc, torch
from rhp.panel import load_panel, model_cfg
from rhp.loader import load_model_any
from src.retrieval_head_detector import RetrievalHeadDetector
config = load_panel(CONFIG); ok_src = False
try:
    m, tok = load_model_any(model_cfg(config, 'qwen25_7b_instruct'), 'qwen25_7b_instruct')
    det = RetrievalHeadDetector(m, tok, config, score_threshold=0.1, seed=42)
    s = det.generate_niah_samples(5, [2048], [0.5]); sc = det.score_heads(s)
    print('SMOKE OK -> src works under tf', __import__('transformers').__version__, '| matrix', sc.shape)
    ok_src = True; del m, tok, det
except Exception:
    import traceback; traceback.print_exc()
    print('SMOKE FAILED -> src not compatible with tf 4.51. STOP; keep bnb4-only (nb 11).')
finally:
    gc.collect(); torch.cuda.empty_cache()
""")

# Rings: GPTQ first (Triton, best chance on torch 2.11), AWQ best-effort.
RINGS = code("""
# GPTQ (gptqmodel/Triton — works on torch 2.11) first; AWQ best-effort (may skip
# without torch-2.6 kernels). Resume-safe; verbose on failure.
import time, json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json, time_guard)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; RD = Path(RESULTS_DIR)

if not ok_src:
    print('Smoke did not pass -> skipping rings.')
else:
    start = time.time(); times = []
    for key in ['qwen25_7b_instruct_gptq4', 'qwen25_7b_instruct_awq4']:
        prof = RD/'profile'/f'{key}_seed{SEED}.json'
        beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
        util = RD/'utility'/f'{key}_seed{SEED}.json'
        if prof.exists() and beh.exists() and util.exists():
            print(key, 'done -> skip'); continue
        ok, el, eh = time_guard(start, times, first_est_h=8.0)
        if not ok:
            print('STOP; re-run to resume.'); break
        t0 = time.time(); cfg = dict(model_cfg(config, key))
        try:
            if not prof.exists():
                save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
                print(key, 'profile saved')
            if not beh.exists():
                r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
                save_json(r, beh); print(key, 'behaviour saved')
            if not util.exists():
                d = json.load(open(prof, encoding='utf-8'))
                save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                                argmax_scores=d['argmax_scores'], seed=SEED), util)
                print(key, 'utility saved')
            times.append((time.time()-t0)/3600)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(key, 'FAILED ->', e, '(GPTQ should load on Triton; AWQ may need torch 2.6)')
    print('rings pass complete.')
""")

# E14 cross-method table — pure JSON read.
E14_TABLE = code("""
# E14 cross-method: how does each 4-bit METHOD preserve the qwen instruct retrieval
# circuit? copy-head Jaccard (identity) + freq_com. Pure JSON (no transformers).
import json
from pathlib import Path
RD = Path(RESULTS_DIR); SEED = 42
def prof(key):
    p = RD/'profile'/f'{key}_seed{SEED}.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None
def jac(a, b):
    sa = {tuple(x) for x in a}; sb = {tuple(x) for x in b}
    return len(sa & sb)/len(sa | sb) if (sa or sb) else float('nan')
ref = prof('qwen25_7b_instruct')
print('E14 — qwen instruct -> 4-bit, by METHOD:')
print(f\"  {'method':7s} {'copyJaccard':>12s} {'freqCom':>10s}\")
if ref is not None:
    rh = ref.get('copy_heads', [])
    for method, key in [('bnb4','qwen25_7b_instruct_bnb4'),
                        ('gptq4','qwen25_7b_instruct_gptq4'),
                        ('awq4','qwen25_7b_instruct_awq4')]:
        c = prof(key)
        if c is None:
            print(f'  {method:7s}  (missing)'); continue
        print(f'  {method:7s} {jac(rh, c.get(\"copy_heads\", [])):12.3f} '
              f'{str(c[\"profile\"][\"scalars\"].get(\"freq_com\")):>10s}')
print('\\n2+ methods present -> a real E14 cross-method result '
      '(do methods preserve the circuit differently?).')
""")


def nb_12_e14():
    cells = [md(
        "# 12 · E14 cross-method quant — RESTART approach (GPTQ via Triton; AWQ best-effort)\n"
        "**GPU: A100.** AWQ/GPTQ installs corrupt the base env (`numpy._center` + "
        "transformers churn). The clean fix is the one you noticed: **install, then "
        "RESTART the runtime** so the new numpy/transformers load fresh.\n\n"
        "Kernel reality on torch 2.11:\n"
        "- **GPTQ** via `gptqmodel` = **Triton** kernels (compiled at runtime) → "
        "works on torch 2.11. **This is the realistic E14 arm.**\n"
        "- **AWQ** via `autoawq` = prebuilt CUDA kernels that don't exist for torch "
        "2.11 → best-effort (use the venv/torch-2.6 route if AWQ is required).\n\n"
        "So this yields **bnb4 vs GPTQ** cross-method (a real result); AWQ may skip. "
        "AWQ/GPTQ are different methods from bnb4 — bnb4 does not substitute.\n\n"
        "**NOT a single Run-all — one manual restart in the middle:**\n"
        "1. Run **Cell A** (install). 2. `Runtime → Restart session`. 3. Run every "
        "cell **below** Cell A.\n\n"
        "A src-compat smoke test runs after restart; if src breaks under tf 4.51, "
        "STOP and keep the bnb4-only story (notebook 11). qwen2.5 is ungated.")]
    cells.append(md("## Cell A — install the stack (run this, THEN restart)"))
    cells.append(INSTALL_THEN_RESTART)
    cells.append(RESTART_NOTE)
    cells.append(md("### ↓↓↓ Run everything below AFTER the restart ↓↓↓"))
    cells += [GPU_DRIVE_TEST, SETUP_CLONE, SETUP_PATHS]
    cells.append(md("## Post-restart sanity (clean imports + kernels)"))
    cells.append(VERIFY)
    cells.append(md("## SRC-COMPAT smoke test (stop early if src breaks under tf 4.51)"))
    cells.append(SMOKE)
    cells.append(md("## Seed test folder (no-clobber) + run GPTQ/AWQ rings"))
    cells.append(SEED_FROM_MAIN)
    cells.append(RINGS)
    cells.append(md("## E14 cross-method table (instruct → bnb4 / gptq / awq)"))
    cells.append(E14_TABLE)
    return notebook(cells, gpu=True)


def main():
    name = "12_e14_awq_gptq_colab.ipynb"
    nb = nb_12_e14()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-11 untouched. GPU: A100. Restart-in-the-middle; GPTQ via Triton on torch 2.11.")


if __name__ == "__main__":
    main()
