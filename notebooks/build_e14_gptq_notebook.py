"""
Generate notebook 13 — E14 GPTQ arm (separate from AWQ, RESTART approach).

AWQ (nb 12) and GPTQ can't share a transformers: autoawq needs tf<=4.51.3, while
gptqmodel needs tf>=4.52 (transformers.masking_utils). So GPTQ gets its own
notebook with gptqmodel + the transformers it wants.

gptqmodel uses TRITON kernels (compiled at runtime) -> works on the base torch
2.11, no torch downgrade. The '_center' numpy crash is handled the same way: install
then RESTART so the fresh interpreter loads numpy cleanly. Installing gptqmodel +
its matching transformers TOGETHER means its generation monkeypatch arity matches
(the nb-12 failure was gptqmodel's patch vs the wrong, pinned tf 4.51.3).

RISK: gptqmodel pulls a recent transformers (>=4.52, maybe 5.x) and the inherited
src/ may not be compatible. A SRC-COMPAT SMOKE TEST runs after the restart and
stops early if so -> then GPTQ is not feasible on this stack and we keep the
3-family bnb4 (nb 11) + AWQ (nb 12, if it worked).

FLOW (one manual restart): Cell A install -> Runtime>Restart -> run the rest.
ONLY writes 13_e14_gptq_colab.ipynb. Writes to rhprofile_results_other. GPU: A100.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import md, code, notebook, SETUP_CLONE, SETUP_PATHS
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN

INSTALL_THEN_RESTART = code("""
%%bash
# STEP 1 of 2. GPTQ-ONLY install, then RESTART (next cell tells you). Install
# gptqmodel + optimum and let them pull the transformers they need (>=4.52, has
# masking_utils); installing them TOGETHER keeps gptqmodel's generation patch arity
# consistent with that transformers. gptqmodel uses Triton kernels -> torch 2.11 ok.
echo '== gptqmodel + optimum (pull their transformers >=4.52) =='
pip install -q gptqmodel optimum 2>&1 | tail -1 || echo 'gptqmodel install failed'
echo '== consistent numpy/scipy LAST (restart clears the stale _center) =='
pip install -q "numpy<2.2" scipy scikit-learn 2>&1 | tail -1
echo
echo '##################################################################'
echo '#  DONE. NOW: Runtime > Restart session (NOT Disconnect).        #'
echo '#  Then run every cell BELOW this one (do NOT re-run this cell).  #'
echo '##################################################################'
""")

RESTART_NOTE = md(
    "## ⛔ RESTART NOW — `Runtime → Restart session`\n"
    "Then run the cells **below** (skip the install cell above). The restart clears "
    "the stale numpy (`_center` fix). Don't re-run the install cell.")

VERIFY = code("""
# Post-restart sanity: imports clean + gptqmodel present + which transformers.
import importlib
for m in ['numpy', 'scipy.stats', 'transformers', 'gptqmodel', 'optimum']:
    try:
        importlib.import_module(m); print('OK  ', m)
    except Exception as e:
        print('MISS', m, '->', str(e)[:90])
import transformers, numpy, torch
print('transformers', transformers.__version__, '| numpy', numpy.__version__,
      '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
""")

SMOKE = code("""
# SRC-COMPAT smoke test under gptqmodel's transformers. If it crashes, src/ is not
# compatible -> STOP; keep 3-family bnb4 (nb 11) + AWQ (nb 12).
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
    print('SMOKE FAILED -> src not compatible. STOP; keep bnb4 (nb11) + AWQ (nb12).')
finally:
    gc.collect(); torch.cuda.empty_cache()
""")

RING = code("""
# GPTQ ring only (gptqmodel/Triton). Resume-safe; verbose on failure.
import json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; RD = Path(RESULTS_DIR)

if not ok_src:
    print('Smoke did not pass -> skipping GPTQ.')
else:
    key = 'qwen25_7b_instruct_gptq4'
    prof = RD/'profile'/f'{key}_seed{SEED}.json'
    beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
    util = RD/'utility'/f'{key}_seed{SEED}.json'
    # FORCE EAGER: the prior GPTQ run detected 0 heads ("No attention weights
    # captured") because the newer transformers default attention doesn't return
    # weights with output_attentions. Eager returns 4D weights -> detector works.
    cfg = dict(model_cfg(config, key)); cfg['attn_implementation'] = 'eager'
    try:
        # OVERWRITE old/partial files (the previous run saved a 0-head profile).
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print(key, 'profile saved [overwrite]')
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh); print(key, 'behaviour saved [overwrite]')
        d = json.load(open(prof, encoding='utf-8'))
        save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                        argmax_scores=d['argmax_scores'], seed=SEED), util)
        print(key, 'utility saved [overwrite]')
        print('GPTQ ring done. Check #heads > 0 (eager fixed detection) and freq '
              '(via dense-weight extraction).')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(key, 'FAILED ->', e)
""")

E14_TABLE = code("""
# E14 cross-method: instruct -> each 4-bit method. copy-head Jaccard + freq_com.
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
                        ('awq4','qwen25_7b_instruct_awq4'),
                        ('gptq4','qwen25_7b_instruct_gptq4')]:
        c = prof(key)
        if c is None:
            print(f'  {method:7s}  (missing — from its own notebook)'); continue
        print(f'  {method:7s} {jac(rh, c.get(\"copy_heads\", [])):12.3f} '
              f'{str(c[\"profile\"][\"scalars\"].get(\"freq_com\")):>10s}')
print('\\n2+ methods present -> a real E14 cross-method result.')
""")


def nb_13_gptq():
    cells = [md(
        "# 13 · E14 GPTQ arm (separate from AWQ) — RESTART approach\n"
        "**GPU: A100.** AWQ (nb 12) needs transformers ≤4.51.3; GPTQ needs ≥4.52 "
        "(`transformers.masking_utils`) — they **can't share** a transformers, so "
        "GPTQ is its own notebook. `gptqmodel` uses **Triton** kernels → works on "
        "the base torch 2.11 (no downgrade). Installing gptqmodel + its matching "
        "transformers TOGETHER keeps its generation monkeypatch consistent (the nb-12 "
        "AWQ failure was that patch vs a mismatched pinned tf 4.51.3).\n\n"
        "`_center` numpy crash → fixed by **install, then RESTART** (fresh numpy).\n\n"
        "**RISK:** gptqmodel may pull a recent transformers (≥4.52, maybe 5.x) that "
        "breaks the inherited `src/`. The smoke test after restart stops early if so "
        "— then GPTQ isn't feasible here and we keep 3-family bnb4 (nb 11) + AWQ "
        "(nb 12). Yields **bnb4 vs GPTQ** cross-method E14. qwen2.5 is ungated.\n\n"
        "**One manual restart:** run Cell A → `Runtime → Restart session` → run the rest.")]
    cells.append(md("## Cell A — install gptqmodel (run this, THEN restart)"))
    cells.append(INSTALL_THEN_RESTART)
    cells.append(RESTART_NOTE)
    cells.append(md("### ↓↓↓ Run everything below AFTER the restart ↓↓↓"))
    cells += [GPU_DRIVE_TEST, SETUP_CLONE, SETUP_PATHS]
    cells.append(md("## Post-restart sanity (clean imports + gptqmodel + tf version)"))
    cells.append(VERIFY)
    cells.append(md("## SRC-COMPAT smoke test (stop early if src breaks)"))
    cells.append(SMOKE)
    cells.append(md("## Seed test folder (no-clobber) + run the GPTQ ring"))
    cells.append(SEED_FROM_MAIN)
    cells.append(RING)
    cells.append(md("## E14 cross-method table (instruct → bnb4 / awq / gptq)"))
    cells.append(E14_TABLE)
    return notebook(cells, gpu=True)


def main():
    name = "13_e14_gptq_colab.ipynb"
    nb = nb_13_gptq()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-12 untouched. GPU: A100. GPTQ via gptqmodel/Triton; restart-in-the-middle.")


if __name__ == "__main__":
    main()
