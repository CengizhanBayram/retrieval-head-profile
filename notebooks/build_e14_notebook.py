"""
Generate notebook 12 — the E14 cross-method quant comparison (AWQ + GPTQ) on a
PINNED COMPATIBLE stack, kept separate from the 4.47 base pipeline.

Why separate: AWQ/GPTQ kernels do NOT load on the base stack (torch 2.11/cu128 +
transformers 4.47): autoawq needs tf>=4.51 (imports qwen3) and prebuilt kernels
that don't exist for torch 2.11; transformers-4.47 GPTQ needs optimum+auto-gptq
which won't build. This notebook pins transformers==4.51.3 and uses gptqmodel
(Triton kernels -> work on new torch) for GPTQ and autoawq best-effort for AWQ.

These are DIFFERENT quant methods from bnb4 (activation-aware / Hessian-error-min
vs uniform NF4) and may preserve retrieval/frequency channels differently — that
is exactly the E14 question, so bnb4 does NOT substitute for them.

EXPLORATORY: transformers 4.51 may break the inherited src/ (written for 4.47), and
AWQ kernels may still not match torch 2.11. So Cell A is a SRC-COMPAT SMOKE TEST
that stops early if src breaks under 4.51 — no hours wasted. If only GPTQ loads,
E14 still compares bnb4 vs GPTQ (a real cross-method result). If neither loads,
fall back to the bnb4-only quant story from notebook 11.

ONLY writes 12_e14_awq_gptq_colab.ipynb. Writes to rhprofile_results_other.

Run:  python notebooks/build_e14_notebook.py
GPU: A100 (qwen 7B 4-bit). Run all after tokens in Cell 2 + Drive popup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import md, code, notebook, SETUP_CLONE, SETUP_PATHS
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN

# Pinned compatible deps (transformers 4.51 + Triton GPTQ + best-effort AWQ).
PINNED_PIP = code("""
%%bash
# PINNED stack for AWQ/GPTQ (different from the 4.47 base). transformers 4.51.3 is
# the last autoawq-tested line and has qwen3; gptqmodel brings Triton kernels that
# compile at runtime (work on torch 2.11). This intentionally upgrades transformers
# for THIS notebook only.
echo '== pinned install (transformers 4.51.3 + gptqmodel + autoawq) =='
pip install -q transformers==4.51.3 accelerate datasets bitsandbytes 2>&1 | tail -1
pip install -q scipy scikit-learn pandas pyyaml tqdm huggingface_hub 2>&1 | tail -1
pip install -q gptqmodel 2>&1 | tail -1 || echo 'gptqmodel install failed'
pip install -q autoawq optimum 2>&1 | tail -1 || echo 'autoawq/optimum install failed'
python - <<'PYEOF'
import importlib
for m in ['transformers','numpy','scipy.stats','gptqmodel','awq','optimum']:
    try:
        importlib.import_module(m); print('OK  ', m)
    except Exception as e:
        print('MISS', m, '->', str(e)[:80])
import transformers; print('transformers', transformers.__version__)
PYEOF
""")

# Cell A — SRC-COMPAT SMOKE TEST: stop early if src breaks under transformers 4.51.
SMOKE = code("""
# SRC-COMPAT SMOKE TEST under transformers 4.51. If this crashes, the inherited
# src/ is incompatible with 4.51 -> STOP and keep the bnb4-only result from nb 11.
# (We load a real panel model and run the detector on a few samples.)
import transformers, gc, torch
print('transformers', transformers.__version__)
from rhp.panel import load_panel, model_cfg
from rhp.loader import load_model_any
from src.retrieval_head_detector import RetrievalHeadDetector
config = load_panel(CONFIG)
ok_src = False
try:
    m, tok = load_model_any(model_cfg(config, 'qwen25_7b_instruct'), 'qwen25_7b_instruct')
    det = RetrievalHeadDetector(m, tok, config, score_threshold=0.1, seed=42)
    s = det.generate_niah_samples(5, [2048], [0.5])
    sc = det.score_heads(s)
    print('SMOKE OK -> src works under tf', transformers.__version__, '| score matrix', sc.shape)
    ok_src = True
    del m, tok, det
except Exception as e:
    import traceback; traceback.print_exc()
    print('SMOKE FAILED -> src is NOT compatible with tf 4.51. STOP; keep bnb4-only (nb 11).')
finally:
    gc.collect(); torch.cuda.empty_cache()
""")

# Cell B — run the AWQ + GPTQ rings (only if smoke passed). GPTQ first (best chance).
RINGS = code("""
# AWQ + GPTQ rings on the pinned stack. GPTQ (gptqmodel/Triton) first — most
# likely to load on torch 2.11; AWQ best-effort. Each failure isolated + verbose.
import importlib.util, time, json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json, time_guard)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; RD = Path(RESULTS_DIR)

if not ok_src:
    print('Smoke test did not pass -> skipping rings.')
else:
    RINGS = ['qwen25_7b_instruct_gptq4', 'qwen25_7b_instruct_awq4']   # GPTQ first
    start = time.time(); times = []
    for key in RINGS:
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
            print(key, 'FAILED ->', e, '(kernel/load issue on this stack)')
    print('rings pass complete.')
""")

# Cell C — E14 cross-method table: instruct vs each quant method (bnb4/awq/gptq).
E14 = code("""
# E14 cross-method comparison: how does each 4-bit METHOD preserve the retrieval
# circuit, relative to the same fp16 instruct parent? Copy-Jaccard (identity) +
# delta freq_com (frequency). bnb4 from notebook 11; awq/gptq from this notebook.
import json
from pathlib import Path
from rhp.inheritance import compare_ring
RD = Path(RESULTS_DIR); SEED = 42

def load(key):
    pf = RD/'profile'/f'{key}_seed{SEED}.json'
    if not pf.exists(): return None
    d = json.load(open(pf, encoding='utf-8'))
    bf = RD/'behavior'/f'{key}_seed{SEED}.json'
    if bf.exists(): d['behavior'] = json.load(open(bf, encoding='utf-8')).get('behavior', {})
    uf = RD/'utility'/f'{key}_seed{SEED}.json'
    if uf.exists(): d['utility'] = json.load(open(uf, encoding='utf-8')).get('utility', {})
    return d

ref = load('qwen25_7b_instruct')
print('E14 — qwen instruct -> 4-bit, by METHOD:')
print(f\"  {'method':8s} {'copyJaccard':>12s} {'dFreqCom':>10s} {'dNIAHlong':>10s}\")
for method, key in [('bnb4','qwen25_7b_instruct_bnb4'),
                    ('awq4','qwen25_7b_instruct_awq4'),
                    ('gptq4','qwen25_7b_instruct_gptq4')]:
    child = load(key)
    if ref is None or child is None:
        print(f'  {method:8s}  (missing — not produced)'); continue
    r = compare_ring(ref, child, lineage='qwen')
    jac = r['E10_identity']['copy']['jaccard']
    dfc = r['E12_frequency']['delta_freq_com']
    dnl = r.get('E13_bridge', {}).get('delta_niah', float('nan'))
    print(f'  {method:8s} {jac:12.3f} {dfc:10.3f} {dnl:10.3f}')
print('\\nIf 2+ methods are present, you have a real cross-method E14 result '
      '(do the methods preserve the circuit differently?).')
""")


def nb_12_e14():
    cells = [md(
        "# 12 · E14 cross-method quant (AWQ + GPTQ) — PINNED stack (transformers 4.51)\n"
        "**GPU: A100.** EXPLORATORY + SEPARATE from the 4.47 pipeline. AWQ/GPTQ are "
        "DIFFERENT quant methods from bnb4 (activation-aware / Hessian vs uniform "
        "NF4) and may preserve the retrieval circuit differently — the E14 question. "
        "bnb4 does **not** substitute for them.\n\n"
        "AWQ/GPTQ don't load on the base 4.47 stack. Here we pin **transformers "
        "4.51.3** + **gptqmodel** (Triton kernels, work on torch 2.11) for GPTQ and "
        "autoawq best-effort for AWQ.\n\n"
        "**Cell A is a src-compat SMOKE TEST** — transformers 4.51 may break the "
        "inherited `src/` (written for 4.47). If it fails, STOP and keep the "
        "bnb4-only quant story from notebook 11. If only GPTQ loads, E14 still "
        "compares **bnb4 vs GPTQ** (a real cross-method result).\n\n"
        "Writes to `rhprofile_results_other`. `Run all` after tokens in Cell 2 + the "
        "Drive popup. Needs qwen bnb4 from notebook 11 for the full 3-method table.")]
    cells += [
        md("### Setup — pinned install, then clone + paths. Paste tokens in Cell 2."),
        GPU_DRIVE_TEST, PINNED_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## A — SRC-COMPAT smoke test (STOP early if src breaks under tf 4.51)"))
    cells.append(SMOKE)
    cells.append(md("## B — Seed test folder (no-clobber) + run AWQ/GPTQ rings"))
    cells.append(SEED_FROM_MAIN)
    cells.append(RINGS)
    cells.append(md("## C — E14 cross-method table (instruct → bnb4 / awq / gptq)"))
    cells.append(E14)
    return notebook(cells, gpu=True)


def main():
    name = "12_e14_awq_gptq_colab.ipynb"
    nb = nb_12_e14()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-11 untouched. GPU: A100. Pinned transformers 4.51 (this notebook only).")


if __name__ == "__main__":
    main()
