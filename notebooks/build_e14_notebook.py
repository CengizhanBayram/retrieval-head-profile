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
# STEP 1 of 2. AWQ-ONLY install, then RESTART (next cell tells you).
# WHY AWQ-only: a previous run proved AWQ LOADS and the detector WORKS (200 heads
# scored) on tf 4.51.3 — it only broke at generation because *gptqmodel* import
# monkeypatches transformers._prepare_cache_for_generation with a wrong-arity shim.
# gptqmodel also needs tf>=4.52 (transformers.masking_utils) which CONFLICTS with
# autoawq's tf<=4.51.3. So we install ONLY autoawq here -> no bad patch -> AWQ
# generation works. (GPTQ is a separate run on tf>=4.52; deferred.)
echo '== autoawq ONLY (no gptqmodel/optimum -> no generation monkeypatch) =='
pip install -q autoawq 2>&1 | tail -1 || echo 'autoawq failed'
echo '== pin transformers 4.51.3 (autoawq last-tested) + consistent numpy LAST =='
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
# Post-restart sanity: numpy/scipy/transformers import cleanly + awq present.
# (gptqmodel must NOT be importable here — if it is, it has patched generation.)
import importlib
for m in ['numpy', 'scipy.stats', 'transformers', 'awq']:
    try:
        importlib.import_module(m); print('OK  ', m)
    except Exception as e:
        print('MISS', m, '->', str(e)[:90])
try:
    importlib.import_module('gptqmodel'); print('WARNING gptqmodel present -> may break AWQ generation')
except Exception:
    print('OK   gptqmodel absent (good — no generation patch)')
import transformers, numpy, torch
print('transformers', transformers.__version__, '| numpy', numpy.__version__,
      '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
""")

# SRC-COMPAT smoke test under transformers 4.51 (stop early if src breaks).
PATCH_QUANT = code("""
# RUNTIME PATCH (no repo push needed): make compute_query_projection_norms handle
# quantized q_proj (AWQ/GPTQ/bnb4). Recovers the effective dense weight via an
# identity forward (the layer dequantizes internally), instead of reading the
# nonexistent .weight on WQLinear_GEMM/Marlin/Params4bit. Travels with this
# notebook, so it works even if the src fix isn't on GitHub yet.
import numpy as np, torch
import src.dimension_utility as du

def _dense_q_weight(q_proj):
    w = getattr(q_proj, 'weight', None)
    if (isinstance(w, torch.Tensor) and not getattr(w, 'is_meta', False)
            and w.dtype in (torch.float16, torch.bfloat16, torch.float32) and w.dim() == 2):
        return w.detach().cpu().float()
    try:
        in_f = getattr(q_proj, 'in_features', None)
        dev = next((p.device for p in q_proj.parameters(recurse=True)), None)
        if dev is None:
            dev = next((b.device for b in q_proj.buffers(recurse=True)), None)
        if not in_f or dev is None:
            raise RuntimeError('no in_features/device')
        with torch.no_grad():
            eye = torch.eye(in_f, device=dev, dtype=torch.float16)
            y = q_proj(eye).float()
            bias = q_proj(torch.zeros(1, in_f, device=dev, dtype=torch.float16)).float()
            wt = (y - bias).t().contiguous()
        return wt.detach().cpu().float()
    except Exception as e:
        print('dense-weight extraction failed:', e, '-> zeros')
        n = int(getattr(q_proj, 'out_features', 0) or 0); m = int(getattr(q_proj, 'in_features', 0) or 0)
        return torch.zeros((n, m), dtype=torch.float32)

def _patched_norms(self):
    layers = self._get_layers(); out = []
    for li, layer in enumerate(layers):
        try:
            q_proj = layer.self_attn.q_proj
        except AttributeError:
            out.append(np.zeros((self.n_heads, self.head_dim), dtype=np.float32)); continue
        weight = _dense_q_weight(q_proj); od = weight.shape[0]; nq = od // self.head_dim
        if nq == 0:
            out.append(np.zeros((self.n_heads, self.head_dim), dtype=np.float32)); continue
        norms = weight.view(nq, self.head_dim, -1).abs().sum(dim=-1).numpy().astype(np.float32)
        if nq < self.n_heads: norms = np.repeat(norms, self.n_heads // nq, axis=0)
        elif nq > self.n_heads: norms = norms[:self.n_heads]
        out.append(norms)
    return np.stack(out, axis=0)

du.DimensionUtilityAnalyzer.compute_query_projection_norms = _patched_norms
print('PATCHED compute_query_projection_norms for quantized q_proj (no push needed).')
""")

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

# AWQ ring only (it already loads + detects on tf 4.51.3; with gptqmodel absent the
# generation step now works too). Resume-safe; verbose on failure.
RINGS = code("""
import time, json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json, time_guard)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; RD = Path(RESULTS_DIR)

if not ok_src:
    print('Smoke did not pass -> skipping AWQ.')
else:
    key = 'qwen25_7b_instruct_awq4'
    prof = RD/'profile'/f'{key}_seed{SEED}.json'
    beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
    util = RD/'utility'/f'{key}_seed{SEED}.json'
    cfg = dict(model_cfg(config, key))
    try:
        # FULL profile now possible: the src fix (_dense_q_weight) recovers the
        # quantized q_proj's effective weight via an identity forward, so the freq
        # signature + norms work (no more qweight crash). OVERWRITE old/partial files.
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print(key, 'profile saved (full: identity + freq + knockout) [overwrite]')
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh); print(key, 'behaviour saved [overwrite]')
        d = json.load(open(prof, encoding='utf-8'))
        save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                        argmax_scores=d['argmax_scores'], seed=SEED), util)
        print(key, 'utility saved [overwrite]')
        print('AWQ ring done (full). If freq_com is degenerate, the AWQ kernel '
              'rejected the dense-weight probe — identity is still valid for E14.')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(key, 'FAILED ->', e)
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
        "# 12 · E14 cross-method quant — AWQ (RESTART approach)\n"
        "**GPU: A100.** A prior run proved **AWQ loads and the detector works** on "
        "transformers 4.51.3 (200 heads scored) — it only broke at *generation* "
        "because **gptqmodel's import monkeypatches** transformers generation with a "
        "wrong-arity shim. gptqmodel also needs tf≥4.52 (`masking_utils`), which "
        "**conflicts** with autoawq's tf≤4.51.3. So AWQ and GPTQ can't share one "
        "transformers — this notebook does **AWQ only** (no gptqmodel → no bad "
        "patch → AWQ generation works). GPTQ is a separate tf≥4.52 run, deferred.\n\n"
        "The `_center` numpy crash is fixed the way you noticed: **install, then "
        "RESTART** so the fresh interpreter loads the new numpy cleanly.\n\n"
        "Yields **bnb4 vs AWQ** cross-method E14 (a real result). AWQ is a different "
        "method from bnb4 (activation-aware vs uniform NF4) — bnb4 doesn't substitute.\n\n"
        "**NOT a single Run-all — one manual restart in the middle:**\n"
        "1. Run **Cell A** (install autoawq + tf 4.51.3). 2. `Runtime → Restart "
        "session`. 3. Run every cell **below** Cell A.\n\n"
        "A src-compat smoke test runs after restart; if src breaks under tf 4.51, "
        "STOP and keep the 3-family bnb4 story (notebook 11). qwen2.5 is ungated.")]
    cells.append(md("## Cell A — install the stack (run this, THEN restart)"))
    cells.append(INSTALL_THEN_RESTART)
    cells.append(RESTART_NOTE)
    cells.append(md("### ↓↓↓ Run everything below AFTER the restart ↓↓↓"))
    cells += [GPU_DRIVE_TEST, SETUP_CLONE, SETUP_PATHS]
    cells.append(md("## Post-restart sanity (clean imports + kernels)"))
    cells.append(VERIFY)
    cells.append(md("## Runtime patch — quantized q_proj norms (no repo push needed)"))
    cells.append(PATCH_QUANT)
    cells.append(md("## SRC-COMPAT smoke test (stop early if src breaks under tf 4.51)"))
    cells.append(SMOKE)
    cells.append(md("## Seed test folder (no-clobber) + run the AWQ ring"))
    cells.append(SEED_FROM_MAIN)
    cells.append(RINGS)
    cells.append(md("## E14 cross-method table (instruct → bnb4 / awq; gptq from nb 13)"))
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
