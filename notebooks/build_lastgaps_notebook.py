"""
Generate notebook 11 — the remaining gaps after the 09/10 round, in one file.

Closes (from the latest result scan of rhprofile_results_other):
  • gemma2_9b_it_bnb4 ring  -> profile + behaviour + utility  (gemma still 1 ring;
    bnb4 has no kernel issue, only needs A100 for the 256-dim KV)
  • phi35_mini             -> profile + utility (both still missing — phi PROFILE
    failed in 09) + long-context behaviour (still NaN at >=8k). Wrapped so any
    failure prints a full traceback (so we finally SEE why phi profile breaks).
  • qwen AWQ + GPTQ rings  -> best-effort: --no-deps kernels, verify importable,
    skip cleanly if the prebuilt kernels won't load on this stack (E14 then lacks
    that arm; the quant story still stands on llama+gemma bnb4).
  • re-run prediction + inheritance + a coverage/E14/phi verification read-out.

ONLY writes 11_remaining_gaps_colab.ipynb — never touches 00-10. Reuses the
canonical setup + the SAME test-folder logic (writes rhprofile_results_other,
seeded no-clobber from main; main is never written).

Run:  python notebooks/build_lastgaps_notebook.py

GPU: A100 40 GB (80 GB ideal) — required for gemma2_9b_it_bnb4 (256-dim KV is fp16
even at 4-bit weights). phi + qwen rings also fit; one A100 does everything.
Run all (after tokens in Cell 2 + Drive popup). Resume-safe (skip-if-exists).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import md, code, notebook, SETUP_PIP, SETUP_CLONE, SETUP_PATHS
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN, AWQ_GPTQ_INSTALL

# ---------------------------------------------------------------------------
# gemma2_9b_it_bnb4 ring (A100; bnb4 -> base stack, sdpa to avoid eager OOM)
# ---------------------------------------------------------------------------
GEMMA_BNB4 = code("""
# gemma2_9b_it_bnb4 ring: profile + behaviour + utility. bitsandbytes 4-bit (no
# special kernel), but 256-dim KV is fp16 -> needs A100. sdpa avoids eager OOM.
import json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'gemma2_9b_it_bnb4'
RD = Path(RESULTS_DIR)
prof = RD/'profile'/f'{key}_seed{SEED}.json'
beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
util = RD/'utility'/f'{key}_seed{SEED}.json'
cfg = dict(model_cfg(config, key)); cfg['attn_implementation'] = 'sdpa'
try:
    if not prof.exists():
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print('gemma bnb4 profile saved')
    if not beh.exists():
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh); print('gemma bnb4 behaviour saved:', r['behavior'].get('niah_per_context'))
    if not util.exists() and prof.exists():
        d = json.load(open(prof, encoding='utf-8'))
        save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                        argmax_scores=d['argmax_scores'], seed=SEED), util)
        print('gemma bnb4 utility saved')
    print('gemma bnb4 ring done.')
except Exception as e:
    import traceback; traceback.print_exc(); print('gemma bnb4 FAILED ->', e)
""")

# ---------------------------------------------------------------------------
# phi35_mini: profile + utility + long-context behaviour (verbose on failure)
# ---------------------------------------------------------------------------
PHI_ALL = code("""
# phi35_mini: profile (failed in 09 — print full traceback to see why) + utility
# (needs the profile) + long-context behaviour (was NaN >=8k). Each step is
# wrapped so one failure does not block the others.
# IMPORTANT: Phi3's remote code does NOT support sdpa -> use EAGER (default). On
# A100 eager reaches 8k/16k; 32k may OOM->NaN (handled), which still gives a real
# niah_long. (Install flash_attn if you want phi at full 32k.)
import json, math
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'phi35_mini'
RD = Path(RESULTS_DIR)
prof = RD/'profile'/f'{key}_seed{SEED}.json'
beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
util = RD/'utility'/f'{key}_seed{SEED}.json'
cfg = dict(model_cfg(config, key))   # eager default; do NOT force sdpa for Phi3

# (1) profile
try:
    if not prof.exists():
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print('phi profile saved')
    else:
        print('phi profile exists -> skip')
except Exception as e:
    import traceback; traceback.print_exc(); print('phi PROFILE FAILED ->', e)

# (2) utility (needs the profile)
try:
    if prof.exists() and not util.exists():
        d = json.load(open(prof, encoding='utf-8'))
        save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                        argmax_scores=d['argmax_scores'], seed=SEED), util)
        print('phi utility saved')
except Exception as e:
    import traceback; traceback.print_exc(); print('phi UTILITY FAILED ->', e)

# (3) long-context behaviour (overwrite the NaN result)
def has_real_long(p):
    if not p.exists():
        return False
    pc = json.load(open(p, encoding='utf-8'))['behavior'].get('niah_per_context', {})
    return any(int(c) >= 16384 and v == v for c, v in pc.items())   # v==v is False for NaN
try:
    if not has_real_long(beh):
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh)
        print('phi behaviour per_ctx =', r['behavior'].get('niah_per_context'),
              '| niah_long =', r['behavior'].get('niah_long'))
    else:
        print('phi real long-context already present -> skip')
except Exception as e:
    import traceback; traceback.print_exc(); print('phi BEHAVIOUR FAILED ->', e)
""")

# ---------------------------------------------------------------------------
# qwen AWQ/GPTQ rings — best-effort (verify kernels first; skip if not loadable)
# ---------------------------------------------------------------------------
QWEN_RINGS = code("""
# qwen AWQ + GPTQ rings (E14). Best-effort: only attempt a ring whose kernel is
# importable (the --no-deps install above). If a kernel is missing/unloadable on
# this stack, that arm is skipped (E14 lacks it; quant story still holds on
# llama+gemma bnb4). Resume-safe + 23 h guard.
import time, json, importlib.util
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json, time_guard)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; RD = Path(RESULTS_DIR)

have_awq  = importlib.util.find_spec('awq') is not None
have_gptq = importlib.util.find_spec('gptqmodel') is not None
print('kernels -> autoawq:', have_awq, '| gptqmodel:', have_gptq)
RINGS = [('qwen25_7b_instruct_awq4', have_awq), ('qwen25_7b_instruct_gptq4', have_gptq)]

start = time.time(); times = []
for key, have in RINGS:
    if not have:
        print(key, '-> kernel NOT importable, SKIP (E14 will lack this arm)'); continue
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
        print(key, 'FAILED ->', e, '(kernel/load issue; E14 lacks this arm)')
print('qwen rings pass complete.')
""")

# ---------------------------------------------------------------------------
# analysis refresh + verification
# ---------------------------------------------------------------------------
VERIFY = code("""
# Re-run prediction + inheritance on the test folder, then print coverage / E14 /
# phi / gemma-ring read-out. CPU work; fine on the GPU runtime.
import subprocess, sys, json
from pathlib import Path
P2 = '/content/rope-part2'
def run(args):
    cmd = [sys.executable] + args + ['--results-dir', RESULTS_DIR, '--config', CONFIG,
                                     '--part1-repo', '/content/rope-part1']
    print('>>', ' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2500:]); print(r.stderr[-1000:] if r.returncode else '')
run([f'{P2}/scripts/run_prediction.py', '--seed', '42', '--retest-seed', '123'])
run([f'{P2}/scripts/run_inheritance.py', '--lineage', 'all', '--seed', '42'])

RD = Path(RESULTS_DIR)
def mk(sub, key): return 'Y' if (RD/sub/f'{key}_seed42.json').exists() else '.'
print('\\n== COVERAGE (test folder) ==')
for key in ['gemma2_9b_it_bnb4', 'phi35_mini', 'qwen25_7b_instruct_awq4', 'qwen25_7b_instruct_gptq4']:
    print(f'  {key:28s} prof={mk(\"profile\",key)} beh={mk(\"behavior\",key)} util={mk(\"utility\",key)}')
q = json.load(open(RD/'inheritance'/'qwen.json', encoding='utf-8')); e = q.get('E14_quant_ablation', {})
print('E14 quant: awq4=%s gptq4=%s' % ('set' if e.get('awq4') else 'NULL', 'set' if e.get('gptq4') else 'NULL'))
g = json.load(open(RD/'inheritance'/'gemma.json', encoding='utf-8'))
print('gemma rings:', [(r.get('parent'), r.get('child')) for r in g.get('rings', [])])
pf = RD/'behavior'/'phi35_mini_seed42.json'
if pf.exists():
    b = json.load(open(pf, encoding='utf-8'))['behavior']
    print('phi niah_long:', b.get('niah_long'), '| per_ctx:', b.get('niah_per_context'))
print('phi profile present:', mk('profile', 'phi35_mini'), '| phi utility present:', mk('utility', 'phi35_mini'))
""")


def nb_11_remaining_gaps():
    cells = [md(
        "# 11 · Remaining gaps (A100) — gemma 4-bit ring + phi (profile/util/long) "
        "+ qwen AWQ/GPTQ (best-effort)\n"
        "**GPU: A100 40 GB (80 GB ideal)** — required for `gemma2_9b_it_bnb4` "
        "(256-dim KV is fp16 even at 4-bit). phi + qwen rings also fit; one A100 "
        "does all.\n\n"
        "Writes to the SEPARATE `rhprofile_results_other` test folder (seeded "
        "no-clobber from main; main untouched). `Run all` after tokens in Cell 2 "
        "(**HF token required — gemma is gated**) + the Drive popup. Resume-safe.\n\n"
        "| Task | Why | GPU |\n|---|---|---|\n"
        "| gemma2_9b_it_bnb4 ring | gemma still 1 ring (no 4-bit) | **A100** |\n"
        "| phi profile + utility | both missing (phi profile failed in 09) | A100 |\n"
        "| phi long-context behaviour | NaN >=8k; eager on A100 (Phi3 has no sdpa) | A100 |\n"
        "| qwen AWQ/GPTQ rings | **E14 (null)** — best-effort, kernel-dependent | A100 |\n\n"
        "If the qwen rings skip (kernels won't load on this stack), that is OK: the "
        "quant-inheritance story stands on llama + gemma bnb4; E14 (AWQ-vs-GPTQ) is "
        "the only kernel-dependent extra.")]
    cells += [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + paste tokens in Cell 2."),
        GPU_DRIVE_TEST, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## 1 — AWQ/GPTQ kernels (--no-deps; only the qwen rings need these)"))
    cells.append(AWQ_GPTQ_INSTALL)
    cells.append(md("## 2 — Seed the test folder from main (no-clobber; main read-only)"))
    cells.append(SEED_FROM_MAIN)
    cells.append(md("## 3 — gemma2_9b_it_bnb4 ring (reliable; A100)"))
    cells.append(GEMMA_BNB4)
    cells.append(md("## 4 — phi35_mini: profile + utility + long-context behaviour (verbose on failure)"))
    cells.append(PHI_ALL)
    cells.append(md("## 5 — qwen AWQ/GPTQ rings (best-effort; skips if kernels won't load)"))
    cells.append(QWEN_RINGS)
    cells.append(md("## 6 — Re-run analysis + verify coverage / E14 / phi"))
    cells.append(VERIFY)
    return notebook(cells, gpu=True)


def main():
    name = "11_remaining_gaps_colab.ipynb"
    nb = nb_11_remaining_gaps()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-10 untouched. GPU: A100 40GB (80GB ideal). Writes to rhprofile_results_other.")


if __name__ == "__main__":
    main()
