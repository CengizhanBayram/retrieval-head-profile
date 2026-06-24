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
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN

# ---------------------------------------------------------------------------
# gemma2_9b_it_bnb4 ring (A100; bnb4 -> base stack, sdpa to avoid eager OOM)
# ---------------------------------------------------------------------------
GEMMA_BNB4 = code("""
# gemma2_9b_it_bnb4 ring: profile + behaviour + utility. bitsandbytes 4-bit.
# IMPORTANT: force EAGER. The detector needs output_attentions=True, which under
# sdpa makes gemma2 route to flex_attention -> 3D attn tensor -> detector hook
# crashes (the IndexError seen in 09). Eager also matches gemma's softcapping and
# the non-bnb4 gemma runs. 256-dim KV is fp16 -> use A100 80GB; 32k may OOM->NaN.
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
cfg = dict(model_cfg(config, key)); cfg['attn_implementation'] = 'eager'
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
# phi35_mini: profile + utility + long-context behaviour. Each step wrapped so one
# failure does not block the others.
# IMPORTANT (two phi quirks found in 09):
#  1. Phi3 uses a FUSED qkv_proj (no q_proj) -> the frequency-signature patching
#     crashes. So run the profile with do_freq=False (phi gets head set + knockout
#     + utility, but no frequency signature; phi is low-theta so this is minor).
#  2. Phi3 remote code does NOT support sdpa -> EAGER (default). On A100 eager
#     reaches 8k/16k; 32k may OOM->NaN (handled), still a real niah_long.
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
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096,
                                        do_freq=False), prof)   # Phi3 fused qkv -> skip freq
        print('phi profile saved (no freq signature: Phi3 fused qkv_proj)')
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
# qwen bnb4 ring — the 3rd within-method (bnb NF4) quant ring (works on this
# stack; AWQ/GPTQ are pursued separately in notebook 12 for the E14 cross-method).
# ---------------------------------------------------------------------------
QWEN_BNB4 = code("""
# qwen25_7b_instruct_bnb4 ring: profile + behaviour + utility. bitsandbytes 4-bit,
# same method as llama/gemma bnb4 -> a clean 3-family within-method quant-
# inheritance result. qwen2 + output_attentions falls back gracefully (no crash),
# so no forced attn needed; qwen is long-context native so behaviour reaches 32k.
import json
from pathlib import Path
from scripts._common import (run_profile_for_model, run_behavior_for_model,
                             run_utility_for_model, save_json)
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'qwen25_7b_instruct_bnb4'
RD = Path(RESULTS_DIR)
prof = RD/'profile'/f'{key}_seed{SEED}.json'
beh  = RD/'behavior'/f'{key}_seed{SEED}.json'
util = RD/'utility'/f'{key}_seed{SEED}.json'
cfg = dict(model_cfg(config, key))
try:
    if not prof.exists():
        save_json(run_profile_for_model(key, cfg, config, seed=SEED, context_length=4096), prof)
        print('qwen bnb4 profile saved')
    if not beh.exists():
        r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
        save_json(r, beh); print('qwen bnb4 behaviour saved:', r['behavior'].get('niah_per_context'))
    if not util.exists() and prof.exists():
        d = json.load(open(prof, encoding='utf-8'))
        save_json(run_utility_for_model(key, cfg, config, argmax_heads=d['argmax_heads'],
                                        argmax_scores=d['argmax_scores'], seed=SEED), util)
        print('qwen bnb4 utility saved')
    print('qwen bnb4 ring done.')
except Exception as e:
    import traceback; traceback.print_exc(); print('qwen bnb4 FAILED ->', e)
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
for key in ['gemma2_9b_it_bnb4', 'phi35_mini', 'qwen25_7b_instruct_bnb4']:
    print(f'  {key:28s} prof={mk(\"profile\",key)} beh={mk(\"behavior\",key)} util={mk(\"utility\",key)}')
print('\\n== 4-bit (bnb NF4) quant rings across families ==')
for lin in ['llama', 'gemma', 'qwen']:
    j = json.load(open(RD/'inheritance'/f'{lin}.json', encoding='utf-8'))
    for r in j.get('rings', []):
        c = r.get('child', '')
        if 'bnb4' in c:
            print(f'  {lin:6s} {r.get(\"parent\")} -> {c}: copyJac=%.3f dFreqCom=%.3f'
                  % (r['E10_identity']['copy']['jaccard'], r['E12_frequency']['delta_freq_com']))
pf = RD/'behavior'/'phi35_mini_seed42.json'
if pf.exists():
    b = json.load(open(pf, encoding='utf-8'))['behavior']
    print('phi niah_long:', b.get('niah_long'), '| per_ctx:', b.get('niah_per_context'))
print('phi profile present:', mk('profile', 'phi35_mini'), '| phi utility present:', mk('utility', 'phi35_mini'))
""")


def nb_11_remaining_gaps():
    cells = [md(
        "# 11 · Remaining gaps (A100 80GB) — gemma 4-bit ring + phi + qwen 4-bit ring\n"
        "**GPU: A100 80 GB** — `gemma2_9b_it_bnb4` (256-dim KV is fp16 even at 4-bit) "
        "and phi at long context need the headroom; one A100 80GB does all three.\n\n"
        "All three failures from the 09 run are fixed here (root causes were CODE, "
        "not VRAM):\n"
        "- **gemma bnb4** used sdpa → detector's `output_attentions` routed gemma2 to "
        "flex_attention → 3D tensor crash. **Fixed: force EAGER.**\n"
        "- **phi profile** crashed in the frequency step (Phi3 fused `qkv_proj`, no "
        "`q_proj`). **Fixed: `do_freq=False`** (phi keeps head set + knockout + "
        "utility; no freq signature — phi is low-θ anyway).\n"
        "- **qwen AWQ/GPTQ** kernels are incompatible with the pinned torch 2.11 / "
        "transformers 4.47 stack. **Replaced with qwen bnb4** (the 3rd within-method "
        "ring); AWQ/GPTQ (E14 cross-method) move to notebook 12 on a pinned stack.\n\n"
        "Writes to the SEPARATE `rhprofile_results_other` test folder (seeded "
        "no-clobber from main; main untouched). `Run all` after tokens in Cell 2 "
        "(**HF token required — gemma is gated**) + the Drive popup. Resume-safe.\n\n"
        "| Task | Fix | GPU |\n|---|---|---|\n"
        "| gemma2_9b_it_bnb4 ring | eager (detector) | **A100 80GB** |\n"
        "| phi profile + utility | do_freq=False | A100 |\n"
        "| phi long-context behaviour | eager (Phi3 has no sdpa) | A100 |\n"
        "| qwen25_7b_instruct_bnb4 ring | 3rd bnb4 method | A100 |\n\n"
        "Result: a clean **3-family (llama + gemma + qwen) bnb4 quant-inheritance** "
        "result + phi as the 8th family. E14 (AWQ-vs-GPTQ cross-method) → notebook 12.")]
    cells += [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + paste tokens in Cell 2."),
        GPU_DRIVE_TEST, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## 1 — Seed the test folder from main (no-clobber; main read-only)"))
    cells.append(SEED_FROM_MAIN)
    cells.append(md("## 2 — gemma2_9b_it_bnb4 ring (EAGER; A100 80GB)"))
    cells.append(GEMMA_BNB4)
    cells.append(md("## 3 — phi35_mini: profile (no-freq) + utility + long-context behaviour"))
    cells.append(PHI_ALL)
    cells.append(md("## 4 — qwen25_7b_instruct_bnb4 ring (3rd within-method quant ring)"))
    cells.append(QWEN_BNB4)
    cells.append(md("## 5 — Re-run analysis + verify coverage / bnb4 rings / phi"))
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
