"""
Generate the SEPARATE follow-up notebook (10) for the two data-quality items
flagged in the result read-out:

  10a  phi35_mini long-context behaviour  — phi was only run at 4096 (niah_long
       was NaN); phi-3.5-mini is 128k-native, so re-run the full long-context
       schedule so phi can enter the RQ2 long-context target.
  10b  mistral knockout sweep             — mistral's knockout_drop is ~0 (base
       0.013, instruct 0.128). Sweep the number of top heads ablated (k) to tell
       "top-30 too few" (drop grows with k) from "genuinely redundant/distributed"
       (drop stays ~0 even ablating all detected heads). A clean, bounded answer.

This generator ONLY writes ``10_followups_colab.ipynb`` — it never touches 00-09.
It reuses the canonical setup + the SAME test-folder logic as notebook 09 (writes
to rhprofile_results_other, seeded no-clobber from main; main is never written).

Run:  python notebooks/build_followup_notebook.py

GPU: L4 24 GB (phi 3.8 B + mistral 7 B both fit; no A100 needed). Short: ~5 h.
Independent of 09 — run it before or after, any order; both write the test folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from build_notebooks import (  # canonical helpers + setup cells
    md, code, notebook, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
)
from build_gap_notebook import GPU_DRIVE_TEST, SEED_FROM_MAIN  # same test-folder logic

# ---------------------------------------------------------------------------
# 10a — phi long-context behaviour (overwrite the short-context phi result)
# ---------------------------------------------------------------------------
PHI_LONGCTX = code("""
# phi35_mini long-context behaviour. The on-disk phi result is NaN at >=8k.
# Phi3's remote code only supports EAGER attention (not sdpa), and eager's O(n^2)
# VRAM OOMs at >=8k on L4 -> that is the NaN. So on L4 this still NaNs at >=8k
# (expected); the REAL phi long-context is produced in notebook 11 on an A100.
# This cell is kept harmless/idempotent. OVERWRITES phi in the TEST folder only.
import json, math
from pathlib import Path
from scripts._common import run_behavior_for_model, save_json
from rhp.panel import load_panel, model_cfg
config = load_panel(CONFIG); SEED = 42; key = 'phi35_mini'
out = Path(RESULTS_DIR)/'behavior'/f'{key}_seed{SEED}.json'

def has_real_long(p):
    # True only if a context >=16384 has a REAL (non-NaN) recall — not just the
    # key present. The old NaN result has the keys but NaN values, so this re-runs.
    if not p.exists():
        return False
    pc = json.load(open(p, encoding='utf-8')).get('behavior', {}).get('niah_per_context', {})
    return any(int(c) >= 16384 and v == v for c, v in pc.items())   # v==v is False for NaN

if has_real_long(out):
    print('phi real long-context behaviour already present -> skip')
else:
    cfg = dict(model_cfg(config, key))     # eager default; Phi3 remote code does NOT support sdpa
    # NOTE: phi long-context needs A100 — eager attention's O(n^2) VRAM OOMs at
    # >=8k on L4 (-> NaN, expected here). Notebook 11 (A100) is where phi fills.
    r = run_behavior_for_model(key, cfg, config, seed=SEED); r['family'] = cfg.get('family')
    save_json(r, out)                      # overwrite the NaN phi in the test folder
    b = r['behavior']
    print('phi per_ctx =', b.get('niah_per_context'))
    print('phi niah_long =', b.get('niah_long'), ' maxlen =', b.get('niah_maxlen'))
""")

# ---------------------------------------------------------------------------
# 10b — mistral knockout sweep (drop vs #heads ablated), base + instruct
# ---------------------------------------------------------------------------
MISTRAL_KO_SWEEP = code("""
# mistral knockout sweep: why is knockout_drop ~0? Ablate the top-k detected
# retrieval heads for growing k and watch the recall drop. Reuses the existing
# profile's head set/scores (no re-detection). Cheaper than KnockoutEvaluator.run
# (baseline once + one masked pass per k; skips the C1 random control).
import json
from pathlib import Path
import numpy as np
from scripts._common import save_json
from rhp.panel import load_panel, model_cfg
from rhp.loader import load_model_any as load_model
from rhp.knockout import KnockoutEvaluator
from src.retrieval_head_detector import RetrievalHeadDetector
from src.repro import set_determinism

config = load_panel(CONFIG); SEED = 42
out = Path(RESULTS_DIR)/'diagnostics'/'mistral_knockout_sweep.json'
out.parent.mkdir(parents=True, exist_ok=True)
KS = [10, 20, 30, 50]
thr = config['niah'].get('score_threshold', 0.1)

if out.exists():
    print('mistral knockout sweep already done -> skip')
    print(json.dumps(json.load(open(out, encoding='utf-8')), indent=1)[:1200])
else:
    results = {}
    for key in ['mistral7b_v03', 'mistral7b_v03_instruct']:
        prof = Path(RESULTS_DIR)/'profile'/f'{key}_seed{SEED}.json'
        d = json.load(open(prof, encoding='utf-8'))
        heads = [tuple(h) for h in d['argmax_heads']]
        scores = np.asarray(d['argmax_scores'], dtype=float)
        ordered = sorted(heads, key=lambda lh: scores[lh[0], lh[1]], reverse=True)
        ks = sorted({min(k, len(ordered)) for k in KS + [len(ordered)]})
        set_determinism(SEED)
        model, tok = load_model(model_cfg(config, key), key)
        try:
            det = RetrievalHeadDetector(model, tok, config, score_threshold=thr, seed=SEED)
            samples = det.generate_niah_samples(60, [4096], [0.25, 0.5, 0.75])
            ko = KnockoutEvaluator(model, tok, config)
            base = float(np.mean(ko._accuracy(samples, None)))
            curve = []
            for k in ks:
                acc = float(np.mean(ko._accuracy(samples, ordered[:k])))
                curve.append({'k': k, 'mask_acc': acc, 'drop': base - acc})
                print(f'{key:24s} k={k:3d} mask_acc={acc:.3f} drop={base-acc:.3f}')
            results[key] = {'n_detected': len(ordered), 'baseline': base, 'curve': curve}
        finally:
            del model, tok
            import gc, torch; gc.collect(); torch.cuda.empty_cache()
    save_json(results, out)
    print('\\nsaved', out)
    print('READ: if drop grows with k -> top-30 was too few (methods fix); if drop '
          'stays ~0 even at all heads -> mistral retrieval is genuinely redundant/'
          'distributed (a real finding).')
""")

# ---------------------------------------------------------------------------
# 10c — refresh analysis so phi enters RQ2 (cheap; optional)
# ---------------------------------------------------------------------------
REFRESH = code("""
# Refresh the prediction analysis on the TEST folder so phi's new long-context
# behaviour enters RQ2. (CPU; quick.) The knockout sweep is a standalone
# diagnostic and is not part of this analysis.
import subprocess, sys
P2 = '/content/rope-part2'
cmd = [sys.executable, f'{P2}/scripts/run_prediction.py', '--seed', '42',
       '--retest-seed', '123', '--results-dir', RESULTS_DIR, '--config', CONFIG,
       '--part1-repo', '/content/rope-part1']
print('>>', ' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:] if r.returncode else '')
print('Done. phi now has a real niah_long; re-download rhprofile_results_other.')
""")


def nb_10_followups():
    cells = [md(
        "# 10 · Follow-ups (separate, L4) — phi long-context behaviour + mistral "
        "knockout sweep\n"
        "**GPU: L4 24 GB** (phi 3.8 B + mistral 7 B both fit; no A100 needed). "
        "Short (~5 h). **Independent of notebook 09** — run it before or after, "
        "any order. Writes to the SAME `rhprofile_results_other` test folder "
        "(seeded no-clobber from main; main is never touched).\n\n"
        "`Runtime → Run all` after pasting tokens in Cell 2 + clicking the Drive "
        "popup once. Resume-safe (skip-if-exists).\n\n"
        "| Task | Why | Output |\n|---|---|---|\n"
        "| phi long-context behaviour | phi was 4k-only (niah_long=NaN) but is 128k-native | `behavior/phi35_mini_seed42.json` (overwritten) |\n"
        "| mistral knockout sweep | knockout_drop≈0 — is it redundancy or too-few-heads? | `diagnostics/mistral_knockout_sweep.json` |")]
    cells += [
        md("### Setup — run cells 0–3 once. Edit `PART1`/`PART2` owners + paste tokens in Cell 2."),
        GPU_DRIVE_TEST, SETUP_PIP, SETUP_CLONE, SETUP_PATHS,
    ]
    cells.append(md("## 1 — Seed the test folder from main (no-clobber; main read-only)"))
    cells.append(SEED_FROM_MAIN)
    cells.append(md("## 2 — phi35_mini long-context behaviour (overwrite short-context phi)"))
    cells.append(PHI_LONGCTX)
    cells.append(md("## 3 — mistral knockout sweep (drop vs #heads ablated)"))
    cells.append(MISTRAL_KO_SWEEP)
    cells.append(md("## 4 — refresh prediction analysis so phi enters RQ2"))
    cells.append(REFRESH)
    return notebook(cells, gpu=True)


def main():
    name = "10_followups_colab.ipynb"
    nb = nb_10_followups()
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name, f"({len(nb['cells'])} cells)")
    print("00-09 untouched. GPU: L4 24GB. Writes to rhprofile_results_other.")


if __name__ == "__main__":
    main()
