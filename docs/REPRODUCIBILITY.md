# Reproducibility guide

A reviewer (or you, six months later) should be able to regenerate every number.
This is the step-by-step.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate    # py3.10+
pip install -r requirements.txt
```

Every result JSON embeds an `environment` block (Python, package versions, GPU,
git commit, dirty flag) via the inherited `src.repro.capture_environment`.
Compare it to yours when a number disagrees.

## 2. Locate the inherited Part-1 code

`rhp` imports the Part-1 `src/` package. Make it findable one of three ways:

```bash
export RHP_PART1_REPO=/abs/path/to/Does-RoPE-Prevent-or-Degrade-Retrieval-Heads-...
# OR place this repo as a sibling of the Part-1 repo (auto-detected)
# OR pass --part1-repo /abs/path to every script
```

## 3. Pin model revisions (before the reportable run)

`configs/panel.yaml` ships most revisions as `"main"`. The loader **warns** for
every unpinned model. For the final run, replace each `revision:` with the exact
commit SHA from the model's HF page (the two already-pinned models show the
format). Pinning ties the run to exact weights.

## 4. Authentication

Gated models (Llama, Gemma) need a token:

```bash
export HF_TOKEN=hf_xxx           # also read from .env by src.auth_utils
```

## 5. Determinism

`src.repro.set_determinism(seed)` seeds `random`, `numpy`, `torch`. For the final
run set `reproducibility.strict_determinism: true` in `panel.yaml` (forces
deterministic CUDA kernels — slower, may raise on unsupported ops). Multi-seed
runs report mean ± SD.

## 6. The canonical sequence

```bash
# A — profiles (core at 3 seeds, then the rest at seed 42)
for s in 42 123 2024; do python scripts/run_profile.py --models core --seed $s; done
python scripts/run_profile.py --models panel --seed 42

# B — behaviour
python scripts/run_behavior.py --models all --seed 42

# C — inheritance (after the lineage models have A+B on disk)
python scripts/run_inheritance.py --lineage all

# E8 + E9
python scripts/run_prediction.py --seed 42 --retest-seed 123
```

Everything is resume-safe: re-running skips finished `results/.../<model>_seed<seed>.json`
files. A run interrupted by a Colab disconnect continues exactly where it stopped.

## 7. Colab

Open the notebooks in order (`00`→`05`). Each is a ≤24 h task that writes to
Google Drive and resumes. Set `MODEL_SUBSET` and `TIME_BUDGET_HOURS` per session.
The notebooks call the *same* `scripts._common` helpers as the CLI, so Colab and
local runs are identical.

## 8. What to report in the paper

- Per-model: detector counts (both), Gini, layer COM, freq COM/width, knockout
  drop + McNemar p, frequency_effect + specificity verdict.
- Panel: E8 BH-corrected correlations + LOO R², bounded by E9 test-retest.
- Lineages: E10–E15 ring tables; E14 quant three-level template.
- Robustness: R1–R7 sign/ordering preservation.
- State seed counts explicitly per table (3-seed core vs single-seed panel).
