# The Retrieval-Head Profile

### A Training-Free Diagnostic for Long-Context Ability and Its Inheritance Across Model Transformations

This repository contains all code for the **Part 2** study in the retrieval-head
series:

> **The Retrieval-Head Profile: A Training-Free Diagnostic for Long-Context
> Ability and Its Inheritance Across Model Transformations**
> *(Türkçe: Retrieval-Head Profili: Uzun Bağlam Yeteneği için Eğitimsiz Bir
> Teşhis Aracı ve Model Dönüşümleri Boyunca Devre Kalıtımı)*

It is the direct continuation of **Part 1**, *"Does RoPE Prevent or Degrade
Retrieval Heads? A Mechanistic Analysis Across Model Families."* Part 1
**established** the mechanism (retrieval heads are real, causal, and depend on
low-frequency RoPE dimensions; 4–5 models). Part 2 **opens it to the ecosystem**:
profile diversity across ~24 models, the predictive power of the profile for
behavioural long-context scores, and — for the first time — the quantitative
inheritance of a known circuit across instruction tuning, quantization, and
distillation.

**Everything is inference-only.** No training. Every experiment runs on a single
24 GB GPU (NVIDIA L4 / Colab A100 optional). Target venue: **TMLR** (primary);
ICLR / NeurIPS mech-interp track / COLM as alternatives.

---

## Research questions

| RQ | Question |
|----|----------|
| **RQ1 — Profile** | How does the retrieval-head profile (count, distribution, concentration, frequency signature) vary across models? |
| **RQ2 — Prediction** | Do training-free profile metrics predict behavioural long-context scores (NIAH, RULER)? Can a "10-minute head scan" be a benchmark proxy? |
| **RQ3 — Inheritance** | Are the identity, count, and frequency dependence of retrieval heads preserved along base → instruct → quantized → distilled? Which transformation damages the circuit most? |
| **RQ4 — Mechanism (conditional)** | When SFT/RLHF shifts long-context recall, can the change be localised to retrieval heads (identity loss vs weakening)? |

A weak or null RQ2 result is **reported as a finding** ("a mechanistic profile
does not track the behavioural score"), with CIs — not buried.

---

## What is inherited from Part 1

Part 2 does **not** fork the Part-1 code. It adds the Part-1 repository to the
path and imports its proven modules directly (`rhp/_paths.py` resolves the
location), so a fix in either place benefits both papers and the two stay
bit-for-bit compatible:

| Inherited (`src/…` in the Part-1 repo) | Used for |
|---|---|
| `retrieval_head_detector.py` | argmax-proxy detector, paired-seed NIAH specs, intersection-drop |
| `activation_patching.py` | hook-based dimension zeroing, population patch, McNemar, perplexity specificity (0.33 rule) |
| `niah_evaluator.py` | generation-based NIAH + head masking |
| `dimension_utility.py` | RoPE `freq_order`, Q-projection norms, layer-clustered stats |
| `stats_utils.py` | Cohen's d, bootstrap CI, BH-FDR, layer-clustered permutation, Jaccard |
| `corpus.py`, `repro.py`, `model_loader.py` | shared PG-19 haystack, determinism + environment capture, revision-pinned/quantization-aware loading |

New code lives in **`rhp/`** (the Retrieval-Head Profile package).

---

## The retrieval-head profile

For each model we extract a standardised vector (proposal §3.1):

| Metric | Meaning | Experiment | Code |
|---|---|---|---|
| `n_heads`, `n_heads_copy` | heads found by the **two** detectors | E1 | `copy_score_detector.py`, inherited detector |
| `frac` | retrieval heads / total heads | E1 | `profile.py` |
| `detector_agreement` | Jaccard(argmax-set, copy-set) — reliability | E1 | `profile.detector_agreement` |
| `gini`, `zero_fraction` | score concentration (Qwen-type vs OLMo-type) | E4 | `profile.concentration` |
| `layer_com`, early/mid/late | layer-depth distribution | E4 | `profile.layer_profile` |
| `gqa_*` | KV-group distribution (not a sharing artifact) | E5 | `profile.gqa_group_distribution` |
| `freq_com`, `freq_width` | spectral signature centre / width | E2 | `freq_signature.py` |
| `frequency_effect`, `specificity_ratio` | low−high-freq dose patch + perplexity specificity | E12 / C4 | inherited `run_population_patching` |
| `knockout_drop`, `dissociation` | causal mask effect vs random heads | E3 / C1 | `knockout.py` |

The **second detector** is the key new ingredient: a *teacher-forced copy score*
(close to Wu et al. 2025) that complements Part-1's single-pass argmax proxy.
Running both, and reporting their agreement, keeps Part-1's "validate the claim,
not the metric" discipline.

---

## Project structure

```
retrieval-head-profile/
├── configs/
│   └── panel.yaml          # 24 models (10 families) + 2 quant rings + 4 lineages + hyperparams
├── rhp/                    # the new package
│   ├── _paths.py           # locate & import the inherited Part-1 src/
│   ├── panel.py            # panel + lineage helpers
│   ├── copy_score_detector.py   # E1 — teacher-forced copy score (2nd detector)
│   ├── profile.py          # E1/E4/E5 — profile vector (Gini, layer, GQA, agreement)
│   ├── freq_signature.py   # E2 — 8-window spectral sweep (+ C2/C3 controls)
│   ├── knockout.py         # E3 — knockout double dissociation (+ C1, McNemar)
│   ├── ruler.py            # E7 — RULER subset (multikey / multivalue / vartrack)
│   ├── inheritance.py      # E10–E15 — identity / function / frequency / bridge / quant / localize
│   └── prediction.py       # E8/E9 — profile→behaviour prediction, test-retest
├── scripts/
│   ├── _common.py          # path bootstrap + single-model profile/behaviour pipelines
│   ├── run_profile.py      # Block A (E1–E5), resume + time-budget
│   ├── run_behavior.py     # Block B (E6 NIAH + E7 RULER)
│   ├── run_inheritance.py  # Block C (E10–E15) — CPU analysis
│   └── run_prediction.py   # E8 + E9 — CPU analysis
├── notebooks/              # Colab "tasks", each ≤24 h, resume-safe to Drive
│   ├── build_notebooks.py  # regenerates the .ipynb files
│   ├── 00_pilot_colab.ipynb
│   ├── 01_panel_profile_colab.ipynb
│   ├── 02_panel_behavior_colab.ipynb
│   ├── 03_inheritance_colab.ipynb
│   ├── 04_prediction_analysis_colab.ipynb
│   └── 05_robustness_optional_colab.ipynb
├── docs/
│   ├── EXPERIMENTS.md      # full E/C/R/O inventory → code + notebook map
│   ├── REPRODUCIBILITY.md  # exact reproduction steps
│   └── LIMITATIONS.md      # threats to validity (for the paper)
├── requirements.txt
└── README.md
```

---

## The model panel

**18 base/instruct models across 8 families** + 2 quantized rings — exactly the
proposal §3.2 draft panel (12 base families + 6 instruct/it derivatives that also
serve the lineage chains). All ≤9 B so each loads alone in 8-bit on a 24 GB GPU.
Five are **core** (3 seeds + test-retest); the rest get a single seed
(proposal §5 / R5).

| Family | Models |
|---|---|
| Llama | 3.2-3B*, 3.1-8B*, 3.1-8B-Instruct, 2-7B |
| Qwen | 2.5-3B*, 2.5-3B-Instruct, 2.5-7B*, 2.5-7B-Instruct |
| Gemma | 2-2B, 2-9B*, 2-9B-it |
| Mistral | 7B-v0.3, 7B-Instruct-v0.3 |
| OLMo | 2-7B, 2-7B-Instruct |
| Phi | 3.5-mini-instruct |
| Falcon | 3-7B-Base |
| StableLM | 2-1.6B |

`*` = core. **Quant rings (4)** used inside the lineages (E10–E14):
`Qwen2.5-7B-Instruct-AWQ`, `…-GPTQ-Int4` (pre-quantized repos), and
bitsandbytes-NF4 4-bit of `Llama-3.1-8B-Instruct` and `Gemma-2-9B-it`
(load-time, no separate repo — the §3.4 "→ 4-bit quantized" rings). If §7's "18
points few" risk bites, the panel expansion path is left open; if the §4 budget
bites, trim to the 14-model core.

> **Before the reportable run, pin every `revision` to a commit SHA** (the loader
> warns while a revision is `"main"`). Pre-screening (~2–4 GPU-h/model) finalises
> the list; drop any model that lacks gated access or a clean NIAH baseline,
> following the §4 priority order.

### Inheritance lineages (RQ3)

| Lineage | Chain | Sibling (distillation) |
|---|---|---|
| Qwen | 7B → 7B-Instruct → AWQ-4bit → GPTQ-4bit | 3B-Instruct |
| Llama | 8B → 8B-Instruct → 4bit (NF4) | — |
| Gemma | 9B → 9B-it → 4bit (NF4) | 2B |
| Mistral | 7B-v0.3 → 7B-Instruct-v0.3 | — |

---

## Experiment inventory

The complete inventory (proposal §4) is mapped to code and notebooks in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Summary:

- **Block A — Profile (RQ1):** E1 dual detector · E2 frequency signature ·
  E3 knockout · E4 concentration/layer · E5 GQA control.
- **Block B — Behaviour & prediction (RQ2):** E6 NIAH sweep · E7 RULER subset ·
  E8 profile→behaviour prediction · E9 test-retest.
- **Block C — Inheritance (RQ3):** E10 identity · E11 function · E12 frequency ·
  E13 bridge · E14 quantization ablation · E15 RQ4 localisation.
- **Controls (C1–C6):** random-head, layer-matched, random-dim, perplexity
  specificity (0.33 rule), position, intersection-drop.
- **Robustness (R1–R7):** threshold, detector, coverage, quantization, seed,
  sample-size, haystack-source.
- **Optional (O1–O5):** surgical band extension, OLMo-2 checkpoint inheritance,
  merged-model profile, long-context fine-tune variant, attention-mass detector.

**Core (non-droppable):** E1–E3, E6, E10–E12, C1–C4.

---

## How to run

### Setup

```bash
pip install -r requirements.txt
# tell rhp where the Part-1 repo lives (for the inherited src/):
export RHP_PART1_REPO=/path/to/Does-RoPE-Prevent-or-Degrade-Retrieval-Heads-...
# or place this project as a sibling of the Part-1 repo (auto-detected),
# or pass --part1-repo PATH to every script.
```

Gated models (Llama, Gemma) need a HuggingFace token: `export HF_TOKEN=hf_...`.

### Local CLI (single 24 GB GPU)

```bash
# Block A — profile the 5 core models (resume-safe; stops at the time budget)
python scripts/run_profile.py --models core --seed 42
python scripts/run_profile.py --models core --seed 123     # extra seeds (R5)
python scripts/run_profile.py --models all  --time-budget-hours 20

# Block B — behavioural targets
python scripts/run_behavior.py --models all --time-budget-hours 20

# Block C — inheritance (CPU; needs the lineage models' profiles+behaviours first)
python scripts/run_inheritance.py --lineage all

# E8 prediction + E9 test-retest (CPU)
python scripts/run_prediction.py --seed 42 --retest-seed 123
```

Outputs land under `results/{profile,behavior,inheritance,analysis,robustness}/`.
Every per-model result is written atomically and **skipped on re-run**, so an
interrupted sweep resumes cleanly.

### Google Colab (recommended for the panel)

Each notebook is a **self-contained task sized to finish in one session
(≤24 h)** and resume-safe to Google Drive. Run them in order:

| Notebook | Task | GPU |
|---|---|---|
| `00_pilot_colab.ipynb` | WP1 pilot — full pipeline on 3 models | L4 |
| `01_panel_profile_colab.ipynb` | Block A (E1–E5) for a model subset | L4 |
| `02_panel_behavior_colab.ipynb` | Block B (E6–E7) | L4 |
| `03_inheritance_colab.ipynb` | Block C (E10–E15) | L4 |
| `04_prediction_analysis_colab.ipynb` | E8 + E9 | **None (CPU)** |
| `05_robustness_optional_colab.ipynb` | R1/R3/R4/R6/R7 + O-series | L4 |

In each: edit the `PART1`/`PART2` repo owners and paste tokens in the setup
cell, set `MODEL_SUBSET` to the chunk this session should do, and
`TIME_BUDGET_HOURS` (default 11 h — under the free-tier disconnect; Colab Pro
allows 24 h). Re-run across sessions until the panel is complete. Regenerate the
notebooks after editing the generator with `python notebooks/build_notebooks.py`.

---

## Statistics plan (inherited discipline, proposal §5)

- **Layer-clustered permutation test** for head-level comparisons
  (pseudoreplication guard).
- **Exact McNemar + bootstrap CI** for paired conditions (knockout, frequency
  patch).
- **Benjamini–Hochberg FDR** across the panel correlations (E8).
- **LOO cross-validation**, ≤3 predictors, for the prediction regression (18+
  points → over-fitting guard).
- **Family-demeaned** correlations (family is a confound).
- Main results at **3 seeds** (42/123/2024) on the core models; the rest single
  seed with a 3-seed verification on rings that show a difference — stated
  explicitly in every table.
- Claims kept at the correlation + CI level; no taxonomy claim is made on this
  many points.

---

## Result schema

Each `results/profile/<model>_seed<seed>.json`:

```json
{
  "model": "qwen25_7b", "family": "qwen", "seed": 42, "context_length": 4096,
  "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
  "argmax_heads": [[l,h], ...], "copy_heads": [[l,h], ...],
  "argmax_scores": [[...]], "copy_scores": [[...]],
  "profile": {
    "n_heads": 0, "n_heads_copy": 0, "frac": 0.0,
    "detector_agreement": {"jaccard": 0.0, ...},
    "concentration": {"gini": 0.0, "zero_fraction": 0.0, ...},
    "layer_profile": {"layer_com_weighted": 0.0, ...},
    "gqa": {"full_group_fraction": 0.0, ...},
    "freq_signature": {"freq_com": 0.0, "freq_width": 0.0, "drop": [...], ...},
    "freq_patch": {"frequency_effect": 0.0, "specificity_verdict": "...", ...},
    "knockout": {"knockout_drop": 0.0, "mcnemar": {...}, ...},
    "scalars": { ... }      // flat metrics for the E8 prediction table
  },
  "environment": { ... }    // python/packages/hardware/git (Part-1 capture)
}
```

---

## GPU budget (proposal §4)

| Block | Hours |
|---|---|
| A (E1–E3) | ≈ 90 |
| B (E6, E7, E9) | ≈ 75 |
| C (E10–E15) | ≈ 75 |
| R-series extras (R3, R4, R6, R7) | ≈ 35 |
| O-series (if all) | ≈ 25 |
| **Total** | **≈ 300 GPU-h** (core ≈ 240; ~5–6 weeks on L4, interruptible) |

Each model is an independent, resumable job — built for chunked Colab sessions.

---

## Reproducibility

Pinned revisions, paired-seed harness, deterministic seeding, and a full
environment block in every result JSON (inherited from Part 1). See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) (threats to validity, mapped to the
checklist).

---

## License

Code: **MIT**. Models and datasets keep their own licenses — Llama-2/3.x and
Gemma are gated with use restrictions; Qwen2.5, OLMo-2, Mistral, Yi, InternLM are
Apache-2.0; PG-19 per its dataset card. Cite each in any publication.

## Citation

```bibtex
@article{retrievalheadprofile2026,
  title  = {The Retrieval-Head Profile: A Training-Free Diagnostic for
            Long-Context Ability and Its Inheritance Across Model Transformations},
  author = {Anonymous Authors},
  year   = {2026}
}
```
