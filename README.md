# The Retrieval-Head Profile

### Retrieval Heads Survive Deployment: a training-free mechanistic profile and its inheritance through instruction tuning and 4-bit quantization

This repository holds **all code and all results** for Part 2 of the retrieval-head
series. Everything is inference-only (no training), and every result JSON in
[`datas/results/`](datas/results/) is checked into the repo, so the full analysis
can be regenerated from a clean clone **without a GPU**.

> The write-up (a paper being prepared for peer review) is kept in a **private,
> local `paper/` directory that is not part of this public repository**. What is
> published here is the code and the complete data; the findings below summarise
> what the paper reports.

---

## Contents

- [Background: what is a retrieval head?](#background-what-is-a-retrieval-head)
- [What Part 2 adds: the three research questions](#what-part-2-adds-the-three-research-questions)
- [TL;DR: what we found](#tldr-what-we-found)
- [Repository layout](#repository-layout)
- [The data (start here)](#the-data-start-here)
- [Inspect the results (no GPU needed)](#inspect-the-results-no-gpu-needed)
- [Reproduce from scratch (needs a 24 GB GPU)](#reproduce-the-results-from-scratch-needs-a-24-gb-gpu)
- [The model panel](#the-model-panel)
- [The retrieval-head profile, metric by metric](#the-retrieval-head-profile-metric-by-metric)
- [Experiment inventory](#experiment-inventory)
- [How inheritance is measured (RQ3)](#how-inheritance-is-measured-rq3)
- [Statistics](#statistics)
- [Glossary](#glossary)
- [Documentation, license, citation](#documentation)

---

## Background: what is a retrieval head?

Large language models solve "needle in a haystack" tasks (find one fact buried in
a long context) using a small number of **attention heads** that learn to copy the
relevant token from far back in the sequence to the current position. Wu et al.
(2025) called these **retrieval heads**: mask the handful of them and long-context
recall collapses; mask the same number of random heads and nothing happens.

Part 1 of this series (*"Does RoPE Prevent or Degrade Retrieval Heads? A
Mechanistic Analysis Across Model Families"*) established, on 4-5 models, **how**
these heads work: they lean on the **low-frequency dimensions** of Rotary
Position Embeddings (**RoPE**). RoPE rotates each query/key dimension at a
different frequency; the low-frequency dimensions rotate slowly and therefore
encode *long-range* position, which is exactly what a head needs to reach back
across thousands of tokens. Part 1 showed causally (by patching individual RoPE
frequency bands) that retrieval heads depend on those slow dimensions.

**Part 2 asks a different, ecosystem-level question.** Given that we now *know*
this circuit exists and how it works, (1) how does it **vary** across the many
open models people actually deploy, (2) does a cheap, training-free measurement of
it **predict** how good a model is at long context, and (3) does the circuit
**survive** the transformations every deployed model goes through (instruction
tuning, quantization, distillation)? To answer these we compute a standardised
**retrieval-head profile** for each model and track it across model lineages.

---

## What Part 2 adds: the three research questions

| RQ | Question | Answer (this repo) |
|----|----------|--------------------|
| **RQ1 - Profile diversity** | How does the retrieval-head profile (count, concentration, layer depth, RoPE frequency signature) vary across families? | Highly family-specific. All models concentrate retrieval, but the *frequency signature*, head count, and depth differ enormously; Mistral even retrieves in a *distributed* way. |
| **RQ2 - Prediction** | Do training-free profile metrics predict behavioural long-context scores (NIAH, RULER)? Can a "10-minute head scan" be a benchmark proxy? | A **confound-limited null**. The raw correlation is strong but tracks the model's context-window *class*; on a clean continuous target the within-family signal vanishes. |
| **RQ3 - Inheritance** | Are the identity, function, and frequency dependence of retrieval heads preserved along base -> instruct -> quantized -> distilled? Which transformation hurts most? | The circuit is **inherited, not rebuilt**. Instruction tuning and all three 4-bit quantization methods preserve it; GPTQ degrades the frequency signature the most. |

A weak or null RQ2 result is **reported as a finding**, not buried. This honesty
is the point of the RQ2 section (see below).

---

## TL;DR: what we found

### RQ1 - the profile is real but family-specific

Every model in the panel concentrates retrieval into a small set of heads
(**Gini 0.88-0.97**), and ablating the top heads collapses NIAH recall in
**15 of 18** models. But the *shape* of the profile varies enormously:

- **RoPE frequency signature** (`frequency_effect`, our causal low- vs
  high-frequency dose patch) ranges from strongly low-frequency (Qwen, down to
  about -0.99) to essentially flat (Llama-3.x, Gemma-2-9b-it, Falcon, near 0).
- **Head count** spans roughly 18 to 112 depending on the family.
- **Mistral is the clean exception**: retrieval is *distributed* across many
  heads, so ablating even *all* detected heads barely dents recall. The full
  knockout sweep is in
  [`datas/results/diagnostics/mistral_knockout_sweep.json`](datas/results/diagnostics/mistral_knockout_sweep.json).

The takeaway: "retrieval heads" is a real, causal, concentrated phenomenon
everywhere, but there is **no single canonical profile** - each family builds the
circuit its own way.

### RQ2 - a confound-limited null, reported honestly

Retrieval-head **utility** (how much recall depends on the heads,
`utility_partial_spearman`) correlates strongly with a model's context window:

| target | raw Spearman rho | BH-adjusted p | family-demeaned rho | family-demeaned p |
|---|---|---|---|---|
| `niah_maxlen` (categorical) | **-0.85** | 0.0008 | -0.72 | 0.014 |
| `niah_long` (NIAH long slice) | -0.81 | 0.0021 | -0.72 | 0.014 |
| `ruler_vartrack` (continuous) | -0.60 | 0.12 (n.s.) | **-0.26** | **0.57 (n.s.)** |

The strong headline number is against `niah_maxlen`, but that target is
essentially the model's **context-window class** (4k vs 8k vs 32k), not a
continuous ability, so the correlation is **confounded**. On the clean continuous
target (RULER variable-tracking) the raw correlation is already not significant
after correction, and once we remove family means it **vanishes**
(rho = -0.26, p = 0.57). The NIAH-derived targets survive family-demeaning
precisely *because* they still encode context-window class.

So the appealing "profile a model in 10 minutes and predict its long-context
benchmark" does **not** hold cleanly. We report this as a confound-limited null.
All of these numbers live in
[`datas/results/analysis/prediction_e8.json`](datas/results/analysis/prediction_e8.json)
and are regenerable (see below).

### RQ3 - the circuit survives deployment

**Base to instruct.** The retrieval-head *identity* (which heads they are) is
inherited for Qwen (copy-Jaccard 0.889, above its `0.8 * R_self` reliability
threshold) and Llama (0.826), and partially for Gemma.

**Quantization (E14, on Qwen-2.5-7B-Instruct).** All three 4-bit methods, which
are genuinely different algorithms, preserve the circuit, at genuinely different
fidelities:

| 4-bit method | identity Jaccard vs fp16 | per-head score Spearman | RoPE freq. sign kept | freq. magnitude lost |
|---|---|---|---|---|
| bitsandbytes NF4 | 0.838 | 0.940 | yes | about 3% |
| AWQ | **0.971** | 0.933 | yes | about 9% |
| GPTQ | 0.857 | 0.934 | yes | about 29% |

AWQ is the most faithful; GPTQ keeps the *identity* but loses the most
frequency-effect magnitude. The circuit is **inherited, not rebuilt**: the same
heads, doing the same job, leaning on the same RoPE dimensions, after deployment.

---

## Repository layout

```
retrieval-head-profile/
├── datas/
│   ├── results/            <-- ALL results, checked in (89 JSON files)
│   │   ├── profile/        E1-E5  per-model retrieval-head profile
│   │   ├── behavior/       E6-E7  NIAH + RULER long-context scores
│   │   ├── utility/        E8     how much recall depends on the heads
│   │   ├── inheritance/    E10-E15 per-lineage inheritance + E14 quant ablation
│   │   ├── analysis/       E8/E9  panel prediction + test-retest reliability
│   │   ├── robustness/     R3/R4/R6/R7 threshold / quant / sample-size / haystack
│   │   └── diagnostics/    mistral distributed-retrieval knockout sweep
│   └── README.md           <-- data dictionary: every file and field explained
├── rhp/                    the Retrieval-Head Profile package (new Part-2 code)
│   ├── _paths.py           locate + import the inherited Part-1 src/
│   ├── panel.py            panel + lineage helpers (reads configs/panel.yaml)
│   ├── copy_score_detector.py   2nd detector: teacher-forced copy score
│   ├── profile.py          Gini, layer distribution, GQA, detector agreement
│   ├── freq_signature.py   8-window RoPE frequency sweep -> freqEff scalar
│   ├── knockout.py         causal knockout + exact McNemar
│   ├── ruler.py            RULER subset (multikey / multivalue / vartrack)
│   ├── utility.py          retrieval-head utility (Cohen's d, partial Spearman)
│   ├── inheritance.py      E10-E15 identity / function / frequency / quant
│   └── prediction.py       E8 profile->behaviour, E9 test-retest
├── scripts/                CLI entry points (one per block) + _common.py bootstrap
│   ├── run_profile.py      Block A  (E1-E5)
│   ├── run_behavior.py     Block B  (E6 NIAH + E7 RULER)
│   ├── run_utility.py      utility  (E8 weight-space predictors)
│   ├── run_inheritance.py  Block C  (E10-E15, CPU)
│   └── run_prediction.py   E8 + E9  (CPU)
├── notebooks/              Colab notebooks 00-14, each chunk sized to finish <24 h
├── docs/                   EXPERIMENTS / REPRODUCIBILITY / LIMITATIONS / PREREGISTRATION
├── configs/panel.yaml      the model panel, lineages, and hyperparameters
└── requirements.txt
```

The single results folder is [`datas/results/`](datas/results/). Its
[data dictionary](datas/README.md) documents every subfolder, filename, and JSON
field. (The `paper/` directory holding the LaTeX write-up is deliberately absent
from this public repo, as noted above.)

---

## The data (start here)

Every number in the study is read straight from `datas/results/`. Nothing is
hand-entered. Files are named `<model>_seed<seed>.json`, e.g.
`qwen25_7b_instruct_seed42.json`; a quantization suffix (`_bnb4`, `_awq4`,
`_gptq4`) marks the 4-bit rings used only in the inheritance experiments.

A **profile** record (the core artifact) looks like:

```json
{
  "model": "qwen25_7b", "family": "qwen", "seed": 42,
  "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
  "argmax_heads": [[l, h], ...],      // detector 1: argmax proxy (from Part 1)
  "copy_heads":   [[l, h], ...],      // detector 2: teacher-forced copy score (new)
  "profile": {
    "scalars": {                      // flat metrics consumed by the E8 table
      "gini": 0.95, "layer_com_weighted": 0.77,
      "frequency_effect": -0.92,      // causal low- vs high-freq RoPE dose patch
      "knockout_drop": 0.76, ...
    },
    "concentration": {...}, "layer_profile": {...},
    "freq_signature": {...}, "knockout": {...}
  },
  "environment": {...}                // python / packages / hardware / git SHA
}
```

There are five other record types (behavior, utility, inheritance, analysis,
robustness). Rather than duplicate them here, the
**[data dictionary in datas/README.md](datas/README.md)** documents each field of
each type. Read that file first if you want to work with the data.

---

## Inspect the results (no GPU needed)

Every result is already in the repo, so you can reproduce any number in the paper
without a GPU by reading `datas/results/` directly. For example, the RQ3
cross-method quantization table (E14) comes entirely from one file:

```bash
pip install numpy scipy
python - <<'PY'
import json
e14 = json.load(open("datas/results/inheritance/qwen.json"))["E14_quant_ablation"]
for m in ("bnb4", "awq4", "gptq4"):
    r = e14[m]
    print(m, "identity Jaccard =", round(r["identity_jaccard"], 3),
          "| per-head Spearman =", round(r["per_head_score_spearman"], 3),
          "| freq sign kept:", r["finding_sign_preserved"])
PY
```

And the RQ2 analysis (single + family-demeaned correlations) is fully regenerable
from the per-model JSONs, still no GPU:

```bash
python scripts/run_prediction.py --results-dir datas/results --seed 42
# writes datas/results/analysis/prediction_e8.json over the 18 independent models
# (add --include-rings to reproduce the prediction_e8_withrings.json comparison)
```

Nothing is hand-entered anywhere: the paper's figures and tables are all derived
from these JSONs.

---

## Reproduce the results from scratch (needs a 24 GB GPU)

The full pipeline is inference-only and runs one model at a time on a single
24 GB GPU (Colab L4, or A100 for the longest contexts). Part 2 **imports Part 1's
proven `src/` modules rather than forking them**, so a fix in either place helps
both papers and the two stay bit-for-bit compatible. Point it at the Part-1
checkout first:

```bash
pip install -r requirements.txt
export RHP_PART1_REPO=/path/to/Does-RoPE-Prevent-or-Degrade-Retrieval-Heads
# (or place this repo as a sibling of the Part-1 repo, which is auto-detected,
#  or pass --part1-repo PATH to every script)
export HF_TOKEN=hf_...        # for gated models (Llama, Gemma)
```

Then run the blocks. Each per-model result is written atomically and **skipped on
re-run**, so an interrupted sweep resumes cleanly:

```bash
python scripts/run_profile.py     --models all --time-budget-hours 20   # Block A: E1-E5   -> profile/
python scripts/run_behavior.py    --models all --time-budget-hours 20   # Block B: E6+E7   -> behavior/
python scripts/run_utility.py     --models all                          # utility E8       -> utility/
python scripts/run_inheritance.py --lineage all                         # Block C: E10-E15 -> inheritance/ (CPU)
python scripts/run_prediction.py  --seed 42 --retest-seed 123           # E8 + E9          -> analysis/    (CPU)
```

`--models core` restricts to the five core models; `--seed` sets the RNG seed
(the core models were run at 42/123/2024 for the test-retest). Blocks A and B are
independent and can run in parallel.

### Colab

Colab kills a session past about 24 h, so the panel is pre-split into fixed chunks
that each finish in one session and write to Google Drive. Run them in order; the
profile and behaviour chunks are independent and can run in parallel accounts.
`notebooks/` is numbered `00` (pilot) through `14` (final analysis), and the
`build_*.py` files regenerate the `.ipynb` from `configs/panel.yaml`. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact run order and the
per-notebook GPU requirement.

---

## The model panel

**18 independent models across 8 families**, all <= 9 B so each loads alone on a
24 GB GPU. Five are **core** (multiple seeds + test-retest); the rest are
single-seed. The full spec (HuggingFace ids, pinned revisions, hyperparameters)
is in [`configs/panel.yaml`](configs/panel.yaml).

| Family | Models |
|---|---|
| Qwen | 2.5-3B, 2.5-3B-Instruct, 2.5-7B, 2.5-7B-Instruct |
| Llama | 3.2-3B, 3.1-8B, 3.1-8B-Instruct, 2-7B |
| Gemma | 2-2B, 2-9B, 2-9B-it |
| Mistral | 7B-v0.3, 7B-Instruct-v0.3 |
| OLMo | 2-7B, 2-7B-Instruct |
| Phi | 3.5-mini-instruct |
| Falcon | 3-7B-Base |
| StableLM | 2-1.6B |

**Quantized rings (E14, RQ3):** load-time 4-bit variants used only inside the
lineages, so they are **excluded from the RQ2 panel by default** (they are not
independent training runs). They are `Qwen2.5-7B-Instruct` in AWQ, GPTQ-Int4, and
bitsandbytes-NF4, plus NF4 rings of `Llama-3.1-8B-Instruct` and `Gemma-2-9b-it`.

### Inheritance lineages (RQ3)

| Lineage | Chain | Distillation sibling |
|---|---|---|
| Qwen | 7B -> 7B-Instruct -> {NF4, AWQ, GPTQ} 4-bit | 3B-Instruct |
| Llama | 8B -> 8B-Instruct -> NF4 4-bit | - |
| Gemma | 9B -> 9b-it -> NF4 4-bit | 2B |
| Mistral | 7B-v0.3 -> 7B-Instruct-v0.3 | - |

---

## The retrieval-head profile, metric by metric

For each model we extract a standardised vector of metrics. The key new
ingredient over Part 1 is the **second detector**: Part 1 used a single argmax
proxy (which token does the head attend to most?); Part 2 adds a **teacher-forced
copy score** (close to Wu et al. 2025: does the head actually copy the needle
token?) and runs both, reporting their agreement. This follows Part 1's discipline
of validating the *claim*, not a single metric.

| Metric (`scalars` key) | What it measures | How | Exp |
|---|---|---|---|
| `n_heads`, `n_heads_copy` | number of retrieval heads found by each detector | thresholded detector scores | E1 |
| `detector_agreement` (`detector_jaccard`) | do the two detectors agree on the head set? | Jaccard(argmax-set, copy-set) | E1 |
| `gini` | how concentrated retrieval is over heads (high = a few heads do it all) | Gini coefficient of the copy scores | E4 |
| `zero_fraction` | fraction of heads with essentially no retrieval role | share of near-zero scores | E4 |
| `layer_com_weighted` | at what depth retrieval lives (early / mid / late) | score-weighted centre of mass over layers | E4 |
| `freq_com`, `freq_width` | centre and spread of the RoPE frequency band the heads use | 8-window frequency sweep | E2 |
| `frequency_effect` (**freqEff**) | **causal** dependence on low- vs high-frequency RoPE dimensions | dose patch: recall drop when low-freq bands are perturbed minus high-freq | E2/E12 |
| `knockout_drop` | causal importance of the heads | NIAH recall drop when the top heads are masked, vs random-head control (exact McNemar) | E3 |
| `utility` (Cohen's d, `partial_spearman`) | how much a model's recall depends on its heads | masked-recall effect size, partialling out head count | E8 |
| `R_self` (copy-Jaccard test-retest) | reliability ceiling for the inheritance test | Jaccard of the head set across two independent seeds | E9 |

`freqEff` is the metric that carries the Part-1 mechanism into Part 2: a strongly
negative value means the heads genuinely rely on the slow RoPE dimensions; a value
near 0 means the family's heads have a flat spectral signature. When `|freqEff|`
is below 0.05 the model has no real spectral structure, so its `freq_com` is
treated as noise (set to NaN) and dropped from correlations.

---

## Experiment inventory

The complete inventory (experiments E, controls C, robustness R, optional O) is
mapped to code and notebooks in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
Summary:

- **Block A - Profile (RQ1):** E1 dual detector, E2 frequency signature, E3
  knockout double-dissociation, E4 concentration + layer depth, E5 GQA control.
- **Block B - Behaviour + prediction (RQ2):** E6 NIAH sweep, E7 RULER subset
  (multikey / multivalue / variable-tracking), E8 profile->behaviour prediction,
  E9 test-retest reliability.
- **Block C - Inheritance (RQ3):** E10 identity, E11 function (knockout transfer),
  E12 frequency, E13 bridge, **E14 cross-method quantization ablation**, E15
  localisation.
- **Controls (C1-C6):** random-head, layer-matched, random-dimension, perplexity
  specificity, position, intersection-drop.
- **Robustness (R3/R4/R6/R7):** detector coverage, int8-vs-fp16 stability, sample
  size, haystack source. (See `datas/results/robustness/`.)

---

## How inheritance is measured (RQ3)

The inheritance test is **pre-registered** in
[docs/PREREGISTRATION_inheritance.md](docs/PREREGISTRATION_inheritance.md), so the
decision rules were fixed before looking at the numbers. The core idea is that a
head set cannot be expected to match its parent *more* than the model matches
**itself** across two independent runs. That self-similarity is `R_self` (the
copy-Jaccard test-retest, E9), and it is the denominator we compare against
instead of a naive 1.0.

- **Identity inherited** when the parent->child copy-Jaccard exceeds
  **`0.8 * R_self`** (as similar as the model is to itself, allowing for noise).
- **Function inherited** when masking the *inherited* heads still collapses recall
  in the child (E11).
- **Frequency inherited** when the child keeps the sign, and most of the
  magnitude, of the parent's `freqEff` (E12).

This is why RQ3 reports `identity Jaccard vs 0.8 * R_self`, not a bare Jaccard: it
is calibrated to each model's own reliability ceiling.

---

## Statistics

- **Benjamini-Hochberg FDR** across the panel correlations (E8), applied within
  each target's predictor family.
- **LOO cross-validation**, at most 3 predictors, for the prediction regression
  (guards against over-fitting ~18 points).
- **Family-demeaned** correlations, because family is a confound. This is exactly
  what turns the raw RQ2 correlation into a confound-limited null: the effect is
  stored for three targets (`niah_maxlen`, `niah_long`, `ruler_vartrack`), and
  only the categorical / NIAH-derived ones survive demeaning.
- **Exact McNemar + bootstrap CI** for paired conditions (knockout, frequency
  patch).
- **Layer-clustered permutation** for head-level comparisons (guards against
  pseudoreplication, since heads within a layer are not independent).
- Core results at **3 seeds** (42 / 123 / 2024); the rest single-seed, stated in
  every table. Claims are kept at the correlation + CI level; no taxonomy claim is
  made on this few points.

---

## Glossary

| Term | Meaning |
|---|---|
| **Retrieval head** | an attention head that copies a distant token to the current position; masking a few of them collapses long-context recall |
| **RoPE** | Rotary Position Embedding; rotates each Q/K dimension at its own frequency, so low-frequency dimensions encode long-range position |
| **NIAH** | Needle In A Haystack; retrieve one planted fact from a long distractor context |
| **RULER** | a harder long-context suite; we use multikey, multivalue, and variable-tracking (vartrack) subtasks |
| **freqEff** (`frequency_effect`) | causal low- vs high-frequency RoPE dose-patch effect; negative = the heads depend on slow RoPE dimensions |
| **Gini** | concentration of retrieval scores over heads; high means a few heads do all the work |
| **GQA** | Grouped-Query Attention; several query heads share a KV head. E5 checks retrieval is not just a KV-sharing artifact |
| **Knockout** | masking the top retrieval heads and measuring the recall drop, against a random-head control |
| **Utility** | how much a model's measured recall actually depends on its retrieval heads (E8) |
| **R_self** | a model's head-set self-similarity across two seeds; the reliability ceiling for the inheritance test |
| **Ring** | a load-time quantized variant of a model (NF4 / AWQ / GPTQ), used only in the inheritance experiments |
| **Lineage** | a base -> instruct -> quantized (-> distilled sibling) chain along which inheritance is tracked |

---

## Documentation

| Doc | Contents |
|---|---|
| [datas/README.md](datas/README.md) | **data dictionary** for `datas/results/` (read this to use the data) |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | full E / C / R / O experiment inventory mapped to code + notebooks |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | exact reproduction steps, run order, per-notebook GPU |
| [docs/PREREGISTRATION_inheritance.md](docs/PREREGISTRATION_inheritance.md) | pre-registered inheritance decision rules |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | threats to validity |

---

## License

Code is released under the **MIT License** (see [LICENSE](LICENSE)). Models and
datasets keep their own licenses: Llama-2/3.x and Gemma are gated with use
restrictions; Qwen2.5, OLMo-2, Mistral, Falcon, and StableLM are
Apache-2.0/permissive; PG-19 per its dataset card. Cite each in any publication.

## Citation

```bibtex

}
```

