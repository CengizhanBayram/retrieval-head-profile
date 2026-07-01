# The Retrieval-Head Profile

### Retrieval Heads Survive Deployment: a training-free mechanistic profile and its inheritance through instruction tuning and 4-bit quantization

This repository holds **all code and all results** for Part 2 of the retrieval-head
series. Everything is inference-only (no training), and every result JSON in
[`datas/results/`](datas/results/) is checked into the repo, so the full analysis
can be regenerated from a clean clone without a GPU.

> The write-up (a paper submitted for peer review) is kept in a **private,
> local `paper/` directory that is not part of this public repository**. What is
> published here is the code and the complete data; the findings below summarise
> what the paper reports.

It continues **Part 1** (*"Does RoPE Prevent or Degrade Retrieval Heads? A
Mechanistic Analysis Across Model Families"*), which established the mechanism on
4-5 models: retrieval heads are real, causal, and depend on low-frequency RoPE
dimensions. Part 2 opens that mechanism to the ecosystem: it profiles **18
independent models across 8 families**, asks whether the training-free profile
predicts behavioural long-context scores, and measures whether a *known* circuit
is inherited across instruction tuning and 4-bit quantization.

---

## TL;DR: what we found

**RQ1 - the profile is real but family-specific.** Every model concentrates
retrieval into a small set of heads (Gini 0.88-0.97), and ablating the top heads
collapses NIAH recall in **15 of 18** models. But the *shape* of the profile
varies enormously across families: the causal RoPE frequency signature ranges
from strongly low-frequency (Qwen, `freqEff` down to -0.99) to essentially flat
(Llama-3.x, Gemma-2-9b-it, Falcon, `freqEff` around 0), and the head count spans
18 to 112. **Mistral is the clean exception**: retrieval is *distributed*, so
ablating every detected head barely dents recall (the knockout sweep is in
`datas/results/diagnostics/mistral_knockout_sweep.json`).

**RQ2 - a confound-limited null, reported honestly.** Retrieval-head *utility*
correlates strongly with a model's context window (`niah_maxlen`, Spearman
rho = -0.85, n = 17). But that categorical target mostly encodes the
context-window *class*, not a continuous ability. On a continuous target (RULER
variable-tracking) the raw correlation weakens (rho = -0.60, not significant after
BH correction) and the family-demeaned signal drops further. So the appealing
"10-minute head scan as a benchmark proxy" does **not** hold cleanly. We report
this as a confound-limited null rather than burying it.

**RQ3 - the circuit survives deployment.** Base -> instruct: the retrieval-head
*identity* is inherited for Qwen (Jaccard 0.889 > 0.8 * R_self) and Llama (0.826),
and partially for Gemma. Quantization (E14, on Qwen-2.5-7B-Instruct) preserves the
circuit under **all three** 4-bit methods, which are genuinely different
algorithms and give genuinely different fidelities:

| 4-bit method | identity Jaccard | per-head Spearman | freq. sign kept |
|---|---|---|---|
| bitsandbytes NF4 | 0.838 | 0.940 | yes |
| AWQ | **0.971** | 0.933 | yes |
| GPTQ | 0.857 | 0.934 | yes |

AWQ is the most faithful; GPTQ loses the most frequency-effect magnitude
(about 29%, vs about 9% AWQ and about 3% NF4). The circuit is inherited, not
rebuilt.

---

## Repository layout

```
retrieval-head-profile/
├── datas/
│   ├── results/            <-- ALL results, checked in (89 JSON files)
│   │   ├── profile/        E1-E5 per-model profile (18 models x seeds + 5 quant rings)
│   │   ├── behavior/       E6 NIAH + E7 RULER long-context scores
│   │   ├── utility/        retrieval-head utility (Cohen's d, partial Spearman)
│   │   ├── inheritance/    E10-E15 per-lineage (qwen/llama/gemma/mistral) + E14
│   │   ├── analysis/       E8 prediction + E9 test-retest / reliability
│   │   ├── robustness/     R3 coverage, R4 quant, R6 sample-size
│   │   └── diagnostics/    mistral distributed-retrieval knockout sweep
│   └── README.md           <-- data dictionary: every file and field explained
├── rhp/                    the Retrieval-Head Profile package (new code)
│   ├── copy_score_detector.py   2nd detector (teacher-forced copy score)
│   ├── profile.py               Gini, layer distribution, GQA, detector agreement
│   ├── freq_signature.py        8-window RoPE frequency sweep -> freqEff scalar
│   ├── knockout.py              causal knockout + McNemar
│   ├── ruler.py                 RULER subset (multikey / multivalue / vartrack)
│   ├── inheritance.py           E10-E15 identity / function / frequency / quant
│   └── prediction.py            E8 profile->behaviour, E9 test-retest
├── scripts/                CLI entry points (run_profile / run_behavior /
│                           run_inheritance / run_prediction) + _common.py
├── notebooks/              Colab notebooks, each chunk sized to finish in <24 h
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

Every number in the paper is read straight from `datas/results/`. Nothing is
hand-entered. The naming convention is `<model>_seed<seed>.json`, e.g.
`qwen25_7b_instruct_seed42.json`. A profile record looks like:

```json
{
  "model": "qwen25_7b", "family": "qwen", "seed": 42,
  "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
  "argmax_heads": [[l, h], ...],      // detector 1 (argmax proxy)
  "copy_heads":   [[l, h], ...],      // detector 2 (teacher-forced copy score)
  "profile": {
    "scalars": {                      // flat metrics used by the E8 table
      "gini": 0.95, "layer_com_weighted": 0.77,
      "frequency_effect": -0.92,      // causal low- vs high-freq dose patch
      "knockout_drop": 0.76, ...
    },
    "concentration": {...}, "layer_profile": {...},
    "freq_signature": {...}, "knockout": {...}
  },
  "environment": {...}                // python / packages / hardware / git SHA
}
```

See [datas/README.md](datas/README.md) for the full field-by-field dictionary and
the schema of the behavior / utility / inheritance / analysis records.

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

Nothing is hand-entered anywhere: the paper's figures and tables are all derived
from these JSONs. See [datas/README.md](datas/README.md) for a field-by-field map
of which file backs which result.

---

## Reproduce the results from scratch (needs a 24 GB GPU)

The full pipeline is inference-only and runs one model at a time on a single
24 GB GPU (Colab L4 / A100). Part 2 imports Part 1's proven `src/` modules rather
than forking them, so point it at the Part-1 checkout first:

```bash
pip install -r requirements.txt
export RHP_PART1_REPO=/path/to/Does-RoPE-Prevent-or-Degrade-Retrieval-Heads
export HF_TOKEN=hf_...        # for gated models (Llama, Gemma)
```

Then run the blocks (each per-model result is written atomically and skipped on
re-run, so an interrupted sweep resumes cleanly):

```bash
python scripts/run_profile.py     --models all --time-budget-hours 20   # Block A: E1-E5
python scripts/run_behavior.py    --models all --time-budget-hours 20   # Block B: E6 NIAH + E7 RULER
python scripts/run_inheritance.py --lineage all                         # Block C: E10-E15 (CPU)
python scripts/run_prediction.py  --seed 42 --retest-seed 123           # E8 + E9 (CPU)
```

### Colab

Colab kills a session past about 24 h, so the panel is pre-split into fixed chunks
that each finish in one session and write to Google Drive. Run them in order; the
profile/behaviour chunks are independent and can run in parallel accounts. The
`notebooks/` folder is numbered `00` (pilot) through `14` (final analysis); the
`build_*.py` files regenerate the `.ipynb` from `configs/panel.yaml`. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact run order.

---

## The model panel

**18 independent models across 8 families**, all <= 9 B so each loads alone on a
24 GB GPU. Five are core (multiple seeds + test-retest); the rest are single-seed.

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

**Quantized rings (E14, RQ3):** four 4-bit variants used inside the lineages -
`Qwen2.5-7B-Instruct` in AWQ, GPTQ-Int4, and bitsandbytes-NF4, plus NF4 rings of
`Llama-3.1-8B-Instruct` and `Gemma-2-9b-it`. Inheritance lineages:

| Lineage | Chain |
|---|---|
| Qwen | 7B -> 7B-Instruct -> {NF4, AWQ, GPTQ} 4-bit; sibling 3B-Instruct |
| Llama | 8B -> 8B-Instruct -> NF4 4-bit |
| Gemma | 9B -> 9b-it -> NF4 4-bit; sibling 2B |
| Mistral | 7B-v0.3 -> 7B-Instruct-v0.3 |

---

## The retrieval-head profile (method in one table)

For each model we extract a standardised vector. The key new ingredient over
Part 1 is the **second detector** (a teacher-forced copy score, close to Wu et
al. 2025) run alongside Part-1's argmax proxy, so we validate the claim rather
than a single metric.

| Metric | Meaning | Experiment |
|---|---|---|
| `n_heads`, `n_heads_copy` | heads found by the two detectors | E1 |
| `detector_agreement` | Jaccard(argmax-set, copy-set) | E1 |
| `gini`, `zero_fraction` | score concentration (Qwen-type vs OLMo-type) | E4 |
| `layer_com` | layer-depth centre of mass | E4 |
| `freq_com`, `frequency_effect` | RoPE spectral centre + causal low- vs high-freq dose patch | E2 / E12 |
| `knockout_drop`, McNemar | causal mask effect vs random heads | E3 |
| `utility` (Cohen's d, partial Spearman) | how much recall depends on the heads | E8 |
| `R_self` (copy-Jaccard test-retest) | reliability ceiling for the inheritance test | E9 |

Inheritance decision rules are pre-registered in
[docs/PREREGISTRATION_inheritance.md](docs/PREREGISTRATION_inheritance.md):
identity is "inherited" when copy-Jaccard exceeds `0.8 * R_self`.

---

## Statistics

- Benjamini-Hochberg FDR across the panel correlations (E8).
- LOO cross-validation, <= 3 predictors, for the prediction regression.
- **Family-demeaned** correlations, because family is a confound (this is exactly
  what turns the raw RQ2 correlation into a confound-limited null).
- Exact McNemar + bootstrap CI for paired conditions (knockout, frequency patch).
- Layer-clustered permutation for head-level comparisons (pseudoreplication guard).
- Core results at 3 seeds; the rest single-seed, stated in every table.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | full E / C / R / O experiment inventory mapped to code + notebooks |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | exact reproduction steps and run order |
| [docs/PREREGISTRATION_inheritance.md](docs/PREREGISTRATION_inheritance.md) | pre-registered inheritance decision rules |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | threats to validity |
| [datas/README.md](datas/README.md) | data dictionary for `datas/results/` |

---

## License

Code is released under the **MIT License** (see [LICENSE](LICENSE)). Models and
datasets keep their own licenses: Llama-2/3.x and Gemma are gated with use
restrictions; Qwen2.5, OLMo-2, Mistral, Falcon, StableLM are Apache-2.0/permissive;
PG-19 per its dataset card. Cite each in any publication.

## Citation

```bibtex
@article{bayram2026retrievalprofile,
  title   = {Retrieval Heads Survive Deployment: A Training-Free Mechanistic
             Profile and Its Inheritance Through Instruction Tuning and
             4-Bit Quantization},
  author  = {Bayram, Cengizhan},
  year    = {2026}
}
```

Part 1: *Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis
Across Model Families* (arXiv:2606.21249).
