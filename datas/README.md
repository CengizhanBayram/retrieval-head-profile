# `datas/results/` - data dictionary

This folder holds **all** results for the study (89 JSON files). Every number in
the paper is read straight from here; nothing is hand-entered. This page explains
what each subfolder contains and what each field means.

**Naming:** files are `<model>_seed<seed>.json`, e.g.
`qwen25_7b_instruct_seed42.json`. Models with a quantization suffix
(`_bnb4`, `_awq4`, `_gptq4`) are 4-bit rings used only in the inheritance
experiments; all other files are the 18 independent panel models.

```
results/
├── profile/        E1-E5   per-model retrieval-head profile
├── behavior/       E6-E7   NIAH + RULER long-context scores
├── utility/        E8      how much recall depends on the retrieval heads
├── inheritance/    E10-E15 per-lineage circuit inheritance (incl. E14 quant)
├── analysis/       E8-E9   cross-panel prediction + test-retest reliability
├── robustness/     R3/R4/R6 coverage, quantization, sample-size checks
└── diagnostics/            mistral distributed-retrieval knockout sweep
```

---

## `profile/` - the retrieval-head profile (E1-E5)

One file per model+seed. Top-level keys:

| Field | Meaning |
|---|---|
| `model`, `family`, `seed` | identity |
| `n_layers`, `n_heads`, `n_kv_heads` | architecture (GQA if `n_kv_heads < n_heads`) |
| `argmax_heads` | `[[layer, head], ...]` found by detector 1 (argmax proxy, from Part 1) |
| `copy_heads` | `[[layer, head], ...]` found by detector 2 (teacher-forced copy score, new) |
| `profile` | the metric bundle (below) |
| `environment` | python / package versions / hardware / git SHA (from Part 1's capture) |

`profile.scalars` is the flat vector consumed by the E8 prediction table:

| Scalar | Meaning |
|---|---|
| `gini`, `zero_fraction` | score concentration (Qwen-type concentrated vs OLMo-type spread) |
| `layer_com_weighted` | layer-depth centre of mass of the retrieval heads |
| `frequency_effect` (`freqEff`) | **causal** low- vs high-frequency RoPE dose-patch effect; negative = low-frequency dependent |
| `knockout_drop` | NIAH recall drop when the top retrieval heads are masked |

Nested blocks give the full detail behind those scalars: `concentration`,
`layer_profile`, `gqa`, `freq_signature` (the 8-window sweep), `freq_patch`,
`knockout` (with the McNemar test).

---

## `behavior/` - long-context behaviour (E6 NIAH + E7 RULER)

`behavior` sub-keys:

| Field | Meaning |
|---|---|
| `niah_matrix` | needle-in-a-haystack recall over `context_lengths` x `needle_positions` |
| `niah_overall`, `niah_long`, `niah_worst_pos` | summary recall (mean, long-context slice, worst position) |
| `niah_maxlen` | the longest context at which the model still passes NIAH (a **categorical** proxy for the model's context window) |
| `niah_per_context`, `niah_per_position` | per-length / per-position breakdowns |
| `ruler` | RULER subset: `task_means` for `multikey`, `multivalue`, `vartrack` (variable-tracking is the **continuous** target used in RQ2) |

---

## `utility/` - retrieval-head utility (E8)

How much a model's recall actually depends on its retrieval heads:

| Field | Meaning |
|---|---|
| `cohens_d` | effect size, retrieval-head vs non-retrieval-head recall contribution |
| `partial_spearman`, `partial_p` | utility partialling out head count (the RQ2 predictor) |
| `clustered_permutation_p` | layer-clustered permutation p-value (pseudoreplication guard) |
| `retrieval_mean`, `non_retrieval_mean`, `n_retrieval`, `n_non_retrieval` | the underlying group means and sizes |
| `hypothesis_supported` | whether utility is significantly positive |

---

## `inheritance/` - circuit inheritance (E10-E15)

One file per lineage: `qwen.json`, `llama.json`, `gemma.json`, `mistral.json`.

- `rings`: a list of parent -> child transitions (e.g. base -> instruct,
  instruct -> 4-bit). Each ring has `E10_identity` (copy-Jaccard),
  `E11_function` (knockout transfer), `E12_frequency` (freqEff comparison),
  `M7_utility`, `E13_bridge`, `E15_localization`, and `same_architecture`.
- `E14_quant_ablation`: the cross-method quantization comparison on
  Qwen-2.5-7B-Instruct. Keys `reference`, `bnb4`, `awq4`, `gptq4`; each method
  reports:

  | Field | Meaning |
  |---|---|
  | `identity_jaccard` | copy-head overlap with the fp16 instruct reference |
  | `per_head_score_spearman` | rank correlation of per-head copy scores vs fp16 |
  | `ref_frequency_effect`, `quant_frequency_effect` | freqEff before/after quantization |
  | `finding_sign_preserved` | whether the RoPE frequency sign is kept |

- `sibling`: the distillation sibling comparison (e.g. Qwen 7B vs 3B).

---

## `analysis/` - cross-panel analysis (E8 + E9)

| File | Contents |
|---|---|
| `prediction_e8.json` | **canonical** RQ2 analysis over the 18 independent models: `single_correlations` (each predictor->target Spearman with BH-adjusted p and n), `family_demeaned` (the confound-controlled version), `loo_regression`, `loo_top3_predictors` |
| `prediction_e8_withrings.json` | the same analysis but including the quantized rings (n up to 22); kept for comparison, **not** the headline (rings are not independent training runs) |
| `reliability_e9.json` | `R_self` = test-retest copy-Jaccard per model (the reliability ceiling for the inheritance threshold `0.8 * R_self`) |
| `test_retest_e9.json` | the raw seed-a vs seed-b comparison behind `reliability_e9` |

> RQ2 headline numbers come from `prediction_e8.json`: for
> `utility_partial_spearman`, the correlation with the categorical `niah_maxlen`
> is strong (rho = -0.85, n = 17) but the continuous `ruler_vartrack` correlation
> is not significant after BH correction - the confound-limited null.

---

## `robustness/` - robustness checks (R-series)

| Subfolder | Check |
|---|---|
| `R3_coverage/` | detector coverage sweep (fraction of heads kept: 0.3 / 0.5 / 1.0) |
| `R4_quant/` | int8-vs-fp16 detector-set stability (`jaccard_int8_fp16`); e.g. Qwen 0.982 |
| `R6_nsamples/` | 100-vs-200 sample stability (`jaccard_100_200`); e.g. OLMo 0.948 |
| `R7_haystack/` | reserved (haystack-source check; not populated) |

---

## `diagnostics/`

`mistral_knockout_sweep.json` - the per-k knockout curve that shows Mistral's
retrieval is **distributed**: ablating even all detected heads barely reduces
NIAH recall, unlike every other family.
