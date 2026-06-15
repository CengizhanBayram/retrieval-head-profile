# Experiment inventory → code & notebook map

Every experiment in the proposal (§4) mapped to the function that implements it,
the notebook task that runs it, and the output it writes. This is the
ground-truth checklist for the paper: if a row has code + notebook + output, it
is "done properly."

Legend — **Code**: module/function in `rhp/` or the inherited `src/`.
**NB**: Colab notebook task. **Out**: file under `results/`.

---

## Block A — Profile extraction (RQ1)

| Exp | What | Code | NB | Out |
|---|---|---|---|---|
| **E1** | Dual-detector retrieval-head detection (argmax proxy + teacher-forced copy score); threshold sweep τ∈{.05,.1,.2,.3}; detector Jaccard | `src.RetrievalHeadDetector` + `rhp.copy_score_detector.CopyScoreDetector` → `rhp.profile.detector_agreement` | 00, 01 | `profile/<m>.json` → `profile.detector_agreement`, `argmax_scores`, `copy_scores` |
| **E2** | Frequency-signature sweep: 8 windows of `head_dim/8` dims across the spectrum, population-patch at 50 % coverage, NIAH drop curve | `rhp.freq_signature.frequency_signature` | 00, 01 | `profile.freq_signature` (`drop`, `freq_com`, `freq_width`) |
| **E3** | Knockout double dissociation: mask retrieval heads vs equal random non-retrieval heads; McNemar | `rhp.knockout.KnockoutEvaluator.run` | 00, 01 | `profile.knockout` (`knockout_drop`, `random_drop`, `mcnemar`) |
| **E4** | Concentration (Gini, zero-fraction) + layer-depth distribution | `rhp.profile.concentration`, `rhp.profile.layer_profile` | 01 (CPU-derivable) | `profile.concentration`, `profile.layer_profile` |
| **E5** | GQA group-distribution control (KV-sharing not an artifact) | `rhp.profile.gqa_group_distribution` | 01 | `profile.gqa` |

## Block B — Behavioural targets & prediction (RQ2)

| Exp | What | Code | NB | Out |
|---|---|---|---|---|
| **E6** | **Long-context** NIAH sweep (4k→32k, per-context sample schedule), "lost-in-the-middle" depth. Short NIAH saturates (pilot: every model 1.0), so `niah_long` (≥16k) is the RQ2 target with variance | `src.NIAHEvaluator.evaluate` per context via `scripts._common.run_behavior_for_model` | 00, 02 | `behavior/<m>.json` → `niah_matrix`, `niah_per_context`, **`niah_long`**, `niah_overall`, `niah_worst_pos` |
| **E7** | RULER subset: multi-key, multi-value, variable tracking | `rhp.ruler.RulerEvaluator.run` | 00, 02 | `behavior.ruler` |
| **E8** | Profile → behaviour prediction: BH-corrected Spearman, LOO ≤3-predictor regression, family-demeaned. Targets = `niah_long`, `ruler_multivalue`, `ruler_vartrack` (the ones with variance), not saturated `niah_overall` | `rhp.prediction.run_prediction_analysis` | 04 (CPU) | `analysis/prediction_e8.json` |
| **E9** | Profile test-retest (reliability ceiling for E8) | `rhp.prediction.test_retest` | 04 (CPU) | `analysis/test_retest_e9.json` |

## Block C — Inheritance chains (RQ3) + RQ4

| Exp | What | Code | NB | Out |
|---|---|---|---|---|
| **E10** | Identity inheritance: head-set Jaccard (both detectors) + per-head score Spearman | `rhp.inheritance.compare_identity` | 03 (CPU) | `inheritance/<lineage>.json` → `rings[].E10_identity` |
| **E11** | Function inheritance: knockout drop across the ring | `rhp.inheritance.compare_function` | 03 | `rings[].E11_function` |
| **E12** | Frequency-dependence inheritance: spectral signature + single-dose `frequency_effect` (low−high), sign preservation | `rhp.inheritance.compare_frequency` (data from `profile.freq_signature` + `profile.freq_patch`) | 03 | `rings[].E12_frequency` |
| **E13** | Behaviour bridge: ΔNIAH / ΔRULER vs Δprofile | `rhp.inheritance.behavior_bridge` | 03 | `rings[].E13_bridge` |
| **E14** | Quantization ablation: instruct vs AWQ-4bit & GPTQ-4bit (3-level template) | `rhp.inheritance.quant_ablation` | 03 | `inheritance/qwen.json` → `E14_quant_ablation` |
| **E15** | RQ4 localisation: identity-loss vs weakening when recall drops | `rhp.inheritance.localize_recall_change` | 03 | `rings[].E15_localization` |

## Block D — Controls (validity conditions)

| Ctrl | What | Where |
|---|---|---|
| **C1** | Random-head control (equal-size random non-retrieval set) | embedded in `rhp.knockout.run` (`mask_random`) |
| **C2** | Layer-matched non-retrieval control on every frequency patch | `rhp.freq_signature` (`recall_control`) + `_common` population patch (`control_heads`) |
| **C3** | Random-dimension control (equal count) on every window | `rhp.freq_signature` (`recall_randdim`) |
| **C4** | Perplexity / task-specificity control, pre-registered 0.33 ratio rule | inherited `ActivationPatcher.run_population_patching` (`specificity_verdict`) → `profile.freq_patch` |
| **C5** | Position control (worst-position vs averaged target reported separately) | `behavior.niah_worst_pos` + `niah_per_position`; used in `prediction` |
| **C6** | Intersection-drop audit for paired scoring | inherited `paired_spec_subset` (logs drop %); shared seed + corpus keep samples comparable |

## Block E — Robustness

| Rob | What | Code / NB |
|---|---|---|
| **R1** | Threshold robustness τ∈[.05,.3] — re-derive heads from saved scores | NB 05 (CPU, no model reload) |
| **R2** | Detector robustness — every E8/E10–E12 result computed with both detectors; copy score is the tiebreak | both `argmax_*` and `copy_*` stored in every profile |
| **R3** | Coverage robustness — frequency signature at coverage∈{.3,.5,1.0} | NB 05 (`robustness/R3_coverage`) |
| **R4** | Quantization robustness — fp16 vs 8-bit profile (closes Part-1 future work) | NB 05 (`robustness/R4_quant`) |
| **R5** | Seed robustness — core models + differing rings at 3 seeds | `--seed`/`SEED` loop; rest single-seed, tabled explicitly |
| **R6** | Sample-size sensitivity — profile at n=100 vs 200 | NB 05 (`robustness/R6_nsamples`) |
| **R7** | Haystack-source robustness — alternative neutral corpus (WikiText) | NB 05 (`robustness/R7_haystack`) |

## Block F — Optional / conditional

| Opt | What | Status |
|---|---|---|
| **O1** | Surgical band-based extension (uniform vs NTK/YaRN on the retrieval band) | hook documented in NB 05; single-model add-on |
| **O2** | OLMo-2 checkpoint inheritance (frequency signature pre/post crystallisation) | point `run_profile_for_model` at OLMo-2 checkpoints; compare with `compare_frequency` |
| **O3** | Merged-model profile (union vs intersection of parents' head sets) | add the merge HF id to `panel.yaml`; `compare_identity` vs both parents |
| **O4** | Long-context fine-tune variant (base vs long-context-tuned profile) | add the variant to `panel.yaml`; standard profile diff |
| **O5** | Attention-mass detector (3rd metric, sign-agreement) | re-run E1 with `score_heads(return_mass=True)`; store `mass` matrix; CPU reanalysis in NB 05 |

---

## Experiment → research-question matrix

| Exp | RQ1 | RQ2 | RQ3 | RQ4 |
|---|---|---|---|---|
| E1–E5 | ✔ | input | input | — |
| E6–E9 | — | ✔ | input | — |
| E10–E14 | — | — | ✔ | input |
| E15 | — | — | — | ✔ |
| C1–C6 | validity condition for all | | | |
| R1–R7 | robustness condition for all | | | |
| O1–O5 | extensions | | | |

## Notes on faithfulness

- The copy-score detector teacher-forces the gold answer and checks attention on
  the **exact source token by position** (not by value), which removes the
  repeated-token ambiguity of a naive copy score.
- E12's `frequency_effect` and the C4 specificity verdict come from the **same**
  inherited `run_population_patching` used in Part 1, so the two papers report
  the frequency test identically.
- E2 (8-window spectral curve) and E12 (single low/high-freq dose) are
  **different** experiments and both are stored; E2 gives the *shape*, E12 the
  *causal magnitude + specificity*.
- **4-bit rings.** The proposal §3.4 chains include a 4-bit ring for every
  lineage. Qwen uses the official pre-quantized **AWQ** and **GPTQ-Int4** repos
  (which also drive the dedicated E14 AWQ-vs-GPTQ ablation). Llama and Gemma have
  no equally-standard pre-quantized repo, so their 4-bit ring is **bitsandbytes
  NF4** load-time quantization of the instruct weights (`quant: bnb4` →
  `rhp.loader.load_model_any`); these flow through the regular E10–E13 ring
  comparison. All 4-bit rings carry the Part-1 quantization caveat (R4 / L4): a
  4-bit profile shift is cross-checked against the fp16/8-bit reference before it
  is read as a real change rather than a measurement artifact.
