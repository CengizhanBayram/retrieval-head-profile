w# v2 integration plan — folding the RoPE-paper (A-series) depth into the Profile paper

**Goal.** The current Part-2 paper answers *what the profile is* (RQ1), *whether
it predicts behaviour* (RQ2), and *whether the circuit is inherited* (RQ3). It is
strong but **descriptive on the mechanism side** — it reports a frequency
signature but does not, within this paper, prove the frequency dependence is
*causal, distance-resolved, benchmark-transferable, and retrieval-specific*. The
Part-1 RoPE paper's **A-series** experiments establish exactly that depth. Folding
them in turns the Profile paper into a single, self-contained, mechanistically
grounded study — the "stronger v2."

This document maps every A-series experiment to a Part-2 experiment, marks what we
**already built** vs what is **new code**, and lists the to-dos in build order.

---

## New research question this unlocks

> **RQ5 (Mechanistic grounding).** Is the retrieval-head profile *mechanistically
> real* — i.e. (a) causally tied to low-frequency RoPE dimensions within a model
> (θ control), (b) sharpening with needle distance, (c) transferable to a real
> long-context benchmark, and (d) specific to retrieval rather than induction?

RQ5 sits under the profile (RQ1) and makes RQ2/RQ3 far harder to dismiss: a
profile that is causal, transferable, and retrieval-specific is a real diagnostic,
not a correlational artefact.

---

## A-series → Part-2 mapping (Block M — Mechanistic grounding)

| New | From | Experiment | Status in our repo | Effort |
|----|------|-----------|--------------------|--------|
| **M1** | A1 | **Within-model θ control** — scale `rope_theta` ×{0.5,1,2}, re-measure the frequency effect + perplexity. Causal: if low-freq dims carry retrieval, rescaling θ moves the effect. | **NEW** (θ-rescaling loader hook + sweep) | medium |
| **M2** | A5 | **Distance-resolved low-freq dependence** — the low-freq population patch effect vs needle position/distance, fixed head set. | **NEW** (distance loop over the population patch) | low–med |
| **M3** | A6 | **Frequency-band importance curve** — sliding-window band patch. | **HAVE** = E2 `rhp/freq_signature.py` (8 windows). Optional: bump to ~15 bands for the figure. | done |
| **M4** | A3 | **Copy-score-defined frequency effect** — run the freq patch on the *copy-detected* head set (settles magnitude with the stricter detector). | **HAVE** detector (`rhp/copy_score_detector.py`); **NEW** = a flag to drive the freq patch from copy heads. | low |
| **M5** | A2 | **Benchmark transfer** — detect heads on NIAH, then **mask those heads on RULER** and **low-freq-patch on RULER**. External-validity of the *causal* claim. | **HAVE** RULER tasks (`rhp/ruler.py`), knockout, patcher; **NEW** = a transfer evaluator that wires them on RULER samples. | medium |
| **M6** | A7b | **Induction-vs-retrieval specificity** — detect induction heads (repeated-token / prev-token attention), show the low-freq dependence is retrieval-specific, not a generic induction property. | **NEW** (induction-head detector + overlap/specificity test) | medium |
| **M7** | A7a+A8 | **Quantization ablation + dimension-utility signature** — fp16-vs-8-bit detector (have, R4) **+** the Chiang–Yogatama utility d-test (Cohen's d, sign, layer-clustered p) added to every profile. | fp16 = **HAVE** (R4); utility d-test = **NEW but tiny** (inherited `DimensionUtilityAnalyzer.compute_utility_scores`). | low |

**Already covered, no new work:** M3 (=E2), the fp16 half of M7 (=R4), the copy
*detector* of M4 (=E1).

**Net new code:** M1 (θ hook), M2 (distance loop), M4 flag, M5 (transfer
evaluator), M6 (induction detector), M7 utility hook.

---

## Detailed specs

### M7 — dimension-utility signature (do first; tiny, unblocks A8)
- **Method.** In `run_profile_for_model` we already build `analyzer` and `norms`
  (for E12). Call `analyzer.compute_utility_scores(norms, argmax_heads)` →
  `cohens_d`, `clustered_permutation_p`, retrieval/non-retrieval means; and
  `analyzer.compute_retrieval_utility_correlation(argmax_scores, norms)` →
  layer-partial Spearman. Store under `profile.utility`, add scalars
  `utility_cohens_d`, `utility_perm_p`, `utility_partial_spearman`.
- **Why.** Gives every panel model the *Part-1 Layer-A signature* for free, so the
  profile carries both the behavioural (retrieval-head) and the weight-space
  (utility) view. Adds three strong RQ2 predictors and reproduces A8 on 18 models.
- **Output.** `profile.utility.{cohens_d, perm_p, partial_spearman, retrieval_mean, non_retrieval_mean}`.

### M5 — benchmark transfer (highest external-validity payoff)
- **Method.** New `rhp/transfer.py::RulerTransferEvaluator`. Build RULER samples as
  `{prompt_ids, code}` (reuse the `rhp/ruler.py` task generators), then score
  exact-match accuracy under three conditions on the SAME samples:
  (1) baseline, (2) retrieval-heads masked (o_proj pre-hook, KnockoutEvaluator
  mechanism), (3) low-frequency dims zeroed in the retrieval heads (ActivationPatcher
  `patch_heads` with `freq_order[:k]`). Paired McNemar (masked vs baseline,
  low-freq vs baseline).
- **Claim.** "The same heads and the same low-frequency dependence that drive NIAH
  also drive a real long-context benchmark" — closes the single-synthetic-task gap.
- **Output.** `transfer/<model>.json` → per task: baseline / masked / lowfreq +
  McNemar.

### M1 — within-model θ control
- **Method.** Load the model, then for `mult ∈ {0.5,1,2}` rewrite the rotary base:
  set `model.config.rope_theta *= mult` and rebuild the rotary embedding (or patch
  the `inv_freq` buffer on every `self_attn.rotary_emb`). Re-run the population
  freq patch + a plain-text perplexity control at each θ. The frequency effect
  should track θ if it is genuinely low-frequency-mediated.
- **New code.** `rhp/theta.py::with_scaled_theta(model, mult)` context manager that
  patches and restores the `inv_freq` buffers; a sweep in `_common`.

### M2 — distance-resolved dependence
- **Method.** Fix the retrieval-head set; run the low-freq population patch at each
  needle position {0.1,…,0.9} at a long context; report the effect-vs-distance
  curve. Reuses `ActivationPatcher.run_population_patching` with single-position
  sample sets.

### M4 — copy-defined frequency effect
- **Method.** Add `head_source="copy"` to the freq-patch step so the patched
  population is the copy detector's heads. One extra population patch per model.

### M6 — induction vs retrieval
- **Method.** New `rhp/induction.py`: induction-head score = attention from the
  second occurrence of a repeated random token back to the token *after* its first
  occurrence (standard induction metric), on a repeated-random-tokens prompt.
  Report Jaccard(retrieval heads, induction heads) and whether the low-freq patch
  spares induction-defined heads (specificity).

---

## To-do (build order)

- [ ] **M7** utility d-test into the profile  *(tiny; do first)*
- [ ] **M5** `rhp/transfer.py` RULER transfer evaluator + `run_transfer.py` + notebook task
- [ ] **M4** copy-head source flag on the freq patch
- [ ] **M2** distance-resolved patch sweep + script/notebook
- [ ] **M1** `rhp/theta.py` θ-rescaling + sweep + script/notebook
- [ ] **M6** `rhp/induction.py` induction detector + specificity test
- [ ] update `configs/panel.yaml` (`mechanism:` block: θ mults, bands, distance positions)
- [ ] new notebook `06_mechanism_colab.ipynb` (chunked, 23 h-guarded) running M1/M2/M4/M5/M6
- [ ] docs: add Block M to `EXPERIMENTS.md`, RQ5 to `README.md`, threats to `LIMITATIONS.md`
- [ ] figures: θ-curve, distance-curve, band-curve, transfer bar, induction-overlap

All M-experiments are inference-only, single-GPU, and inherit the 23 h adaptive
guard + Drive resume. Each is independent and chunkable, like the existing blocks.

---

## Paper structure (v2, stronger)

1. Profile (RQ1) — E1–E5.
2. **Mechanistic grounding (RQ5) — M1–M7** ← new spine.
3. Prediction (RQ2) — E6–E9, now with utility predictors (M7) and the transfer
   target (M5).
4. Inheritance (RQ3) — E10–E15.
5. Controls/robustness — C/R + fp16 (M7) + induction (M6).

This makes the profile a *causally validated, benchmark-transferable, retrieval-
specific* diagnostic — the difference between a descriptive and a definitive paper.
