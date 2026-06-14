# Limitations & threats to validity

Stated plainly so the paper can address each. Many carry over from Part 1; the
new ones concern the profile and the inheritance claim.

## Measurement

- **L1 — Detector operationalisation.** The argmax detector is a *single-pass
  proxy* of Wu et al. (2025); the copy-score detector is a *teacher-forced*
  variant, not their full generation-time score. We report **both** and their
  agreement (`detector_agreement`); where they disagree, the copy score is
  primary (R2). The agreement itself is a profile metric, not a nuisance.
- **L2 — Threshold choice.** "Retrieval head" depends on τ. All panel-level
  conclusions are checked over τ∈[0.05, 0.3] (R1); we report sign/ordering
  preservation, not a single τ.
- **L3 — Frequency convention.** The per-dimension RoPE frequency uses the
  HuggingFace `rotate_half` (NeoX) convention (dim *j* and *j+head_dim/2* share a
  frequency). A wrong convention scrambles "high vs low frequency"; the inherited
  `dimension_utility` fixes it.
- **L4 — Quantization perturbs attention.** 8-bit (and 4-bit) change attention
  distributions and therefore detection. R4 (fp16 vs 8-bit) and E14 (the 3-level
  AWQ/GPTQ template) separate a real shift from a measurement artifact.

## Statistical power

- **L5 — Panel size.** Even at ~24 models, the RQ2 regression has few points.
  Claims stay at the correlation + CI level (BH-corrected); LOO guards
  over-fitting; **no taxonomy claim** is made. The expanded panel (vs the
  proposal's ~18) is the mitigation; further expansion is left open.
- **L6 — Family confound.** Family drives both profile and behaviour. We report
  family-demeaned (within-family) correlations alongside the raw ones (E8 iii).
- **L7 — Reliability ceiling.** A metric cannot predict behaviour better than it
  correlates with its own replication; E9 test-retest bounds every E8 claim.

## Inheritance claim

- **L8 — Confounded transformations.** base→instruct bundles SFT + data + recipe
  changes; we attribute to the *transformation*, not a single cause. The
  behaviour bridge (E13) and localisation (E15) localise *where* the circuit
  moved, not *why*.
- **L9 — Architecture-matched only for per-head stats.** Per-head score Spearman
  (E10) requires identical architecture, so the distillation **sibling**
  comparison (different shape) uses aggregate identity only.
- **L10 — Cross-tokenizer pairing.** Behavioural scoring across models uses a
  shared seed + corpus; strict identical-sample pairing is available via the
  inherited `paired_spec_subset` (intersection-drop logged, C6). Large drops
  (>20 %) trigger pool expansion.

## Scope

- **L11 — NIAH/RULER are synthetic.** The behavioural targets are retrieval-style
  tasks; the profile is *not* claimed to predict reasoning or summarisation.
- **L12 — Single-GPU context ceiling.** Profiles are extracted at ≤4096 tokens on
  24 GB (≤16384 for ≤3B). Long-context behaviour beyond that is observed only on
  smaller models; the A100 path (optional) extends it.
- **L13 — Negative RQ2 is a result, not a failure.** If the profile does not
  track behaviour, that bounds how much NIAH/RULER scores reveal about the
  mechanism — reported as such, with CIs.
