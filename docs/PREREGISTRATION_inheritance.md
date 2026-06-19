# Pre-registration — circuit inheritance (E10–E15), the paper's spine

**Purpose.** The retrieval-circuit-inheritance claim is the new paper's thesis.
To keep it honest — and to answer the TMLR "claims not evaluable / argumentation
immature" critique at the root — the decision rules below are fixed **before any
real lineage is run.** Whatever the numbers come out to, we read them against
*these* thresholds, not a story invented after the fact. This mirrors v1's
pre-registered 0.33 specificity rule, applied now to the paper's core.

**Status:** E10–E15 are UNRUN as of writing — code validated on synthetic data
only. No lineage ring (base→instruct, etc.) exists yet. So there is no finding
yet; this document fixes how we will read the first one.

---

## The denominator problem (must run E9 FIRST)

A base→child head-set Jaccard cannot be read against an absolute 1.0, because the
detector is itself noisy (v1: even two detectors on the *same* model agreed at
Jaccard ≈ 0.17–0.30). The right reference is the model's **own** test-retest
reliability — how similar the model is to itself across two independent sample
sets (E9, `rhp.prediction.within_model_reliability`).

> **Run order (fixed):**
> 1. **E9 reliability** on the base model (profile it at a 2nd seed; compute
>    `within_model_reliability`). This is the denominator `R_self`.
> 2. Register the thresholds below relative to `R_self`.
> 3. **Then** run the child ring(s) and read identity against `R_self`.

---

## Decision rules (registered before the numbers)

For each parent→child ring, on the **copy detector** (the trusted one, R2):

| Axis | Statistic | Rule |
|---|---|---|
| **Identity (E10)** | head-set Jaccard `J` vs self-reliability `R_self` | `J ≥ 0.8·R_self` → **identity inherited**; `J < 0.5·R_self` → **identity reconstructed**; between → partial |
| **Function (E11)** | `knockout_drop_child` vs `knockout_drop_base` | `≥ 0.5×` base and ≫ random → **function inherited** |
| **Frequency (E12)** | `frequency_effect` sign + magnitude | same sign and `|effect| ≥ 0.5×` base → **frequency inherited** |
| **Utility (M7)** | `cohens_d` sign | sign preserved → **utility-signature inherited** |
| **Localisation (E15)** | only if child recall (`niah_long`) dropped > 0.02 | identity-loss if `J < 0.8·R_self`; weakening if `J ≥ 0.8·R_self` but per-head score Spearman drops |

A result is **"inheritance"** only if identity *and* function are inherited.
"Functional inheritance without identity inheritance" (identity reconstructed but
function + frequency inherited) is a **distinct, registered outcome** — and an
interesting one, not a fallback story.

---

## Priors (registered before running — the researcher's, written first)

Filling these in *before* the run is what makes the result a test, not a
post-hoc narrative. Each prior is a falsifiable bet. **These are the
researcher's to write** — a collaborator's guess would only bias the reading, so
none is recorded here.

- Identity (E10): _______ (expect Jaccard relative to R_self: high / ≈R_self / low?)
- Function (E11): _______ (masking still breaks the child? yes / no)
- Frequency (E12): _______ (−0.69-type dependence preserved? yes / attenuated / no)
- Utility (M7): _______
- If the axes diverge: _______ (how do you interpret identity-churn + function-conserved?)

---

## First ring to run (cheapest real datum)

`qwen25_7b` (already profiled) → `qwen25_7b_instruct` (one more profile+behaviour
run) gives the first `base→instruct` ring. The Qwen lineage is ideal: one chain
carries **three** transformations — instruct (base→instruct), quantization
(instruct→AWQ→GPTQ), distillation (3B-instruct sibling). After this one ring we
either can or cannot write "circuit inheritance found: ____" — and we will know
which, honestly.
