"""
Inheritance package — circuit survival across transformations (Block C, E10–E13).

Pure analysis over two already-computed per-model results (parent → child ring
of a lineage). The standard inheritance package the proposal applies to every
adjacent ring-pair:

    E10  identity   Jaccard of the retrieval-head sets (both detectors) +
                    per-head score Spearman (when architectures match) or a sign
                    test on the shared heads.
    E11  function   compare the E3 knockout drop across the ring.
    E12  frequency  compare the E2 frequency signature (centre of mass, width)
                    and the population frequency-patch effect; flag direction.
    E13  bridge     pair the behavioural delta (NIAH / RULER) with the profile
                    delta, attributing which component each transformation moved.

The three-level quantization template from Part 1 (identity Jaccard / score
correlation / finding sign) is applied unchanged on the 4-bit rings.
"""

from __future__ import annotations

import logging

import numpy as np

from src.stats_utils import jaccard

logger = logging.getLogger(__name__)


def _heads(result: dict, detector: str) -> list[tuple[int, int]]:
    key = "argmax_heads" if detector == "argmax" else "copy_heads"
    return [tuple(h) for h in result.get(key, [])]


def _scores(result: dict, detector: str) -> np.ndarray | None:
    key = "argmax_scores" if detector == "argmax" else "copy_scores"
    s = result.get(key)
    return np.asarray(s, dtype=np.float64) if s is not None else None


# ---------------------------------------------------------------------------
# E10 — identity inheritance
# ---------------------------------------------------------------------------

def compare_identity(parent: dict, child: dict) -> dict:
    """Jaccard of head sets (both detectors) + per-head score Spearman (E10)."""
    out: dict = {}
    for det in ("argmax", "copy"):
        j = jaccard(_heads(parent, det), _heads(child, det))
        sp, ps = _scores(parent, det), _scores(child, det)
        score_spearman = float("nan")
        if sp is not None and ps is not None and sp.shape == ps.shape:
            from scipy import stats
            a, b = sp.flatten(), ps.flatten()
            if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
                score_spearman = float(stats.spearmanr(a, b)[0])
        out[det] = {
            "jaccard": j["jaccard"],
            "intersection": j["intersection"],
            "union": j["union"],
            "n_parent": j["n_a"],
            "n_child": j["n_b"],
            "lost": j["only_a"],      # heads present in parent, gone in child
            "gained": j["only_b"],    # new heads in child
            "per_head_score_spearman": score_spearman,
        }
    return out


# ---------------------------------------------------------------------------
# E11 — function inheritance
# ---------------------------------------------------------------------------

def compare_function(parent: dict, child: dict) -> dict:
    """Compare the E3 knockout drop across the ring (E11)."""
    pk = (parent.get("profile", {}).get("knockout") or {})
    ck = (child.get("profile", {}).get("knockout") or {})
    return {
        "parent_knockout_drop": pk.get("knockout_drop", float("nan")),
        "child_knockout_drop": ck.get("knockout_drop", float("nan")),
        "delta_knockout_drop": (ck.get("knockout_drop", float("nan"))
                                - pk.get("knockout_drop", float("nan"))),
        "parent_dissociation": pk.get("dissociation", float("nan")),
        "child_dissociation": ck.get("dissociation", float("nan")),
    }


# ---------------------------------------------------------------------------
# E12 — frequency-dependence inheritance
# ---------------------------------------------------------------------------

def compare_frequency(parent: dict, child: dict) -> dict:
    """
    Compare the frequency dependence across the ring (E12).

    Two complementary axes:
      • the E2 *spectral signature* (centre of mass / width / peak drop), and
      • the E12 single-dose *frequency effect* (low_freq − high_freq from the
        50%-coverage population patch) with its perplexity specificity verdict
        (C4). A retrieval circuit that survives the transformation should keep
        both the same sign and a comparable magnitude.
    """
    pf = (parent.get("profile", {}).get("freq_signature") or {})
    cf = (child.get("profile", {}).get("freq_signature") or {})
    pp = (parent.get("profile", {}).get("freq_patch") or {})
    cp = (child.get("profile", {}).get("freq_patch") or {})

    def _peak_drop(f):
        d = f.get("drop")
        return float(np.max(d)) if d else float("nan")

    p_fe = pp.get("frequency_effect", float("nan"))
    c_fe = cp.get("frequency_effect", float("nan"))
    sign_preserved = None
    if p_fe == p_fe and c_fe == c_fe:
        sign_preserved = bool(np.sign(p_fe) == np.sign(c_fe))

    return {
        # E2 spectral signature
        "parent_freq_com": pf.get("freq_com", float("nan")),
        "child_freq_com": cf.get("freq_com", float("nan")),
        "delta_freq_com": cf.get("freq_com", float("nan")) - pf.get("freq_com", float("nan")),
        "parent_freq_width": pf.get("freq_width", float("nan")),
        "child_freq_width": cf.get("freq_width", float("nan")),
        "parent_peak_drop": _peak_drop(pf),
        "child_peak_drop": _peak_drop(cf),
        "com_direction_preserved": bool(
            np.sign(pf.get("freq_com", np.nan) - 0.5) == np.sign(cf.get("freq_com", np.nan) - 0.5)
        ) if (pf.get("freq_com") == pf.get("freq_com") and cf.get("freq_com") == cf.get("freq_com")) else None,
        # E12 single-dose frequency effect (+ C4 specificity)
        "parent_frequency_effect": p_fe,
        "child_frequency_effect": c_fe,
        "delta_frequency_effect": c_fe - p_fe,
        "frequency_effect_sign_preserved": sign_preserved,
        "parent_specificity_verdict": pp.get("specificity_verdict"),
        "child_specificity_verdict": cp.get("specificity_verdict"),
    }


# ---------------------------------------------------------------------------
# M7 — utility-signature inheritance (4th axis; from the separate M7 experiment)
# ---------------------------------------------------------------------------

def compare_utility(parent: dict, child: dict) -> dict:
    """
    Does the weight-space dimension-utility signature survive the ring (M7)?

    Reads the utility summary merged onto each result (from
    ``results/utility/<model>.json``). Reports the change in Cohen's d and
    whether its *sign* (retrieval heads higher/lower utility than the rest) is
    preserved — the inheritance question on the v1 Layer-A axis.
    """
    pu = parent.get("utility", {}) or {}
    cu = child.get("utility", {}) or {}
    pd_, cd_ = pu.get("cohens_d", float("nan")), cu.get("cohens_d", float("nan"))
    sign_preserved = None
    if pd_ == pd_ and cd_ == cd_:
        sign_preserved = bool(np.sign(pd_) == np.sign(cd_))
    return {
        "parent_cohens_d": pd_,
        "child_cohens_d": cd_,
        "delta_cohens_d": cd_ - pd_,
        "sign_preserved": sign_preserved,
        "parent_verdict": pu.get("hypothesis_supported"),
        "child_verdict": cu.get("hypothesis_supported"),
    }


# ---------------------------------------------------------------------------
# E13 — behaviour bridge
# ---------------------------------------------------------------------------

def behavior_bridge(parent: dict, child: dict) -> dict:
    """Pair the behavioural delta with the profile delta (E13 / RQ4 input)."""
    def _niah(r):
        # Long-context NIAH is the meaningful axis; fall back to overall if absent.
        b = r.get("behavior", {})
        return b.get("niah_long", b.get("niah_overall", float("nan")))

    def _ruler(r):
        b = r.get("behavior", {}).get("ruler", {})
        tm = b.get("task_means", {})
        return float(np.mean(list(tm.values()))) if tm else float("nan")

    def _scalar(r, name):
        return (r.get("profile", {}).get("scalars", {}) or {}).get(name, float("nan"))

    profile_keys = ["n_heads", "frac", "gini", "layer_com", "freq_com", "knockout_drop"]
    return {
        "delta_niah": _niah(child) - _niah(parent),
        "delta_ruler": _ruler(child) - _ruler(parent),
        "delta_profile": {k: _scalar(child, k) - _scalar(parent, k) for k in profile_keys},
    }


# ---------------------------------------------------------------------------
# Full ring comparison
# ---------------------------------------------------------------------------

def compare_ring(parent: dict, child: dict, *, lineage: str = "") -> dict:
    """Run the full E10–E13 inheritance package for one parent→child ring."""
    same_arch = _scores(parent, "argmax") is not None and \
        _scores(child, "argmax") is not None and \
        _scores(parent, "argmax").shape == _scores(child, "argmax").shape
    return {
        "lineage": lineage,
        "parent": parent.get("model"),
        "child": child.get("model"),
        "same_architecture": bool(same_arch),
        "E10_identity": compare_identity(parent, child),
        "E11_function": compare_function(parent, child),
        "E12_frequency": compare_frequency(parent, child),
        "M7_utility": compare_utility(parent, child),
        "E13_bridge": behavior_bridge(parent, child),
    }


def quant_ablation(
    instruct: dict,
    bnb4: dict | None = None,
    awq: dict | None = None,
    gptq: dict | None = None,
) -> dict:
    """
    E14 — quantization CROSS-METHOD ablation on one instruct model.

    For each 4-bit METHOD (bnb NF4 / AWQ / GPTQ) vs the SAME fp16 instruct
    reference, applies the Part-1 three-level comparison through the shared
    ``compare_identity`` / ``compare_frequency`` code path (so every number is
    pipeline-traceable, not hand-computed):
        level 1  identity   head-set Jaccard (copy detector)
        level 2  scores      per-head score Spearman (same architecture)
        level 3  finding     does the frequency_effect keep its sign?
    A change that is artifactual typically perturbs level 1 while leaving level 3
    intact; a real degradation moves all three. Comparing the three methods
    against the one reference is the cross-method E14 result (do different quant
    methods preserve the retrieval circuit differently?).
    """
    def _one(ref: dict, q: dict | None) -> dict | None:
        if q is None:
            return None
        ident = compare_identity(ref, q)["copy"]
        freq = compare_frequency(ref, q)
        return {
            "identity_jaccard": ident["jaccard"],
            "per_head_score_spearman": ident["per_head_score_spearman"],
            "ref_frequency_effect": freq["parent_frequency_effect"],
            "quant_frequency_effect": freq["child_frequency_effect"],
            "finding_sign_preserved": freq["frequency_effect_sign_preserved"],
        }

    return {
        "reference": instruct.get("model"),
        "bnb4": _one(instruct, bnb4),
        "awq4": _one(instruct, awq),
        "gptq4": _one(instruct, gptq),
    }


def compare_invariant(parent: dict, child: dict) -> dict:
    """
    Architecture-INVARIANT comparison for cross-architecture pairs (distillation
    siblings, e.g. Gemma-2-9B vs Gemma-2-2B).

    Head-set Jaccard is meaningless across different (n_layers, n_heads) — the
    head index spaces don't align, so a near-zero Jaccard is an artifact, not a
    finding. This compares only scalars that are defined regardless of
    architecture: the causal frequency effect, the knockout drop, the utility
    sign, and behaviour. Use this (not ``compare_identity``) for siblings.
    """
    def _fe(r):
        return (r.get("profile", {}).get("freq_patch") or {}).get("frequency_effect", float("nan"))

    def _ko(r):
        return (r.get("profile", {}).get("knockout") or {}).get("knockout_drop", float("nan"))

    def _ud(r):
        return (r.get("utility", {}) or {}).get("cohens_d", float("nan"))

    p_fe, c_fe = _fe(parent), _fe(child)
    sign = None
    if p_fe == p_fe and c_fe == c_fe:
        sign = bool(np.sign(p_fe) == np.sign(c_fe))
    return {
        "note": "cross-architecture: head-set identity (Jaccard) is NOT comparable; "
                "only architecture-invariant axes are reported.",
        "identity_comparable": False,
        "parent_frequency_effect": p_fe, "child_frequency_effect": c_fe,
        "frequency_effect_sign_preserved": sign,
        "parent_knockout_drop": _ko(parent), "child_knockout_drop": _ko(child),
        "parent_utility_d": _ud(parent), "child_utility_d": _ud(child),
        "delta_niah": behavior_bridge(parent, child)["delta_niah"],
    }


def localize_recall_change(ring: dict) -> dict:
    """
    RQ4 conditional read-out (E15): when the child's recall dropped, is it an
    *identity loss* (head set changed) or a *weakening* (same heads, lower
    score)? Uses only the E10 outputs already in ``ring``.
    """
    ident = ring["E10_identity"]["copy"]      # copy score is the trusted detector
    bridge = ring["E13_bridge"]
    recall_dropped = bridge["delta_niah"] < -0.02
    jacc = ident["jaccard"]
    score_corr = ident["per_head_score_spearman"]
    if not recall_dropped:
        verdict = "no-recall-drop"
    elif jacc < 0.6:
        verdict = "identity-loss (head set changed)"
    elif score_corr == score_corr and score_corr > 0.7:
        verdict = "weakening (same heads, lower scores)"
    else:
        verdict = "mixed/ambiguous"
    return {
        "recall_dropped": bool(recall_dropped),
        "delta_niah": bridge["delta_niah"],
        "head_jaccard": jacc,
        "per_head_score_spearman": score_corr,
        "verdict": verdict,
    }
