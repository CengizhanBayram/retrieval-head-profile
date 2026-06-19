"""
Standardised retrieval-head profile (proposal §3.1, Block A: E1, E4, E5).

The *profile* is the per-model vector that the whole study turns on:

    n_heads, n_heads_copy   detected retrieval heads (argmax / copy detector)   [E1]
    frac                    n_heads / total heads                               [E1]
    detector_agreement      Jaccard(argmax-set, copy-set)                       [E1]
    gini, zero_fraction     score concentration (Qwen-type vs OLMo-type)        [E4]
    layer_com, layer_*_frac layer-depth distribution (early/mid/late mass)      [E4]
    gqa_*                   KV-group distribution control (not a sharing artifact)[E5]
    freq_com, freq_width    summary of the spectral signature (filled by E2)
    knockout_drop           causal mask effect (filled by E3)

Everything in this module is pure analysis over the two score matrices and the
model config — **no GPU**. The GPU work (running the two detectors, the
frequency sweep E2 and the knockout E3) lives in ``scripts/run_profile.py`` and
in ``rhp.freq_signature`` / ``rhp.knockout``; their outputs are merged in here.
"""

from __future__ import annotations

import logging

import numpy as np

from src.stats_utils import jaccard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# E4 — concentration and layer-distribution
# ---------------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative score distribution (E4).

    0 ⇒ perfectly uniform (OLMo-type, diffuse retrieval); →1 ⇒ a few heads carry
    all the mass (Qwen-type, concentrated). Computed over the flattened score
    matrix including zeros, which is the point: the zero-inflation *is* the
    concentration signal.
    """
    v = np.asarray(values, dtype=np.float64).flatten()
    v = np.clip(v, 0, None)
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    v_sorted = np.sort(v)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * v_sorted)) / (n * v_sorted.sum()))


def concentration(scores: np.ndarray, threshold: float) -> dict:
    """Concentration metrics for E4: Gini, zero-score fraction, top-k share."""
    flat = np.asarray(scores, dtype=np.float64).flatten()
    nonzero = flat[flat > 0]
    total_mass = flat.sum()
    top5_share = float(np.sort(flat)[-5:].sum() / total_mass) if total_mass > 0 else 0.0
    return {
        "gini": gini(flat),
        "zero_fraction": float(np.mean(flat <= 0.0)),
        "below_threshold_fraction": float(np.mean(flat < threshold)),
        "n_active_heads": int((flat > 0).sum()),
        "score_mean_active": float(nonzero.mean()) if nonzero.size else 0.0,
        "top5_mass_share": top5_share,
    }


def layer_profile(scores: np.ndarray, retrieval_heads: list[tuple[int, int]]) -> dict:
    """
    Layer-depth distribution of the retrieval heads (E4).

    Returns the score-weighted centre of mass (normalised to [0,1] over depth)
    and the early/mid/late thirds' share of retrieval-head count. With no
    retrieval heads everything is NaN/0 and a warning is logged.
    """
    scores = np.asarray(scores)
    n_layers, _ = scores.shape
    if not retrieval_heads:
        logger.warning("layer_profile: no retrieval heads — layer metrics are NaN.")
        return {
            "layer_com": float("nan"), "layer_com_weighted": float("nan"),
            "early_frac": float("nan"), "mid_frac": float("nan"), "late_frac": float("nan"),
            "n_layers": n_layers,
        }
    layers = np.array([l for (l, _h) in retrieval_heads], dtype=np.float64)
    weights = np.array([scores[l, h] for (l, h) in retrieval_heads], dtype=np.float64)
    com_count = layers.mean() / max(1, n_layers - 1)
    com_weighted = (np.average(layers, weights=weights) / max(1, n_layers - 1)
                    if weights.sum() > 0 else com_count)
    thirds = layers / n_layers
    return {
        "layer_com": float(com_count),
        "layer_com_weighted": float(com_weighted),
        "early_frac": float(np.mean(thirds < 1 / 3)),
        "mid_frac": float(np.mean((thirds >= 1 / 3) & (thirds < 2 / 3))),
        "late_frac": float(np.mean(thirds >= 2 / 3)),
        "n_layers": n_layers,
    }


# ---------------------------------------------------------------------------
# E5 — GQA group-distribution control
# ---------------------------------------------------------------------------

def gqa_group_distribution(
    retrieval_heads: list[tuple[int, int]],
    n_heads: int,
    n_kv_heads: int,
) -> dict:
    """
    Show the detected set is not a KV-sharing artifact (E5 — Part-1 control
    generalised). For GQA, query heads ``h`` map to KV group ``h // group``;
    if retrieval heads were a sharing artifact they would fill whole KV groups.

    Returns the mean retrieval-heads-per-active-group and the fraction of active
    groups that are *fully* retrieval (a high full-group fraction would be the
    red flag). For MHA (n_kv == n_heads) this is trivially 1-per-group.
    """
    group = max(1, n_heads // max(1, n_kv_heads))
    from collections import defaultdict

    by_group: dict[tuple[int, int], set[int]] = defaultdict(set)
    for (layer, head) in retrieval_heads:
        by_group[(layer, head // group)].add(head)
    if not by_group:
        return {"group_size": group, "n_active_groups": 0,
                "mean_heads_per_active_group": 0.0, "full_group_fraction": 0.0}
    sizes = [len(s) for s in by_group.values()]
    full = sum(1 for s in sizes if s == group)
    return {
        "group_size": group,
        "n_active_groups": len(by_group),
        "mean_heads_per_active_group": float(np.mean(sizes)),
        "full_group_fraction": float(full / len(by_group)),
    }


# ---------------------------------------------------------------------------
# E1 — detector agreement
# ---------------------------------------------------------------------------

def detector_agreement(
    argmax_heads: list[tuple[int, int]],
    copy_heads: list[tuple[int, int]],
) -> dict:
    """Jaccard + set deltas between the two detectors (E1 reliability metric)."""
    j = jaccard(argmax_heads, copy_heads)
    return {
        "jaccard": j["jaccard"],
        "intersection": j["intersection"],
        "union": j["union"],
        "n_argmax": j["n_a"],
        "n_copy": j["n_b"],
        "only_argmax": j["only_a"],
        "only_copy": j["only_b"],
    }


def per_head_score_correlation(
    argmax_scores: np.ndarray, copy_scores: np.ndarray
) -> dict:
    """Spearman ρ between the two detectors' dense per-head scores (E1/R2)."""
    from scipy import stats

    a = np.asarray(argmax_scores).flatten()
    b = np.asarray(copy_scores).flatten()
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return {"spearman_rho": float("nan"), "spearman_p": float("nan")}
    rho, p = stats.spearmanr(a, b)
    return {"spearman_rho": float(rho), "spearman_p": float(p)}


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------

def build_profile(
    *,
    model_key: str,
    argmax_scores: np.ndarray,
    copy_scores: np.ndarray,
    threshold: float,
    n_kv_heads: int,
    argmax_heads: list[tuple[int, int]] | None = None,
    copy_heads: list[tuple[int, int]] | None = None,
    freq_summary: dict | None = None,
    freq_patch_summary: dict | None = None,
    knockout_summary: dict | None = None,
    seed: int = 42,
) -> dict:
    """
    Assemble the standardised profile vector for one model (proposal §3.1).

    The two score matrices come from the two detectors. Optional, merged when
    available: ``freq_summary`` (E2 — the 8-window spectral signature),
    ``freq_patch_summary`` (E12/C4 — the single 50%-coverage low/high-freq
    population patch with perplexity specificity), and ``knockout_summary``
    (E3). Heads are recomputed from the score matrices at ``threshold`` unless
    passed in.
    """
    argmax_scores = np.asarray(argmax_scores)
    copy_scores = np.asarray(copy_scores)
    n_layers, n_heads = argmax_scores.shape
    total = n_layers * n_heads

    if argmax_heads is None:
        ls, hs = np.where(argmax_scores >= threshold)
        argmax_heads = sorted(zip(ls.tolist(), hs.tolist()))
    if copy_heads is None:
        ls, hs = np.where(copy_scores >= threshold)
        copy_heads = sorted(zip(ls.tolist(), hs.tolist()))

    conc = concentration(argmax_scores, threshold)
    lp = layer_profile(argmax_scores, argmax_heads)
    gqa = gqa_group_distribution(argmax_heads, n_heads, n_kv_heads)
    agree = detector_agreement(argmax_heads, copy_heads)
    corr = per_head_score_correlation(argmax_scores, copy_scores)

    profile = {
        "model": model_key,
        "seed": seed,
        "threshold": threshold,
        "n_layers": n_layers,
        "n_heads_total": total,
        # E1
        "n_heads": len(argmax_heads),
        "n_heads_copy": len(copy_heads),
        "frac": len(argmax_heads) / total,
        "frac_copy": len(copy_heads) / total,
        "detector_agreement": agree,
        "detector_score_spearman": corr,
        # E4
        "concentration": conc,
        "layer_profile": lp,
        # E5
        "gqa": gqa,
        # filled by E2 / E12+C4 / E3 when present (M7 utility is a SEPARATE
        # experiment, stored in results/utility/<model>.json, not here)
        "freq_signature": freq_summary,
        "freq_patch": freq_patch_summary,
        "knockout": knockout_summary,
        # convenience scalars for the E8 prediction table
        "scalars": {
            "n_heads": len(argmax_heads),
            "frac": len(argmax_heads) / total,
            "gini": conc["gini"],
            "zero_fraction": conc["zero_fraction"],
            "layer_com": lp["layer_com_weighted"],
            "detector_jaccard": agree["jaccard"],
            "freq_com": (freq_summary or {}).get("freq_com", float("nan")),
            "freq_width": (freq_summary or {}).get("freq_width", float("nan")),
            "frequency_effect": (freq_patch_summary or {}).get("frequency_effect", float("nan")),
            "specificity_ratio": (freq_patch_summary or {}).get("specificity_ratio", float("nan")),
            "knockout_drop": (knockout_summary or {}).get("knockout_drop", float("nan")),
        },
    }
    return profile
