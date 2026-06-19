"""
M7 — dimension-utility signature (a SEPARATE experiment, folded from v1 A-series).

This is the Chiang–Yogatama / Part-1 Layer-A test, run as its own experiment on
top of an already-extracted profile — **not** wired into the core profile
pipeline. It asks a weight-space question the behavioural profile does not:

    Do the retrieval heads' Q-projection dimensions carry *more utility* (higher
    L1 norm) than non-retrieval heads? (Cohen's d, layer-clustered permutation p,
    layer-partial Spearman of per-head retrieval score vs utility.)

Why separate (user's call): it is a distinct experiment with a distinct claim;
it should not bloat the E1–E5 profile flow. It is cheap — it needs only the model
*weights* (no generation) plus the already-saved retrieval-head set — so it runs
fast on any model that already has a profile JSON, and it contributes:
  • a per-model weight-space signature (reproduces v1 A8 on the full panel), and
  • a 4th **inheritance** axis: does the utility gap survive instruct/quant?
      (consumed by rhp.inheritance.compare_utility).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def utility_signature(
    analyzer,
    norms: np.ndarray,
    argmax_scores: np.ndarray,
    argmax_heads: list[tuple[int, int]],
) -> dict:
    """
    Compute the dimension-utility signature for one model (M7).

    Args:
        analyzer: a Part-1 ``DimensionUtilityAnalyzer`` bound to the model.
        norms: ``analyzer.compute_query_projection_norms()`` — (L, H, head_dim).
        argmax_scores: the (L, H) retrieval-score matrix (for the correlation).
        argmax_heads: detected retrieval heads (the d-test grouping).

    Returns a dict: cohens_d, clustered_permutation_p, retrieval/non_retrieval
    means, layer-partial Spearman (the statistic to trust), raw Spearman, and the
    v1 "hypothesis_supported" screening label (H2 / anomaly / inconclusive).
    """
    ustats = analyzer.compute_utility_scores(norms, argmax_heads)
    ucorr = analyzer.compute_retrieval_utility_correlation(argmax_scores, norms)

    d = ustats["cohens_d"]
    p = ustats["clustered_permutation_p"]
    ret_m = ustats["retrieval_mean"]
    non_m = ustats["non_retrieval_mean"]

    # Part-1 screening label (report the cluster-aware p as primary).
    if p != p:  # NaN
        verdict = "inconclusive"
    elif p < 0.05 and ret_m < non_m:
        verdict = "H2 (retrieval heads lower-utility — degradation-consistent)"
    elif p < 0.05 and ret_m > non_m:
        verdict = "anomaly (retrieval heads higher-utility)"
    else:
        verdict = "inconclusive"

    out = {
        "cohens_d": d,
        "clustered_permutation_p": p,
        "retrieval_mean": ret_m,
        "non_retrieval_mean": non_m,
        "partial_spearman": ucorr["partial_spearman_r"],
        "partial_p": ucorr["partial_p"],
        "spearman_rho": ucorr["spearman_rho"],
        "hypothesis_supported": verdict,
        "n_retrieval": ustats["n_retrieval"],
        "n_non_retrieval": ustats["n_non_retrieval"],
    }
    logger.info(
        "M7 utility: d=%.3f (clustered p=%.4g) | layer-partial ρ=%.3f → %s",
        d, p, ucorr["partial_spearman_r"], verdict,
    )
    return out
