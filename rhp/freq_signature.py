"""
Frequency-signature sweep — the spectral profile (proposal §3.1, E2).

For a model, slide a ``k = head_dim / divisor`` window across the RoPE frequency
spectrum at ``n_windows`` positions and, at each position, population-patch
(zero) those dimensions in the retrieval-head population, measuring the NIAH
recall drop. The resulting curve is the model's *frequency signature*; its
centre of mass and width become two profile scalars (``freq_com``,
``freq_width``).

Built entirely on the inherited ``src.activation_patching.ActivationPatcher``
(``_evaluate_population`` / ``patch_heads``) and ``DimensionUtilityAnalyzer``
(``freq_order``). Two mandatory controls travel with every window:

    C2  layer-matched non-retrieval heads  — same dims, control heads. A drop
        that vanishes here ⇒ the effect is retrieval-head-specific.
    C3  random dimensions (equal count)    — isolates "this frequency band"
        from "removing any k dims".
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _coverage_subset(
    retrieval_heads: list[tuple[int, int]],
    coverage: float,
    top_k: int,
    scores: np.ndarray,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Highest-scoring ``coverage`` fraction of retrieval heads, capped at top_k."""
    ranked = sorted(retrieval_heads, key=lambda lh: scores[lh[0], lh[1]], reverse=True)
    n = max(1, int(round(len(ranked) * coverage)))
    return ranked[: min(n, top_k)]


def _layer_matched_controls(
    patched_heads: list[tuple[int, int]],
    all_retrieval_heads: list[tuple[int, int]],
    n_heads: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """One non-retrieval head in the SAME layer as each patched head (C2)."""
    excl = set(all_retrieval_heads)
    rng = np.random.default_rng(seed)
    used: set[tuple[int, int]] = set()
    controls: list[tuple[int, int]] = []
    for (layer, _h) in patched_heads:
        cand = [h for h in range(n_heads) if (layer, h) not in excl and (layer, h) not in used]
        if cand:
            ch = (layer, int(rng.choice(cand)))
            controls.append(ch)
            used.add(ch)
    return controls


def frequency_signature(
    patcher,
    *,
    retrieval_heads: list[tuple[int, int]],
    retrieval_scores: np.ndarray,
    freq_order: np.ndarray,
    head_dim: int,
    samples: list[dict],
    n_heads: int,
    n_windows: int = 8,
    window_divisor: int = 8,
    coverage: float = 0.5,
    top_k: int = 30,
    seed: int = 42,
) -> dict:
    """
    Run the E2 spectral sweep and return the curve + summary scalars.

    Args:
        patcher: an ``ActivationPatcher`` bound to the loaded model.
        retrieval_heads / retrieval_scores: from the (argmax) detector.
        freq_order: ascending-frequency dim order (``analyzer.freq_order``).
        head_dim: per-head dimension.
        samples: NIAH samples to score recall on (one context, mid position).
        n_heads: model's query-head count (for control matching).

    Returns dict with:
        window_centers   normalised window position on the spectrum [0,1]
        window_logfreq   mean log10(freq) of each window (interpretable axis)
        baseline         unpatched recall
        recall           per-window recall (retrieval population patched)
        drop             baseline − recall  (the signature curve)
        recall_control   per-window recall with layer-matched control heads (C2)
        recall_randdim   per-window recall with random dims, equal count (C3)
        freq_com, freq_width   centre of mass / width of ``drop`` over [0,1]
        n_patched        size of the patched population
    """
    fo = np.asarray(freq_order).astype(int)
    k = max(2, head_dim // window_divisor)
    starts = np.unique(np.linspace(0, head_dim - k, n_windows).astype(int))
    windows = [fo[s : s + k].tolist() for s in starts]

    patched = _coverage_subset(retrieval_heads, coverage, top_k, retrieval_scores, seed)
    controls = _layer_matched_controls(patched, retrieval_heads, n_heads, seed)
    logger.info(
        "E2 frequency signature: %d windows of %d dims; patching %d/%d heads (coverage=%.2f).",
        len(windows), k, len(patched), len(retrieval_heads), coverage,
    )

    baseline, _ = patcher._evaluate_population(samples, None)

    rng = np.random.default_rng(seed)
    recall, recall_ctrl, recall_rand, logfreqs, centers = [], [], [], [], []
    # crude per-dim frequency for the interpretable log-freq axis
    half = head_dim // 2
    # freq_order already sorts dims ascending; reconstruct per-dim freq via rank.
    for wi, (start, dims) in enumerate(zip(starts, windows)):
        centers.append(float((start + k / 2) / max(1, head_dim)))
        # mean rank-position → pseudo-logfreq (monotone proxy is enough here)
        logfreqs.append(float(np.mean([start + i for i in range(k)]) / head_dim))

        head_dims = {lh: dims for lh in patched}
        r, _ = patcher._evaluate_population(samples, head_dims)
        recall.append(r)

        # C2: same dims in layer-matched non-retrieval heads
        if controls:
            rc, _ = patcher._evaluate_population(samples, {lh: dims for lh in controls})
        else:
            rc = float("nan")
        recall_ctrl.append(rc)

        # C3: random dims of equal count in the retrieval population
        rand_dims = rng.choice(head_dim, size=k, replace=False).tolist()
        rr, _ = patcher._evaluate_population(samples, {lh: rand_dims for lh in patched})
        recall_rand.append(rr)
        logger.info(
            "  window %d/%d [center=%.2f]: recall=%.3f (ctrl=%.3f rand=%.3f) drop=%.3f",
            wi + 1, len(windows), centers[-1], r, rc, rr, baseline - r,
        )

    drop = (np.asarray(recall) * -1 + baseline).clip(min=0)
    x = np.asarray(centers)
    if drop.sum() > 0:
        com = float(np.average(x, weights=drop))
        width = float(np.sqrt(np.average((x - com) ** 2, weights=drop)))
    else:
        com, width = float("nan"), float("nan")

    return {
        "window_centers": centers,
        "window_logfreq": logfreqs,
        "window_size_dims": k,
        "baseline": baseline,
        "recall": recall,
        "drop": drop.tolist(),
        "recall_control": recall_ctrl,
        "recall_randdim": recall_rand,
        "freq_com": com,
        "freq_width": width,
        "n_patched": len(patched),
        "n_controls": len(controls),
        "coverage": coverage,
    }
