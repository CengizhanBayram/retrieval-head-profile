"""
Shared helpers for the Part-2 scripts.

Centralises: path bootstrap (make ``rhp`` and the inherited ``src`` importable),
the full single-model profile pipeline (E1–E5), and the single-model behaviour
pipeline (E6–E7). Both notebooks and the CLI scripts call into here, so the
exact same code path runs in Colab and locally.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

def bootstrap(part1_repo: str | None = None) -> str:
    """Put the project root and the Part-1 repo on sys.path. Returns Part-1 path."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from rhp._paths import ensure_part1_on_path
    return ensure_part1_on_path(part1_repo)


# ---------------------------------------------------------------------------
# Atomic JSON save (resume-safe, mirrors Part-1 checkpointing)
# ---------------------------------------------------------------------------

def save_json(obj, path: str | Path) -> None:
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Profile pipeline (Block A: E1–E5) for one model
# ---------------------------------------------------------------------------

def run_profile_for_model(
    model_key: str,
    model_cfg: dict,
    config: dict,
    *,
    seed: int = 42,
    n_samples: int | None = None,
    context_length: int = 4096,
    do_freq: bool = True,
    do_knockout: bool = True,
    top_k_heads: int | None = None,
) -> dict:
    """
    Full Block-A profile for one model. Loads the model, runs both detectors
    (E1), dimension utility (for freq_order), the frequency signature (E2), the
    knockout (E3), assembles the profile (E4/E5), unloads the model, and returns
    a self-contained result dict (also holds the dense score matrices for E10).
    """
    import numpy as np
    import torch

    import random as _random

    from src.model_loader import purge_hf_cache
    from src.repro import capture_environment, set_determinism
    from src.retrieval_head_detector import RetrievalHeadDetector
    from src.dimension_utility import DimensionUtilityAnalyzer
    from src.activation_patching import ActivationPatcher
    from src.corpus import build_haystack, load_haystack_corpus

    from rhp.loader import load_model_any as load_model
    from rhp.copy_score_detector import CopyScoreDetector
    from rhp.knockout import KnockoutEvaluator
    from rhp.freq_signature import frequency_signature
    from rhp.profile import build_profile

    niah = config["niah"]
    pcfg = config["profile"]
    threshold = niah.get("score_threshold", 0.1)
    n_samples = n_samples or niah.get("n_samples", 200)
    top_k_heads = top_k_heads or pcfg.get("top_k_heads", 30)
    positions = niah["needle_positions"]

    set_determinism(seed, strict=config.get("reproducibility", {}).get("strict_determinism", False))
    t0 = time.time()
    model = tok = None
    try:
        model, tok = load_model(model_cfg, model_key)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)

        # E1 — argmax detector
        det = RetrievalHeadDetector(model, tok, config, score_threshold=threshold, seed=seed)
        a_samples = det.generate_niah_samples(n_samples, [context_length], positions)
        argmax_scores = det.score_heads(a_samples)
        argmax_heads = det.get_retrieval_heads(argmax_scores)

        # E1 — copy-score detector (second detector)
        cdet = CopyScoreDetector(model, tok, config, score_threshold=threshold, seed=seed)
        c_samples = cdet.generate_samples(max(80, n_samples // 2), [context_length], positions)
        copy_scores = cdet.score_heads(c_samples)
        copy_heads = cdet.get_retrieval_heads(copy_scores)

        # Dimension utility → freq_order (for E2)
        analyzer = DimensionUtilityAnalyzer(model, config)
        freq_order = analyzer.freq_order
        head_dim = analyzer.head_dim

        # E2 — frequency signature (8-window spectral sweep) and
        # E12/C4 — single 50%-coverage low/high-freq population patch with the
        # perplexity specificity control (the inherited run_population_patching).
        freq_summary = None
        freq_patch_summary = None
        if do_freq and argmax_heads:
            patcher = ActivationPatcher(model, tok, config)
            fs_samples = det.generate_niah_samples(
                pcfg.get("freq_n_samples", 120), [context_length], [0.5]
            )
            freq_summary = frequency_signature(
                patcher,
                retrieval_heads=argmax_heads,
                retrieval_scores=argmax_scores,
                freq_order=freq_order,
                head_dim=head_dim,
                samples=fs_samples,
                n_heads=n_heads,
                n_windows=pcfg.get("freq_n_windows", 8),
                window_divisor=pcfg.get("freq_window_divisor", 8),
                coverage=pcfg.get("coverage", 0.5),
                top_k=top_k_heads,
                seed=seed,
            )

            # Build the patched population (coverage-capped), the layer-matched
            # control heads (C2), and plain-text passages for the perplexity
            # specificity control (C4 — the 0.33 rule is inside the inherited
            # run_population_patching).
            norms = analyzer.compute_query_projection_norms()
            top_heads = sorted(
                argmax_heads, key=lambda lh: argmax_scores[lh[0], lh[1]], reverse=True
            )[:top_k_heads]
            excl = set(argmax_heads)
            crng = np.random.default_rng(seed)
            used: set = set()
            control_heads: list = []
            for (l, _h) in top_heads:
                cand = [h for h in range(n_heads) if (l, h) not in excl and (l, h) not in used]
                if cand:
                    ch = (l, int(crng.choice(cand)))
                    control_heads.append(ch)
                    used.add(ch)
            psent = load_haystack_corpus(max_sentences=5000)
            ppr = _random.Random(12345)
            ppl_texts = [build_haystack(min(context_length, 4096), psent, ppr) for _ in range(8)]

            freq_patch_summary = patcher.run_population_patching(
                retrieval_heads=top_heads,
                utility_scores={"_norms": norms},
                samples=fs_samples,
                k_dims=config.get("activation_patching", {}).get("k_dims", 16),
                n_samples=len(fs_samples),
                random_seeds=[0, 1, 2],
                freq_order=freq_order,
                control_heads=control_heads,
                perplexity_texts=ppl_texts,
                perplexity_max_len=min(context_length, 4096),
            )

        # E3 — knockout double dissociation
        knockout_summary = None
        if do_knockout and argmax_heads:
            ko = KnockoutEvaluator(model, tok, config)
            ko_samples = det.generate_niah_samples(80, [context_length], [0.25, 0.5, 0.75])
            knockout_summary = ko.run(
                ko_samples, argmax_heads, n_heads, n_layers,
                random_seeds=pcfg.get("knockout_random_seeds", [0, 1, 2, 3, 4]),
            )

        profile = build_profile(
            model_key=model_key,
            argmax_scores=argmax_scores,
            copy_scores=copy_scores,
            threshold=threshold,
            n_kv_heads=n_kv,
            argmax_heads=argmax_heads,
            copy_heads=copy_heads,
            freq_summary=freq_summary,
            freq_patch_summary=freq_patch_summary,
            knockout_summary=knockout_summary,
            seed=seed,
        )

        result = {
            "model": model_key,
            "hf_id": model_cfg.get("hf_id"),
            "family": model_cfg.get("family", "unknown"),
            "seed": seed,
            "context_length": context_length,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv,
            "argmax_heads": argmax_heads,
            "copy_heads": copy_heads,
            "argmax_scores": argmax_scores.tolist(),
            "copy_scores": copy_scores.tolist(),
            "profile": profile,
            "elapsed_sec": round(time.time() - t0, 1),
            "environment": capture_environment(),
        }
        logger.info("[%s] profile done in %.0fs: %d argmax / %d copy heads.",
                    model_key, result["elapsed_sec"], len(argmax_heads), len(copy_heads))
        return result
    finally:
        model = tok = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            purge_hf_cache(model_cfg["hf_id"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Behaviour pipeline (Block B: E6 NIAH + E7 RULER) for one model
# ---------------------------------------------------------------------------

def run_behavior_for_model(
    model_key: str,
    model_cfg: dict,
    config: dict,
    *,
    seed: int = 42,
    context_lengths: list[int] | None = None,
    n_samples: int | None = None,
    do_ruler: bool = True,
) -> dict:
    """E6 NIAH position/length sweep + E7 RULER subset for one model."""
    import gc as _gc

    import numpy as np

    from src.model_loader import purge_hf_cache
    from src.repro import capture_environment, set_determinism
    from src.niah_evaluator import NIAHEvaluator

    from rhp.loader import load_model_any as load_model
    from rhp.ruler import RulerEvaluator

    rcfg = config.get("ruler", {})
    behcfg = config.get("behavior", {})
    # Long-context schedule: per-context sample counts (fewer samples as context
    # grows). The RQ2 target needs variance, which lives at long context.
    schedule_raw = behcfg.get("context_schedule") or {4096: 60, 8192: 36, 16384: 20, 32768: 10}
    schedule = {int(k): int(v) for k, v in schedule_raw.items()}
    positions = behcfg.get("needle_positions") or config["niah"]["needle_positions"]
    long_thresh = int(behcfg.get("long_context_threshold", 16384))
    # Allow an explicit override (e.g. the pilot passes a shorter list).
    ctxs = sorted(context_lengths) if context_lengths else sorted(schedule)

    set_determinism(seed)
    t0 = time.time()
    model = tok = None
    try:
        model, tok = load_model(model_cfg, model_key)

        # E6 — NIAH behavioural sweep, one context at a time so each length gets
        # its own (decreasing) sample budget; OOM at a length → NaN row.
        ev = NIAHEvaluator(model, tok, config, seed=seed)
        rows, per_ctx = [], {}
        for c in ctxs:
            n_c = schedule.get(c, n_samples or 40)
            acc_c = np.asarray(ev.evaluate([c], positions, n_c), dtype=float)  # (1, n_pos)
            rows.append(acc_c[0])
            per_ctx[c] = float(np.nanmean(acc_c))
            logger.info("[%s] NIAH @ %d (n=%d): %.3f", model_key, c, n_c, per_ctx[c])
        acc = np.vstack(rows)                       # (n_ctx, n_pos)
        overall = float(np.nanmean(acc))
        per_pos = np.nanmean(acc, axis=0)
        worst_pos = float(np.nanmin(per_pos))
        long_ctxs = [c for c in ctxs if c >= long_thresh]
        niah_long = (float(np.nanmean([per_ctx[c] for c in long_ctxs]))
                     if long_ctxs else overall)

        behavior = {
            "niah_matrix": acc.tolist(),
            "context_lengths": ctxs,
            "needle_positions": positions,
            "niah_overall": overall,
            "niah_long": niah_long,                 # primary RQ2 target (≥ long_thresh)
            "niah_worst_pos": worst_pos,
            "niah_per_position": per_pos.tolist(),
            "niah_per_context": {str(c): per_ctx[c] for c in ctxs},
        }

        # E7 — RULER subset
        if do_ruler and rcfg:
            rev = RulerEvaluator(model, tok, config, seed=seed)
            behavior["ruler"] = rev.run(
                rcfg.get("tasks", ["multikey", "multivalue", "vartrack"]),
                rcfg.get("context_lengths", [2048, 4096]),
                rcfg.get("n_samples", 100),
                rcfg.get("seeds", [42, 123]),
                n_keys=rcfg.get("multikey_n_keys", 4),
                n_values=rcfg.get("multivalue_n_values", 3),
                chain_len=rcfg.get("vartrack_chain_len", 4),
            )

        result = {
            "model": model_key,
            "hf_id": model_cfg.get("hf_id"),
            "family": model_cfg.get("family", "unknown"),
            "seed": seed,
            "behavior": behavior,
            "elapsed_sec": round(time.time() - t0, 1),
            "environment": capture_environment(),
        }
        logger.info("[%s] behaviour done in %.0fs: NIAH overall=%.3f long=%.3f",
                    model_key, result["elapsed_sec"], overall, niah_long)
        return result
    finally:
        model = tok = None
        _gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            purge_hf_cache(model_cfg["hf_id"])
        except Exception:
            pass
