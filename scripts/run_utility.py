"""
M7 — dimension-utility signature, as a SEPARATE experiment (folded from v1 A-series).

Runs the Chiang–Yogatama / Layer-A utility d-test on models that ALREADY have a
profile JSON (it reuses their saved retrieval-head set, so no re-detection). Pure
weight-space — fast. Writes ``results/utility/<model>_seed<seed>.json``, which
``run_prediction.py`` and ``run_inheritance.py`` pick up as an extra predictor /
a 4th inheritance axis.

Resume-safe + adaptive 23 h guard, like every GPU script here.

Examples:
    python scripts/run_utility.py --models all
    python scripts/run_utility.py --models core --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_utility")


def parse_args():
    ap = argparse.ArgumentParser(description="M7 dimension-utility signature (separate experiment).")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="model keys, or one of: all / core / panel.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def resolve_models(config, spec):
    from rhp.panel import all_model_keys, core_models, models_by_tier
    if spec == ["all"]:
        return all_model_keys(config, include_quant=True)
    if spec == ["core"]:
        return core_models(config)
    if spec == ["panel"]:
        return models_by_tier(config, "panel") + core_models(config)
    return spec


def main():
    args = parse_args()
    from scripts._common import bootstrap, run_utility_for_model, save_json, time_guard
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel, model_cfg

    config = load_panel(args.config)
    results_dir = Path(args.results_dir or config["output"]["results_dir"])
    out_dir = results_dir / "utility"
    out_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = results_dir / "profile"
    models = resolve_models(config, args.models)
    logger.info("M7 utility over %d models: %s", len(models), models)

    start = time.time()
    model_times: list[float] = []
    for key in models:
        out = out_dir / f"{key}_seed{args.seed}.json"
        if out.exists():
            logger.info("[%s] exists → skip.", key)
            continue
        prof = prof_dir / f"{key}_seed{args.seed}.json"
        if not prof.exists():
            logger.warning("[%s] no profile yet (%s) — run the profile first; skipping.", key, prof)
            continue
        ok, elapsed_h, est_h = time_guard(start, model_times, first_est_h=4.0)
        if not ok:
            logger.warning("Adaptive cap: %.1f h + est %.1f h > 23 h. Re-run to resume at %s.",
                           elapsed_h, est_h, key)
            break
        with open(prof, encoding="utf-8") as f:
            pres = json.load(f)
        t0 = time.time()
        try:
            res = run_utility_for_model(
                key, model_cfg(config, key), config,
                argmax_heads=pres.get("argmax_heads", []),
                argmax_scores=pres.get("argmax_scores", []),
                seed=args.seed,
            )
            save_json(res, out)
            model_times.append((time.time() - t0) / 3600)
            logger.info("[%s] saved → %s (d=%.3f)", key, out, res["utility"]["cohens_d"])
        except Exception as exc:
            logger.error("[%s] FAILED: %s", key, exc, exc_info=True)

    logger.info("Done. Elapsed %.1f h.", (time.time() - start) / 3600)


if __name__ == "__main__":
    main()
