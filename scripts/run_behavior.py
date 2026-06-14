"""
Block B — behavioural targets (E6 NIAH sweep + E7 RULER subset) over the panel.

Same resume + time-budget contract as ``run_profile.py``. Writes
``results/behavior/<model>_seed<seed>.json``. ≤3B models additionally get the
long-context lengths (8192/16384) from ``niah.context_lengths_small``.

Examples:
    python scripts/run_behavior.py --models core
    python scripts/run_behavior.py --models all --time-budget-hours 20 --no-ruler
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_behavior")


def parse_args():
    ap = argparse.ArgumentParser(description="Block B behaviour (E6 NIAH + E7 RULER).")
    ap.add_argument("--models", nargs="+", default=["core"])
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--no-ruler", action="store_true")
    ap.add_argument("--time-budget-hours", type=float, default=23.0)
    ap.add_argument("--overwrite", action="store_true")
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
    from scripts._common import bootstrap, run_behavior_for_model, save_json
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel, model_cfg

    config = load_panel(args.config)
    results_dir = Path(args.results_dir or config["output"]["results_dir"]) / "behavior"
    results_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_models(config, args.models)
    logger.info("Behaviour run over %d models: %s", len(models), models)

    small_lengths = config["niah"].get("context_lengths_small")
    start = time.time()
    budget = args.time_budget_hours * 3600
    for key in models:
        out = results_dir / f"{key}_seed{args.seed}.json"
        if out.exists() and not args.overwrite:
            logger.info("[%s] exists → skip.", key)
            continue
        if time.time() - start > budget:
            logger.warning("Time budget reached — stopping before %s. Re-run to resume.", key)
            break
        cfg = model_cfg(config, key)
        # ≤3B models can afford the long-context lengths.
        ctx = None
        if small_lengths and any(t in key for t in ("3b", "2b", "1_6b", "mini")):
            ctx = small_lengths
        try:
            res = run_behavior_for_model(
                key, cfg, config, seed=args.seed, context_lengths=ctx,
                n_samples=args.n_samples, do_ruler=not args.no_ruler,
            )
            res["family"] = cfg.get("family", "unknown")
            save_json(res, out)
            logger.info("[%s] saved → %s", key, out)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", key, exc, exc_info=True)

    logger.info("Done. Elapsed %.1f h.", (time.time() - start) / 3600)


if __name__ == "__main__":
    main()
