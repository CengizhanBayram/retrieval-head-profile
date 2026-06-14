"""
Block A — retrieval-head profile extraction (E1–E5) over the panel.

Resume-safe and time-budgeted so it runs across multiple ≤24 h Colab sessions:
every finished model is written to ``results/profile/<model>_seed<seed>.json``
and skipped on the next run; the loop stops launching new models once the wall
clock passes ``--time-budget-hours``.

Examples:
    python scripts/run_profile.py --models core
    python scripts/run_profile.py --models qwen25_7b llama31_8b --seed 42
    python scripts/run_profile.py --models all --time-budget-hours 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make the project root importable so ``from scripts._common import ...`` works
# regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_profile")


def parse_args():
    ap = argparse.ArgumentParser(description="Block A profile extraction (E1–E5).")
    ap.add_argument("--models", nargs="+", default=["core"],
                    help="model keys, or one of: all / core / panel.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None, help="Path to the Part-1 repo (for src/).")
    ap.add_argument("--results-dir", default=None, help="Overrides config output.results_dir.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--no-freq", action="store_true", help="Skip E2 (frequency signature).")
    ap.add_argument("--no-knockout", action="store_true", help="Skip E3 (knockout).")
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
    from scripts._common import bootstrap, run_profile_for_model, save_json
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel, model_cfg

    config = load_panel(args.config)
    results_dir = Path(args.results_dir or config["output"]["results_dir"]) / "profile"
    results_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_models(config, args.models)
    logger.info("Profile run over %d models: %s", len(models), models)

    start = time.time()
    budget = args.time_budget_hours * 3600
    for key in models:
        out = results_dir / f"{key}_seed{args.seed}.json"
        if out.exists() and not args.overwrite:
            logger.info("[%s] exists → skip.", key)
            continue
        if time.time() - start > budget:
            logger.warning("Time budget (%.1f h) reached — stopping before %s. Re-run to resume.",
                           args.time_budget_hours, key)
            break
        try:
            res = run_profile_for_model(
                key, model_cfg(config, key), config,
                seed=args.seed, n_samples=args.n_samples, context_length=args.context,
                do_freq=not args.no_freq, do_knockout=not args.no_knockout,
            )
            save_json(res, out)
            logger.info("[%s] saved → %s", key, out)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", key, exc, exc_info=True)

    logger.info("Done. Elapsed %.1f h.", (time.time() - start) / 3600)


if __name__ == "__main__":
    main()
