"""
Block B analysis — profile → behaviour prediction (E8) and test-retest (E9). CPU.

Loads every per-model profile + behaviour JSON, joins them, and runs the RQ2
prediction analysis: BH-corrected single correlations, a LOO 3-predictor
regression per target, and family-demeaned correlations. If two seeds are on
disk it also computes the E9 test-retest reliability ceilings.

Examples:
    python scripts/run_prediction.py
    python scripts/run_prediction.py --seed 42 --retest-seed 123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_prediction")


def parse_args():
    ap = argparse.ArgumentParser(description="E8 prediction + E9 test-retest.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retest-seed", type=int, default=None,
                    help="Second seed for the E9 test-retest (e.g. 123).")
    return ap.parse_args()


def collect(results_dir: Path, seed: int) -> list[dict]:
    """Load + merge profile/behaviour results for every model at one seed."""
    merged = []
    prof_dir = results_dir / "profile"
    if not prof_dir.exists():
        return merged
    for pf in sorted(prof_dir.glob(f"*_seed{seed}.json")):
        with open(pf, encoding="utf-8") as f:
            res = json.load(f)
        model = res.get("model")
        bf = results_dir / "behavior" / f"{model}_seed{seed}.json"
        if bf.exists():
            with open(bf, encoding="utf-8") as f:
                res["behavior"] = json.load(f).get("behavior", {})
        merged.append(res)
    return merged


def main():
    args = parse_args()
    from scripts._common import bootstrap, save_json
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel
    from rhp.prediction import run_prediction_analysis, test_retest

    config = load_panel(args.config)
    results_dir = Path(args.results_dir or config["output"]["results_dir"])
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect(results_dir, args.seed)
    if not results:
        logger.error("No profile results found under %s/profile for seed %d.", results_dir, args.seed)
        return
    logger.info("E8 prediction analysis over %d models.", len(results))

    analysis = run_prediction_analysis(results)
    save_json(analysis, out_dir / "prediction_e8.json")

    # Console summary of the strongest correlations.
    sc = sorted(analysis["single_correlations"],
                key=lambda r: (r["p_value"] if r["p_value"] == r["p_value"] else 1.0))
    logger.info("Top correlations (predictor → target | rho | p_bh):")
    for rec in sc[:8]:
        logger.info("  %-16s → %-14s  rho=%+.3f  p_bh=%.3f%s",
                    rec["predictor"], rec["target"], rec["spearman_rho"],
                    rec["p_adjusted_bh"], "  *" if rec["significant_bh"] else "")
    for tgt, loo in analysis["loo_regression"].items():
        logger.info("  LOO %s: R²=%.3f (predictors=%s)", tgt, loo.get("loo_r2", float('nan')),
                    loo.get("predictors"))

    # E9 test-retest, if a second seed is available.
    if args.retest_seed is not None:
        rep2 = collect(results_dir, args.retest_seed)
        if rep2:
            tr = test_retest(results, rep2)
            save_json({"seed_a": args.seed, "seed_b": args.retest_seed,
                       "test_retest": tr.to_dict(orient="records")},
                      out_dir / "test_retest_e9.json")
            logger.info("E9 test-retest reliability (ceiling for E8):\n%s", tr.to_string(index=False))
        else:
            logger.warning("No results at retest seed %d — skipping E9.", args.retest_seed)

    logger.info("Analysis written to %s", out_dir)


if __name__ == "__main__":
    main()
