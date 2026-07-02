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


# Quantized rings are load-time variants of an instruct model, not independent
# training runs, so they are excluded from the RQ2 panel by default (the canonical
# prediction_e8.json is over the 18 independent base+instruct models). Pass
# --include-rings to reproduce the prediction_e8_withrings.json comparison.
RING_SUFFIXES = ("awq4", "gptq4", "bnb4")


def parse_args():
    ap = argparse.ArgumentParser(description="E8 prediction + E9 test-retest.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retest-seed", type=int, default=None,
                    help="Second seed for the E9 test-retest (e.g. 123).")
    ap.add_argument("--include-rings", action="store_true",
                    help="Include the quantized rings (writes prediction_e8_withrings.json). "
                         "Default excludes them -> the canonical prediction_e8.json.")
    return ap.parse_args()


def collect(results_dir: Path, seed: int, include_rings: bool = False) -> list[dict]:
    """Load + merge profile/behaviour results for every model at one seed.

    By default the quantized rings (``*_awq4/gptq4/bnb4``) are skipped: they are
    not independent training runs, so they do not belong in the RQ2 panel.
    """
    merged = []
    prof_dir = results_dir / "profile"
    if not prof_dir.exists():
        return merged
    for pf in sorted(prof_dir.glob(f"*_seed{seed}.json")):
        with open(pf, encoding="utf-8") as f:
            res = json.load(f)
        model = res.get("model")
        if not include_rings and any(s in model for s in RING_SUFFIXES):
            continue
        bf = results_dir / "behavior" / f"{model}_seed{seed}.json"
        if bf.exists():
            with open(bf, encoding="utf-8") as f:
                res["behavior"] = json.load(f).get("behavior", {})
        uf = results_dir / "utility" / f"{model}_seed{seed}.json"
        if uf.exists():
            with open(uf, encoding="utf-8") as f:
                res["utility"] = json.load(f).get("utility", {})
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

    results = collect(results_dir, args.seed, include_rings=args.include_rings)
    if not results:
        logger.error("No profile results found under %s/profile for seed %d.", results_dir, args.seed)
        return
    out_name = "prediction_e8_withrings.json" if args.include_rings else "prediction_e8.json"
    logger.info("E8 prediction analysis over %d models (%s rings) -> %s.",
                len(results), "with" if args.include_rings else "without", out_name)

    analysis = run_prediction_analysis(results)
    save_json(analysis, out_dir / out_name)

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

    # E9 reliability, if a second seed is available.
    if args.retest_seed is not None:
        from rhp.prediction import within_model_reliability
        rep2 = collect(results_dir, args.retest_seed)
        rep2_by_model = {r["model"]: r for r in rep2}
        # (a) Per-model head-set reliability R_self — the artifact the inheritance
        # thresholds actually use. Computable per model (no n>=4 requirement), so
        # it is the trustworthy R_self source (test_retest below needs >=4 paired
        # models and is NaN until then).
        rself = []
        for r in results:
            other = rep2_by_model.get(r["model"])
            if other is not None:
                rself.append(within_model_reliability(r, other))
        if rself:
            save_json({"seed_a": args.seed, "seed_b": args.retest_seed,
                       "reliability": rself},
                      out_dir / "reliability_e9.json")
            import numpy as _np
            cj = [x["copy_jaccard"] for x in rself if x["copy_jaccard"] == x["copy_jaccard"]]
            logger.info("E9 R_self (copy-Jaccard) over %d models: median=%.3f  values=%s",
                        len(rself), float(_np.median(cj)) if cj else float("nan"),
                        {x["model"]: round(x["copy_jaccard"], 3) for x in rself})
        # (b) Cross-panel per-metric test-retest (needs >=4 paired models).
        if rep2:
            tr = test_retest(results, rep2)
            save_json({"seed_a": args.seed, "seed_b": args.retest_seed,
                       "test_retest": tr.to_dict(orient="records")},
                      out_dir / "test_retest_e9.json")
            n_paired = len(rep2_by_model.keys() & {r["model"] for r in results})
            if n_paired < 4:
                logger.warning("test_retest is NaN: only %d paired models (need >=4). "
                               "Use reliability_e9.json (R_self) instead until more "
                               "2-seed models exist.", n_paired)
        else:
            logger.warning("No results at retest seed %d — skipping E9.", args.retest_seed)

    logger.info("Analysis written to %s", out_dir)


if __name__ == "__main__":
    main()
