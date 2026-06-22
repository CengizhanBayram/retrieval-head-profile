"""
Block C — inheritance chains (E10–E13, with E14/E15 read-outs). CPU only.

Consumes the per-model JSONs already written by ``run_profile.py`` and
``run_behavior.py`` and, for every adjacent ring-pair in each lineage, runs the
standard inheritance package: identity (E10), function (E11), frequency (E12),
behaviour bridge (E13), plus the RQ4 localisation read-out (E15). The 4-bit
rings double as the E14 quantization ablation (AWQ vs GPTQ identity Jaccard).

Run this AFTER the lineage's models have profiles + behaviours on disk.

Examples:
    python scripts/run_inheritance.py --lineage qwen
    python scripts/run_inheritance.py --lineage all
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
logger = logging.getLogger("run_inheritance")


def parse_args():
    ap = argparse.ArgumentParser(description="Block C inheritance analysis (E10–E15).")
    ap.add_argument("--lineage", default="all", help="lineage name or 'all'.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_model_result(results_dir: Path, model: str, seed: int) -> dict | None:
    """Merge a model's profile JSON with its behaviour JSON (if present)."""
    pf = results_dir / "profile" / f"{model}_seed{seed}.json"
    if not pf.exists():
        logger.warning("Missing profile for %s (%s).", model, pf)
        return None
    with open(pf, encoding="utf-8") as f:
        res = json.load(f)
    bf = results_dir / "behavior" / f"{model}_seed{seed}.json"
    if bf.exists():
        with open(bf, encoding="utf-8") as f:
            beh = json.load(f)
        res["behavior"] = beh.get("behavior", {})
    uf = results_dir / "utility" / f"{model}_seed{seed}.json"
    if uf.exists():
        with open(uf, encoding="utf-8") as f:
            res["utility"] = json.load(f).get("utility", {})
    return res


def main():
    args = parse_args()
    from scripts._common import bootstrap, save_json
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel, lineage_ring_pairs, lineage_sibling, lineage_chain
    from rhp.inheritance import (
        compare_ring, compare_identity, compare_invariant, localize_recall_change,
        quant_ablation,
    )

    config = load_panel(args.config)
    results_dir = Path(args.results_dir or config["output"]["results_dir"])
    out_dir = results_dir / "inheritance"
    out_dir.mkdir(parents=True, exist_ok=True)

    lineages = list(config["lineages"].keys()) if args.lineage == "all" else [args.lineage]
    for lineage in lineages:
        rings = []
        for parent_key, child_key in lineage_ring_pairs(config, lineage):
            parent = load_model_result(results_dir, parent_key, args.seed)
            child = load_model_result(results_dir, child_key, args.seed)
            if parent is None or child is None:
                logger.warning("[%s] skip ring %s→%s (missing result).", lineage, parent_key, child_key)
                continue
            ring = compare_ring(parent, child, lineage=lineage)
            ring["E15_localization"] = localize_recall_change(ring)
            rings.append(ring)
            logger.info("[%s] %s→%s: copy-Jaccard=%.3f Δfreq_com=%.3f verdict=%s",
                        lineage, parent_key, child_key,
                        ring["E10_identity"]["copy"]["jaccard"],
                        ring["E12_frequency"]["delta_freq_com"],
                        ring["E15_localization"]["verdict"])

        out = {"lineage": lineage, "description": config["lineages"][lineage].get("description"),
               "rings": rings}

        # E14 — quantization ablation: the instruct reference vs its AWQ/GPTQ
        # 4-bit rings (three-level template). Only fires when the chain carries
        # quantized rings (e.g. the Qwen lineage).
        quant_keys = set(config.get("quant_models", {}).keys())
        chain = lineage_chain(config, lineage)
        rings_quant = [k for k in chain if k in quant_keys]
        if rings_quant:
            first_q = chain.index(rings_quant[0])
            ref_key = chain[first_q - 1] if first_q > 0 else chain[0]
            ref = load_model_result(results_dir, ref_key, args.seed)
            awq = next((load_model_result(results_dir, k, args.seed)
                        for k in rings_quant if "awq" in k), None)
            gptq = next((load_model_result(results_dir, k, args.seed)
                         for k in rings_quant if "gptq" in k), None)
            if ref is not None:
                out["E14_quant_ablation"] = quant_ablation(ref, awq, gptq)
                logger.info("[%s] E14 quant ablation vs %s done.", lineage, ref_key)

        # Distillation sibling (E10 identity only — architectures differ).
        sib = lineage_sibling(config, lineage)
        if sib:
            chain = config["lineages"][lineage]["chain"]
            base = load_model_result(results_dir, chain[0], args.seed)
            sib_res = load_model_result(results_dir, sib, args.seed)
            if base and sib_res:
                same_arch = (base.get("n_layers") == sib_res.get("n_layers")
                             and base.get("n_heads") == sib_res.get("n_heads"))
                if same_arch:
                    out["sibling"] = {"base": chain[0], "sibling": sib,
                                      "same_architecture": True,
                                      "identity": compare_identity(base, sib_res)}
                else:
                    # Distillation sibling at a DIFFERENT size: head-set Jaccard is
                    # not comparable across architectures (it would be a ~0
                    # artifact). Report architecture-invariant axes instead.
                    out["sibling"] = {"base": chain[0], "sibling": sib,
                                      "same_architecture": False,
                                      "invariant": compare_invariant(base, sib_res)}
                    logger.info("[%s] sibling %s is cross-architecture -> invariant axes only.",
                                lineage, sib)

        save_json(out, out_dir / f"{lineage}.json")
        logger.info("[%s] saved %d rings → %s", lineage, len(rings), out_dir / f"{lineage}.json")


if __name__ == "__main__":
    main()
