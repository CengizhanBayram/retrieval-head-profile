"""
Model-panel and lineage helpers (proposal §3.2, §3.4).

Thin convenience layer over ``configs/panel.yaml``: load the config, normalise
per-model dicts (apply defaults, merge ``quant_models``) into the schema that
the inherited ``src.model_loader.load_model`` expects, and expand the
inheritance lineages into ordered ring-pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import yaml

logger = logging.getLogger(__name__)


def load_panel(config_path: str | Path) -> dict:
    """Load and return the raw panel config dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_model_keys(config: dict, *, include_quant: bool = True) -> list[str]:
    """Return every model key (panel + optionally the quantized rings)."""
    keys = list(config.get("models", {}).keys())
    if include_quant:
        keys += list(config.get("quant_models", {}).keys())
    return keys


def model_cfg(config: dict, key: str) -> dict:
    """
    Return the fully-resolved per-model config for ``key``.

    Merges (in order): top-level ``defaults`` → the model entry. Works for both
    ``models`` and ``quant_models``. The result is exactly what
    ``src.model_loader.load_model(model_cfg, key)`` consumes, plus the extra
    bookkeeping fields (``family``, ``tier``, ``quant``) which the loader
    ignores.
    """
    defaults = dict(config.get("defaults", {}))
    pool = {**config.get("models", {}), **config.get("quant_models", {})}
    if key not in pool:
        raise KeyError(
            f"Unknown model key '{key}'. Known: {sorted(pool.keys())}"
        )
    merged = {**defaults, **pool[key]}
    # Quantized rings ignore the 8-bit default — they bring their own scheme.
    if merged.get("quant"):
        merged["load_in_8bit"] = False
    return merged


def models_by_tier(config: dict, tier: str) -> list[str]:
    """Return model keys whose ``tier`` matches (e.g. 'core')."""
    return [
        k for k, v in config.get("models", {}).items() if v.get("tier") == tier
    ]


def core_models(config: dict) -> list[str]:
    """The 5 'core' models that get 3 seeds + test-retest (proposal §5)."""
    return models_by_tier(config, "core")


def family_of(config: dict, key: str) -> str:
    """Family label for a model key (confound axis for E8)."""
    return model_cfg(config, key).get("family", "unknown")


# ---------------------------------------------------------------------------
# Inheritance lineages
# ---------------------------------------------------------------------------

def lineage_chain(config: dict, lineage: str) -> list[str]:
    """Ordered list of ring keys for a lineage (proposal §3.4)."""
    lz = config.get("lineages", {})
    if lineage not in lz:
        raise KeyError(f"Unknown lineage '{lineage}'. Known: {sorted(lz.keys())}")
    return list(lz[lineage]["chain"])


def lineage_ring_pairs(config: dict, lineage: str) -> list[tuple[str, str]]:
    """
    Adjacent ring-pairs for a lineage, e.g.
    ``base→instruct→awq4`` ⇒ ``[(base, instruct), (instruct, awq4)]``.

    These are the pairs the inheritance package (E10–E13) compares.
    """
    chain = lineage_chain(config, lineage)
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


def all_ring_pairs(config: dict) -> Iterator[tuple[str, str, str]]:
    """Yield ``(lineage, parent_key, child_key)`` over every lineage."""
    for lineage in config.get("lineages", {}):
        for parent, child in lineage_ring_pairs(config, lineage):
            yield lineage, parent, child


def lineage_sibling(config: dict, lineage: str) -> str | None:
    """Optional distillation sibling for a lineage (e.g. Qwen2.5-3B-Instruct)."""
    return config.get("lineages", {}).get(lineage, {}).get("sibling")
