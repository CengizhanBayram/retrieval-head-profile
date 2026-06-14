"""
Retrieval-Head Profile (Part 2).

A training-free diagnostic — the *retrieval-head profile* — for long-context
ability and its inheritance across model transformations (instruction tuning,
quantization, distillation). All experiments are inference-only and run on a
single 24 GB GPU.

This package builds on the Part-1 repository ("Does RoPE Prevent or Degrade
Retrieval Heads?"). Importing ``rhp`` makes a best-effort attempt to put the
Part-1 ``src`` package on ``sys.path`` (see ``rhp._paths``); submodules then
``import src.*`` for the inherited detector / patching / statistics code.
"""

from __future__ import annotations

import logging as _logging

from rhp._paths import ensure_part1_on_path

# Best-effort path setup at import time. If it fails here we stay silent and let
# the first ``import src.*`` raise a clear ImportError — callers (scripts /
# notebooks) can also call ensure_part1_on_path(explicit=...) first.
try:
    ensure_part1_on_path()
except ImportError:  # pragma: no cover - resolved later by the caller
    _logging.getLogger(__name__).debug(
        "Part-1 repo not auto-resolved at import; call ensure_part1_on_path(path)."
    )

__all__ = ["ensure_part1_on_path"]
__version__ = "0.1.0"
