"""
Locate the Part-1 repository so its proven ``src/`` package can be imported.

Part 2 *inherits* the Part-1 infrastructure verbatim (detector, knockout,
frequency-patch, statistics — see the proposal §2.3). Rather than fork that
code, we add the Part-1 repo to ``sys.path`` and ``import src.*`` directly, so a
bug fix in one place benefits both papers and the two stay bit-for-bit
compatible.

Resolution order for the Part-1 repo:
    1. explicit argument to ``ensure_part1_on_path``
    2. ``$RHP_PART1_REPO`` environment variable
    3. sibling-directory guesses next to this project

In Colab the path is explicit (the notebook clones Part 1 to a known location),
so this module mostly matters for local runs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Folder name of the Part-1 repository (the first paper).
PART1_DIRNAME = "Does-RoPE-Prevent-or-Degrade-Retrieval-Heads-A-Mechanistic-Analysis-Across-Model-Families"

_RESOLVED: str | None = None


def _looks_like_part1(path: Path) -> bool:
    return (path / "src" / "retrieval_head_detector.py").exists()


def ensure_part1_on_path(explicit: str | os.PathLike | None = None) -> str:
    """
    Make the Part-1 ``src`` package importable; return the resolved repo path.

    Idempotent: the first successful resolution is cached and reused.

    Raises:
        ImportError: if the Part-1 repo cannot be located. The message lists the
            two ways to fix it (env var or argument).
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("RHP_PART1_REPO")
    if env:
        candidates.append(Path(env))

    here = Path(__file__).resolve()
    # Guess: sibling of the project root, or one level further up.
    for base in (here.parents[2], here.parents[3] if len(here.parents) > 3 else here.parents[2]):
        candidates.append(base / PART1_DIRNAME)

    for cand in candidates:
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if _looks_like_part1(cand):
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            _RESOLVED = str(cand)
            logger.info("Part-1 repo resolved at %s", cand)
            return _RESOLVED

    raise ImportError(
        "Could not locate the Part-1 repo (the folder named "
        f"'{PART1_DIRNAME}' containing src/retrieval_head_detector.py).\n"
        "Fix one of:\n"
        "  • set the RHP_PART1_REPO environment variable to that folder, or\n"
        "  • pass --part1-repo PATH to the script, or\n"
        "  • place this project as a sibling of the Part-1 repo.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )
