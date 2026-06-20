"""
Haystack-corpus fix (script-free).

The inherited ``src.corpus`` loads PG-19 via ``load_dataset("pg19",
trust_remote_code=True)``. Recent ``datasets`` removed loading-script support, so
that call now FAILS and the code silently falls back to a tiny 50-sentence list
repeated ×40 — a highly repetitive haystack that is not defensible for the paper
(it would also undermine the R7 haystack-source control).

This installs a real, diverse, **script-free** corpus (WikiText-103, Parquet)
into ``src.corpus._CORPUS_CACHE`` so every downstream caller (both detectors and
the behavioural evaluators) uses it consistently. Call it once at session start,
*before* any detection/behaviour run.

NOTE: changing the corpus changes the haystack, so a run that uses this must NOT
be mixed with results produced under the old fallback — re-run those for
consistency.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_haystack_corpus(max_sentences: int = 6000, source: str = "wikitext") -> int:
    """
    Load a script-free corpus and inject it into ``src.corpus._CORPUS_CACHE``.

    Returns the number of sentences installed (0 if it fell back, leaving the
    inherited loader's own fallback in place).
    """
    import src.corpus as corpus

    specs = {
        "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        "c4": ("allenai/c4", "en"),
    }
    if source not in specs:
        raise ValueError(f"Unknown source '{source}'. Known: {list(specs)}")
    repo, cfg = specs[source]

    try:
        from datasets import load_dataset

        ds = load_dataset(repo, cfg, split="train", streaming=True)
        sentences: list[str] = []
        for example in ds:
            text = example.get("text", "")
            for sent in text.split(". "):
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    sentences.append(sent + ".")
                if len(sentences) >= max_sentences:
                    break
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= 500:
            corpus._CORPUS_CACHE = sentences
            logger.info("Installed %d-sentence %s haystack corpus (script-free).",
                        len(sentences), source)
            return len(sentences)
        logger.warning("%s yielded only %d sentences; keeping inherited fallback.",
                       source, len(sentences))
    except Exception as exc:
        logger.warning("Could not install %s corpus (%s); keeping inherited fallback.",
                       source, exc)
    return 0
