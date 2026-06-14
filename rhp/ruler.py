"""
RULER subset — behavioural targets beyond plain NIAH (proposal §3.3, E7).

A defensible, low-cost slice of RULER (Hsieh et al., 2024). Three generation-
based tasks, each scored by exact string presence in the model's output:

    multikey   N distinct "magic <word> is <value>" facts in a haystack; the
               query asks for ONE key's value (the others are hard distractors).
    multivalue one key carries several values; the query asks to list them all
               (credit only if every value appears).
    vartrack   a variable-assignment chain (X1 = <num>; X2 = X1; …); the query
               asks for the final variable's resolved value.

Filler comes from the shared PG-19 corpus (``src.corpus``); budgeting reuses the
token-aware trimming idea from ``NIAHEvaluator.build_prompt`` so the trailing
query is never truncated at long context.
"""

from __future__ import annotations

import gc
import logging
import random
import string
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.corpus import load_haystack_corpus

logger = logging.getLogger(__name__)

_WORDS = [
    "amber", "crimson", "cobalt", "ivory", "jade", "onyx", "scarlet", "azure",
    "olive", "maroon", "teal", "violet", "saffron", "indigo", "coral", "khaki",
]


def _rand_num(rng: random.Random, digits: int = 6) -> str:
    return "".join(rng.choice(string.digits) for _ in range(digits))


class RulerEvaluator:
    """Generation-based RULER-subset evaluator (E7)."""

    def __init__(self, model: Any, tokenizer: Any, config: dict, seed: int = 42) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.seed = seed
        self.model.eval()
        self._sentences: list[str] | None = None

    def _corpus(self) -> list[str]:
        if self._sentences is None:
            self._sentences = load_haystack_corpus(max_sentences=5000)
        return self._sentences

    # -- prompt building -----------------------------------------------------

    def _filler(self, token_budget: int, rng: random.Random) -> str:
        tok = self.tokenizer
        pool = self._corpus().copy()
        rng.shuffle(pool)
        parts, chars = [], 0
        for s in pool:
            parts.append(s)
            chars += len(s) + 1
            if chars >= token_budget * 6:
                break
        ids = tok(" ".join(parts), add_special_tokens=False)["input_ids"][:token_budget]
        return tok.decode(ids)

    def _embed(self, facts: list[str], context_length: int, rng: random.Random) -> str:
        """Spread ``facts`` at random positions through a length-budgeted filler."""
        tok = self.tokenizer
        reserve = 64 + sum(len(tok(f, add_special_tokens=False)["input_ids"]) for f in facts)
        filler = self._filler(max(16, context_length - reserve), rng)
        words = filler.split()
        for fact in facts:
            idx = rng.randint(0, max(0, len(words)))
            words.insert(idx, fact)
        return " ".join(words)

    @torch.no_grad()
    def _generate(self, prompt: str, context_length: int, max_new_tokens: int = 32) -> str:
        device = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            next(self.model.parameters()),
        ).device
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length + 512)
        input_ids = enc["input_ids"].to(device)
        try:
            out = self.model.generate(
                input_ids,
                attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            del input_ids
            if "out" in locals():
                del out
            torch.cuda.empty_cache()
        return text

    # -- the three tasks -----------------------------------------------------

    def multikey(self, context_length: int, n_keys: int, n_samples: int, seed: int) -> float:
        rng = random.Random(seed)
        correct = 0
        for _ in range(n_samples):
            keys = rng.sample(_WORDS, k=min(n_keys, len(_WORDS)))
            vals = {k: _rand_num(rng) for k in keys}
            facts = [f"The magic {k} number is {vals[k]}." for k in keys]
            target = rng.choice(keys)
            ctx = self._embed(facts, context_length, rng)
            prompt = (f"{ctx}\n\nWhat is the magic {target} number? "
                      f"Answer with the number only.\n\nAnswer:")
            if vals[target] in self._generate(prompt, context_length):
                correct += 1
        return correct / n_samples

    def multivalue(self, context_length: int, n_values: int, n_samples: int, seed: int) -> float:
        rng = random.Random(seed + 1)
        correct = 0
        for _ in range(n_samples):
            key = rng.choice(_WORDS)
            values = [_rand_num(rng) for _ in range(n_values)]
            facts = [f"One magic {key} number is {v}." for v in values]
            ctx = self._embed(facts, context_length, rng)
            prompt = (f"{ctx}\n\nList all the magic {key} numbers, separated by commas."
                      f"\n\nAnswer:")
            text = self._generate(prompt, context_length, max_new_tokens=48)
            if all(v in text for v in values):
                correct += 1
        return correct / n_samples

    def vartrack(self, context_length: int, chain_len: int, n_samples: int, seed: int) -> float:
        rng = random.Random(seed + 2)
        correct = 0
        for _ in range(n_samples):
            root = _rand_num(rng)
            names = [f"VAR_{rng.choice(string.ascii_uppercase)}{i}" for i in range(chain_len)]
            facts = [f"{names[0]} = {root}."]
            for i in range(1, chain_len):
                facts.append(f"{names[i]} = {names[i-1]}.")
            rng.shuffle(facts)
            ctx = self._embed(facts, context_length, rng)
            prompt = (f"{ctx}\n\nResolve the chain of assignments. What is the numeric "
                      f"value of {names[-1]}? Answer with the number only.\n\nAnswer:")
            if root in self._generate(prompt, context_length):
                correct += 1
        return correct / n_samples

    # -- driver --------------------------------------------------------------

    def run(
        self,
        tasks: list[str],
        context_lengths: list[int],
        n_samples: int,
        seeds: list[int],
        n_keys: int = 4,
        n_values: int = 3,
        chain_len: int = 4,
    ) -> dict:
        """Run the requested RULER tasks; return per-task, per-context accuracy."""
        out: dict = {"tasks": {}, "n_samples": n_samples, "seeds": seeds}
        for task in tasks:
            out["tasks"][task] = {}
            for ctx in context_lengths:
                per_seed = []
                for seed in seeds:
                    if task == "multikey":
                        acc = self.multikey(ctx, n_keys, n_samples, seed)
                    elif task == "multivalue":
                        acc = self.multivalue(ctx, n_values, n_samples, seed)
                    elif task == "vartrack":
                        acc = self.vartrack(ctx, chain_len, n_samples, seed)
                    else:
                        raise ValueError(f"Unknown RULER task '{task}'")
                    per_seed.append(acc)
                    gc.collect(); torch.cuda.empty_cache()
                out["tasks"][task][str(ctx)] = {
                    "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                    "per_seed": per_seed,
                }
                logger.info("E7 RULER %s @ %d: %.3f ± %.3f", task, ctx,
                            out["tasks"][task][str(ctx)]["mean"],
                            out["tasks"][task][str(ctx)]["std"])
        # one scalar per task averaged over contexts, for the E8 table
        out["task_means"] = {
            t: float(np.mean([out["tasks"][t][str(c)]["mean"] for c in context_lengths]))
            for t in tasks
        }
        return out
