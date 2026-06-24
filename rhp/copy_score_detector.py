"""
Second retrieval-head detector: a teacher-forced copy score (proposal §3.1).

The Part-1 detector (``src.retrieval_head_detector.RetrievalHeadDetector``) is a
single-pass *attention-argmax proxy*: it asks whether a head's last-token
attention argmax lands on the needle. That is cheap but coarse. This module adds
the complementary detector the proposal requires — a **teacher-forced copy
score** much closer to the original Wu et al. (2025) retrieval score:

    For a NIAH sample whose answer is a multi-token value copied verbatim from
    the context, teacher-force the gold answer and, at each answer position,
    check whether the head attends to the *exact source token it must copy*.
    The head's copy score is the fraction of copy events it gets right,
    averaged over samples.

Running both detectors per model lets us report ``detector_agreement`` (the
Jaccard of the two head sets) as a profile-reliability metric, and to fall back
on the copy score when the two disagree (Part-1 jurisprudence, proposal R2).

Key construction trick (avoids the repeated-token ambiguity that plagues a
value-matching copy score): the answer is teacher-forced with the *same token
ids* that the value has inside the prompt, so answer token ``j`` corresponds to
needle-value token ``v_start + j`` by position. The "correct source" is that
exact index, not "any token equal to the generated one".
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

from src.corpus import build_haystack, load_haystack_corpus
from src.retrieval_head_detector import find_subsequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Copy-score sample generation (tokenizer-independent specs)
# ---------------------------------------------------------------------------

def _make_value(rng: random.Random, n_tokens: int) -> str:
    """A space-separated value whose pieces tokenize to distinct tokens.

    Space-separated single alnum characters reliably tokenize to one token each
    for BPE/SentencePiece vocabularies, giving ``n_tokens`` clean copy events.
    """
    alphabet = string.ascii_uppercase + string.digits
    return " ".join(rng.choice(alphabet) for _ in range(n_tokens))


def generate_copy_specs(
    seed: int,
    n_samples: int,
    context_lengths: list[int],
    needle_positions: list[float],
    value_tokens: int = 8,
    max_corpus_sentences: int = 5000,
) -> list[dict]:
    """
    Generate tokenizer-independent copy-score specs (deterministic given seed).

    Each spec carries a multi-token ``value`` embedded in a distinctive needle
    sentence, so a downstream tokenizer yields several unambiguous copy events.
    """
    rng = random.Random(seed)
    sentences = load_haystack_corpus(max_sentences=max_corpus_sentences)
    combos = [(cl, pos) for cl in context_lengths for pos in needle_positions]
    per_combo = max(1, n_samples // len(combos))

    specs: list[dict] = []
    spec_id = 0
    for context_length, needle_position in combos:
        for _ in range(per_combo):
            value = _make_value(rng, value_tokens)
            needle = f"The hidden access code is {value}."
            query = "What is the hidden access code? Answer with the code only."
            haystack = build_haystack(context_length, sentences, rng)
            words = haystack.split()
            insert_idx = max(0, min(int(len(words) * needle_position), len(words) - 1))
            words.insert(insert_idx, needle)
            prompt = f"{' '.join(words)}\n\n{query}\n\nAnswer:"
            specs.append({
                "spec_id": spec_id,
                "prompt": prompt,
                "value": value,
                "context_length": context_length,
                "needle_position": needle_position,
            })
            spec_id += 1
    return specs


def _locate_value(prompt: str, value: str, context_length: int, tokenizer) -> dict | None:
    """Tokenize ``prompt`` and locate the value token span (the copy target)."""
    enc = tokenizer(
        prompt, return_tensors=None, truncation=True, max_length=context_length + 96
    )
    prompt_ids: list[int] = enc["input_ids"]
    bos = tokenizer.bos_token_id
    # Try a few surface forms; the value sits after "is " inside the needle, so a
    # leading space is the usual correct variant.
    for prefix in (" ", "", "is "):
        cand = tokenizer(prefix + value, add_special_tokens=False)["input_ids"]
        if bos is not None and cand and cand[0] == bos:
            cand = cand[1:]
        # Strip the leading "is " token(s) we added only to coax the right merge.
        if prefix == "is ":
            is_ids = tokenizer("is", add_special_tokens=False)["input_ids"]
            if cand[: len(is_ids)] == is_ids:
                cand = cand[len(is_ids):]
        span = find_subsequence(prompt_ids, cand)
        if span is not None:
            return {
                "prompt_ids": prompt_ids,
                "value_token_ids": cand,
                "value_start_idx": span[0],
                "value_end_idx": span[1],
            }
    return None


def build_copy_samples_from_specs(specs: list[dict], tokenizer) -> tuple[list[dict], list[int]]:
    """Tokenize copy specs for one model. Returns (samples, valid_spec_ids)."""
    samples: list[dict] = []
    valid: list[int] = []
    for spec in specs:
        loc = _locate_value(spec["prompt"], spec["value"], spec["context_length"], tokenizer)
        if loc is None or loc["value_end_idx"] - loc["value_start_idx"] < 2:
            continue
        samples.append({
            "prompt_ids": loc["prompt_ids"],
            "value": spec["value"],
            "value_token_ids": loc["value_token_ids"],
            "value_start_idx": loc["value_start_idx"],
            "value_end_idx": loc["value_end_idx"],
            "context_length": spec["context_length"],
            "needle_position": spec["needle_position"],
            "actual_token_length": len(loc["prompt_ids"]),
            "spec_id": spec["spec_id"],
        })
        valid.append(spec["spec_id"])
    return samples, valid


class CopyScoreDetector:
    """
    Teacher-forced copy-score detector (the second detector, proposal §3.1).

    Usage mirrors ``RetrievalHeadDetector``:
        det = CopyScoreDetector(model, tokenizer, config, score_threshold=0.1)
        samples = det.prepare_samples(generate_copy_specs(seed, ...))
        scores  = det.score_heads(samples)          # (n_layers, n_heads)
        heads   = det.get_retrieval_heads(scores)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict,
        score_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.score_threshold = score_threshold
        self.seed = seed
        self.model.eval()

    # -- model introspection (kept identical to the Part-1 detector) ---------

    def _get_device(self) -> torch.device:
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_layers(self):
        model_inner = getattr(self.model, "model", self.model)
        for attr in ("layers", "transformer.blocks", "transformer.h", "decoder.layers"):
            obj = model_inner
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError("Cannot locate layer list in model architecture.")

    def _get_n_layers_heads(self) -> tuple[int, int]:
        return (
            self.model.config.num_hidden_layers,
            self.model.config.num_attention_heads,
        )

    # -- sample prep ---------------------------------------------------------

    def prepare_samples(self, specs: list[dict]) -> list[dict]:
        samples, valid = build_copy_samples_from_specs(specs, self.tokenizer)
        dropped = len(specs) - len(valid)
        if dropped:
            logger.warning(
                "CopyScoreDetector.prepare_samples: %d/%d specs failed to locate a "
                "≥2-token value for this tokenizer.", dropped, len(specs),
            )
        return samples

    def generate_samples(
        self,
        n_samples: int,
        context_lengths: list[int],
        needle_positions: list[float],
        value_tokens: int = 8,
    ) -> list[dict]:
        specs = generate_copy_specs(
            self.seed, n_samples, context_lengths, needle_positions, value_tokens
        )
        return self.prepare_samples(specs)

    # -- scoring -------------------------------------------------------------

    @torch.no_grad()
    def score_heads(self, samples: list[dict]) -> np.ndarray:
        """
        Teacher-forced copy score for every head: (n_layers, n_heads) in [0, 1].

        For each sample we feed ``prompt_ids + value_token_ids`` in one pass and,
        at the position that *generates* each value token, check whether the
        head's attention argmax lands on the exact source position the token was
        copied from. Per-sample copy events are pooled across the panel of
        samples; the score is hits / copy-events.
        """
        device = self._get_device()
        n_layers, n_heads = self._get_n_layers_heads()
        layers = self._get_layers()

        hit_counts = np.zeros((n_layers, n_heads), dtype=np.int64)
        event_counts = np.zeros((n_layers, n_heads), dtype=np.int64)
        attn_seen = [False]

        for i, sample in enumerate(tqdm(samples, desc="Copy-score heads")):
            prompt_ids = sample["prompt_ids"]
            value_ids = sample["value_token_ids"]
            v_start = sample["value_start_idx"]
            P, m = len(prompt_ids), len(value_ids)
            if m < 2:
                continue
            full_ids = prompt_ids + value_ids
            # Positions that *predict* each answer token j: (P-1)+j for j in 0..m-1.
            gen_idx = list(range(P - 1, P - 1 + m))
            # Gold source positions inside the needle for each answer token.
            gold_src = [v_start + j for j in range(m)]

            captured: dict[int, torch.Tensor] = {}
            hooks = []

            def _make_hook(layer_idx: int):
                def hook(module, input_, output):
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        attn_w = output[1]
                        # (batch, heads, seq, seq) under eager; newer torch routes
                        # some models (Gemma-2 w/ output_attentions) to flex_attention
                        # which drops the batch dim -> (heads, seq, seq). Handle both.
                        if attn_w.dim() == 4:
                            rows = attn_w[0, :, gen_idx, :].float().cpu()   # (heads, m, seq)
                        elif attn_w.dim() == 3:
                            rows = attn_w[:, gen_idx, :].float().cpu()       # (heads, m, seq)
                        else:
                            return output
                        captured[layer_idx] = rows
                        attn_seen[0] = True
                        return (output[0], None) + output[2:]
                    return output
                return hook

            try:
                for layer_idx, layer in enumerate(layers):
                    hooks.append(layer.self_attn.register_forward_hook(_make_hook(layer_idx)))
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                self.model(input_ids=input_ids, output_attentions=True, use_cache=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("OOM on copy sample %d (len=%d); skipping.", i, len(full_ids))
                    gc.collect(); torch.cuda.empty_cache(); captured.clear()
                else:
                    raise
            finally:
                for h in hooks:
                    h.remove()
                if "input_ids" in dir():
                    del input_ids

            for layer_idx, rows in captured.items():
                # rows: (heads_captured, m, seq)
                hc = rows.shape[0]
                argmax_keys = rows.argmax(dim=-1).numpy()  # (heads_captured, m)
                if hc == 1 and n_heads > 1:                 # MQA fallback
                    argmax_keys = np.repeat(argmax_keys, n_heads, axis=0)
                    hc = n_heads
                for h in range(min(hc, n_heads)):
                    for j in range(m):
                        event_counts[layer_idx, h] += 1
                        if int(argmax_keys[h, j]) == gold_src[j]:
                            hit_counts[layer_idx, h] += 1
            captured.clear()
            if i % 10 == 9:
                gc.collect(); torch.cuda.empty_cache()

        if not attn_seen[0]:
            logger.error(
                "No attention weights captured for the copy score (output_attentions "
                "unsupported under this config). All copy scores will be 0."
            )
        with np.errstate(invalid="ignore"):
            scores = np.where(event_counts > 0, hit_counts / event_counts, 0.0)
        logger.info(
            "Copy-score complete. max=%.3f, heads ≥ %.2f: %d",
            scores.max(), self.score_threshold, int((scores >= self.score_threshold).sum()),
        )
        return scores.astype(np.float32)

    def get_retrieval_heads(self, scores: np.ndarray) -> list[tuple[int, int]]:
        ls, hs = np.where(scores >= self.score_threshold)
        heads = sorted(zip(ls.tolist(), hs.tolist()))
        logger.info("Copy detector found %d retrieval heads (τ=%.2f).", len(heads), self.score_threshold)
        return heads
