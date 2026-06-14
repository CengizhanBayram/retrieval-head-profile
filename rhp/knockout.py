"""
Knockout double-dissociation — causality confirmation (proposal §3.1, E3).

The profile must be a *causal* map, not a correlational one. For each model:

    baseline       no mask
    mask_retrieval zero ALL detected retrieval heads' outputs (before o_proj)
    mask_random    zero an equal number of random NON-retrieval heads (C1)

A genuine retrieval-head set produces ``knockout_drop = baseline − mask_retrieval``
much larger than ``random_drop = baseline − mask_random``. Paired significance is
an exact McNemar test on the same samples (retrieval-mask vs random-mask),
inherited verbatim from Part 1's ``_mcnemar_exact``.

Masking reuses Part 1's mechanism: a forward-pre-hook on ``o_proj`` zeroes the
concatenated head slice before the output projection (same as
``NIAHEvaluator.evaluate_with_head_masking``), but here we score per-sample so a
McNemar/bootstrap is possible.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager

import numpy as np
import torch

from src.activation_patching import _mcnemar_exact

logger = logging.getLogger(__name__)


class KnockoutEvaluator:
    """Per-sample NIAH accuracy under attention-head masking (E3)."""

    def __init__(self, model, tokenizer, config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    def _get_layers(self):
        inner = getattr(self.model, "model", self.model)
        for attr in ("layers", "transformer.blocks", "transformer.h", "decoder.layers"):
            obj = inner
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError("Cannot locate layer list in model architecture.")

    def _head_dim(self) -> int:
        cfg = self.model.config
        if hasattr(cfg, "head_dim") and cfg.head_dim:
            return cfg.head_dim
        return cfg.hidden_size // cfg.num_attention_heads

    @contextmanager
    def _mask(self, heads: list[tuple[int, int]]):
        """Zero the listed heads' contribution just before each o_proj."""
        head_dim = self._head_dim()
        layers = self._get_layers()
        by_layer: dict[int, list[int]] = {}
        for (l, h) in heads:
            by_layer.setdefault(l, []).append(h)

        def _make_pre_hook(hidxs: list[int]):
            def hook(module, input_):
                if not input_:
                    return input_
                inp = input_[0].clone()
                width = inp.shape[-1]
                for h in hidxs:
                    s, e = h * head_dim, (h + 1) * head_dim
                    if s < width:
                        inp[:, :, s:min(e, width)] = 0.0
                return (inp,) + input_[1:]
            return hook

        hooks = []
        try:
            for l, hidxs in by_layer.items():
                hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(_make_pre_hook(hidxs)))
            yield
        finally:
            for h in hooks:
                h.remove()

    @torch.no_grad()
    def _accuracy(self, samples: list[dict], heads: list[tuple[int, int]] | None) -> list[int]:
        """Per-sample 1/0 correctness; OOM counts as 0 to keep conditions paired."""
        device = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            next(self.model.parameters()),
        ).device
        per_sample: list[int] = []
        ctx = self._mask(heads) if heads else _null()
        with ctx:
            for sample in samples:
                ok = 0
                input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)
                try:
                    out = self.model.generate(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                    gen = self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
                    ok = 1 if sample["code"] in gen else 0
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        gc.collect(); torch.cuda.empty_cache()
                    else:
                        raise
                finally:
                    del input_ids
                    if "out" in locals():
                        del out
                    torch.cuda.empty_cache()
                per_sample.append(ok)
        return per_sample

    def run(
        self,
        samples: list[dict],
        retrieval_heads: list[tuple[int, int]],
        n_heads: int,
        n_layers: int,
        random_seeds: list[int] | None = None,
    ) -> dict:
        """
        Run the double dissociation and return drops + paired significance (E3).

        ``samples`` must carry a ``code`` field (use the argmax-detector samples
        from ``RetrievalHeadDetector.generate_niah_samples``).
        """
        random_seeds = random_seeds or [0, 1, 2, 3, 4]
        retrieval_set = set(retrieval_heads)
        k = len(retrieval_heads)

        base_v = np.asarray(self._accuracy(samples, None))
        ret_v = np.asarray(self._accuracy(samples, retrieval_heads))

        # C1: equal-size random non-retrieval sets, averaged over seeds, with one
        # representative per-sample vector (first seed) kept for McNemar.
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        non_ret = [lh for lh in all_heads if lh not in retrieval_set]
        rand_accs, rand_v0 = [], None
        for si, seed in enumerate(random_seeds):
            if k == 0 or k > len(non_ret):
                break
            rng = np.random.default_rng(seed)
            pick_idx = rng.choice(len(non_ret), size=k, replace=False)
            pick = [non_ret[i] for i in pick_idx]
            v = np.asarray(self._accuracy(samples, pick))
            rand_accs.append(float(v.mean()))
            if si == 0:
                rand_v0 = v

        baseline = float(base_v.mean())
        ret_acc = float(ret_v.mean())
        rand_acc = float(np.mean(rand_accs)) if rand_accs else float("nan")

        # McNemar: retrieval-mask vs random-mask (first seed), same samples.
        if rand_v0 is not None:
            b = int(np.sum((ret_v == 0) & (rand_v0 == 1)))   # retrieval worse
            c = int(np.sum((ret_v == 1) & (rand_v0 == 0)))   # retrieval better
            mcnemar = {"b_retrieval_worse": b, "c_retrieval_better": c,
                       "n_discordant": b + c, "p_value": _mcnemar_exact(b, c)}
        else:
            mcnemar = {"p_value": float("nan")}

        result = {
            "n_samples": len(samples),
            "n_retrieval_heads": k,
            "baseline": baseline,
            "mask_retrieval": ret_acc,
            "mask_random": rand_acc,
            "knockout_drop": baseline - ret_acc,
            "random_drop": baseline - rand_acc if rand_acc == rand_acc else float("nan"),
            "dissociation": (baseline - ret_acc) - (baseline - rand_acc) if rand_acc == rand_acc else float("nan"),
            "random_seeds": random_seeds,
            "mcnemar": mcnemar,
        }
        logger.info(
            "E3 knockout: baseline=%.3f mask_ret=%.3f mask_rand=%.3f → drop=%.3f (rand %.3f) McNemar p=%.4g",
            baseline, ret_acc, rand_acc, result["knockout_drop"], result["random_drop"],
            mcnemar["p_value"],
        )
        return result


@contextmanager
def _null():
    yield
