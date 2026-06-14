"""
Quantization-aware model loading wrapper (proposal §3.4 inheritance rings).

The inherited ``src.model_loader.load_model`` handles fp16 and 8-bit. The
lineage chains also need **4-bit** rings:

    quant: "bnb4"        bitsandbytes NF4 load-time 4-bit (no separate HF repo;
                         used for the Llama and Gemma 4-bit rings)
    quant: "awq4"        a pre-quantized AWQ repo (e.g. Qwen2.5-7B-Instruct-AWQ)
    quant: "gptq4"       a pre-quantized GPTQ-Int4 repo

For ``awq4``/``gptq4`` the weights are already quantized in the repo, so we load
through the inherited fp16 path (transformers auto-detects the embedded
quantization config). For ``bnb4`` we build a ``BitsAndBytesConfig`` here. Any
model without a ``quant`` field falls straight through to the inherited loader,
so the panel's behaviour is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_model_any(
    model_cfg: dict,
    model_name: str = "model",
    *,
    load_in_8bit: bool | None = None,
) -> tuple[Any, Any]:
    """
    Load (model, tokenizer) honouring an optional ``quant`` field.

    Delegates to the inherited ``src.model_loader.load_model`` for the
    non-quantized panel and for the pre-quantized AWQ/GPTQ rings; builds a 4-bit
    NF4 config for ``quant: bnb4``.
    """
    quant = model_cfg.get("quant")

    if quant == "bnb4":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_id = model_cfg["hf_id"]
        revision = model_cfg.get("revision") or "main"
        if revision == "main":
            logger.warning(
                "Model '%s' revision is unpinned ('main'). Pin a commit SHA "
                "before the reportable run (item C1).", model_name,
            )
        tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading %s (%s @ %s, 4-bit NF4) …", model_name, hf_id, revision)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision, device_map="auto",
            trust_remote_code=True, quantization_config=bnb,
        )
        model.eval()
        if torch.cuda.is_available():
            logger.info("VRAM after load: %.1f GB", torch.cuda.memory_allocated() / 1e9)
        return model, tokenizer

    # awq4 / gptq4: the repo is already quantized → fp16 path (the inherited
    # loader passes torch_dtype=float16 and transformers reads the embedded
    # quant config). Non-quant models also take this branch unchanged.
    from src.model_loader import load_model

    eight = load_in_8bit
    if quant in ("awq4", "gptq4"):
        eight = False
    return load_model(model_cfg, model_name, load_in_8bit=eight)
