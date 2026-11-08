"""LLM adaptation and selective input gradients used by Consensus3D.

The adaptation policy follows the Chat-Scene implementation base. It is
documented here as part of the full framework, not as a new contribution.
See THIRD_PARTY_NOTICES.md for attribution and upstream licensing.
"""

from typing import Dict, Sequence

import torch
from torch import Tensor, nn


LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def prepare_language_model(
    language_model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = LORA_TARGETS,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """Freeze the backbone and enable LoRA, input embeddings and output head.

    Supply a causal language model whose vocabulary has already been extended
    with object identifiers. This function does not download model weights.
    """
    from peft import LoraConfig, get_peft_model

    for parameter in language_model.parameters():
        parameter.requires_grad_(False)
    adapted = get_peft_model(
        language_model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    for module in (adapted.get_input_embeddings(), adapted.get_output_embeddings()):
        module.to(dtype=torch.float32)
        module.weight.requires_grad_(True)
    if gradient_checkpointing:
        adapted.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        adapted.config.use_cache = False
    return adapted


def selective_input_embeddings(
    embedding: nn.Module,
    token_ids: Tensor,
    *,
    object_id_start: int,
    object_id_end: int,
    system_prompt: bool = False,
) -> Tensor:
    """Pass input-path gradients only through the added object identifiers.

    System-prompt input activations are always detached. The embedding matrix
    remains a trainable optimizer parameter; detaching an input path does not
    disable weight decay or an independently shared output-head gradient.
    """
    if not 0 <= object_id_start < object_id_end <= embedding.num_embeddings:
        raise ValueError("Object identifier range must lie within the vocabulary.")
    embeddings = embedding(token_ids)
    if system_prompt:
        return embeddings.detach()
    active = ((token_ids >= object_id_start) & (token_ids < object_id_end)).unsqueeze(-1)
    return torch.where(active, embeddings, embeddings.detach())


def trainable_parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Return parameter counts for an inspectable adaptation setup."""
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

