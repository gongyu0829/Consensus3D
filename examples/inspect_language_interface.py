"""Check DOP-to-language-model connectivity using random CPU modules.

The small Llama and synthetic token IDs exercise the released interfaces.
They do not load Vicuna, use task prompts, or evaluate benchmark accuracy.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from consensus3d import Consensus3DObjectEncoder
from consensus3d.configuration import load_configuration
from consensus3d.finetuning import prepare_language_model, selective_input_embeddings


def main():
    config = load_configuration(Path(__file__).resolve().parents[1] / "configs/consensus3d_7b.json")
    architecture, adaptation, training = (
        config["architecture"], config["fine_tuning"], config["training"]
    )
    torch.manual_seed(training["seed"])

    # Reduced dimensions keep this interface check independent of model downloads.
    vocabulary_size, hidden_size, feature_size = 64, 32, 8
    object_id_start = vocabulary_size - 3
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=vocabulary_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))
    language_model = prepare_language_model(
        backbone,
        rank=adaptation["lora_rank"], alpha=adaptation["lora_alpha"],
        dropout=adaptation["lora_dropout"], target_modules=adaptation["lora_targets"],
        gradient_checkpointing=adaptation["gradient_checkpointing"],
    )
    encoder = Consensus3DObjectEncoder(
        feature_size, feature_size, hidden_size,
        deterministic_warmup_steps=training["deterministic_dop_warmup_steps"],
        log_variance_bias=architecture["log_variance_head_bias_init"],
        log_variance_offset=architecture["log_variance_forward_offset"],
        log_variance_max=architecture["log_variance_upper_bound"],
    )
    geometry, appearance = torch.randn(1, 3, feature_size), torch.randn(1, 3, feature_size)
    valid_objects = torch.tensor([[True, False, True]])
    identifier_ids = torch.tensor([[object_id_start, object_id_start + 1, object_id_start + 2]])
    response_ids = torch.tensor([[6, 7, 2]])
    embeddings = language_model.get_input_embeddings()

    def embed(ids, *, system_prompt=False):
        return selective_input_embeddings(
            embeddings, ids, object_id_start=object_id_start,
            object_id_end=vocabulary_size, system_prompt=system_prompt,
        )

    def assemble():
        representations = encoder(
            geometry, appearance, valid_objects,
            global_step=training["deterministic_dop_warmup_steps"],
        )
        visual = encoder.object_token_sequences(representations, embed(identifier_ids))[0].unsqueeze(0)
        system = embed(torch.tensor([[1, 3]]), system_prompt=True)
        instruction = embed(torch.tensor([[4, 5]]))
        prefix = torch.cat((system, visual, instruction), dim=1)
        return representations, prefix, visual.shape[1]

    encoder.train()
    language_model.train()
    _, prefix, visual_length = assemble()
    inputs = torch.cat((prefix, embed(response_ids)), dim=1)
    # Only the synthetic response is supervised; scene and instruction positions
    # are ignored. The causal LM applies the next-token shift internally.
    labels = torch.cat((torch.full(prefix.shape[:2], -100, dtype=torch.long), response_ids), dim=1)
    attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long)
    loss = language_model(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels).loss
    if not torch.isfinite(loss):
        raise RuntimeError("The interface check produced a non-finite language loss.")
    loss.backward()

    def has_gradient(parameter):
        gradient = parameter.grad
        return gradient is not None and bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum() > 0)

    both_branches = all(
        has_gradient(head.weight)
        for branch in (encoder.geometry_projector, encoder.appearance_projector)
        for head in (branch.mean_head, branch.log_variance_head)
    )
    adapter_gradients = any(has_gradient(p) for name, p in language_model.named_parameters() if "lora_B" in name)
    frozen_backbone = all(
        p.grad is None for p in language_model.parameters() if not p.requires_grad
    )
    input_gradient = embeddings.weight.grad
    input_routing = (
        input_gradient is not None
        and bool(torch.isfinite(input_gradient).all())
        and bool((input_gradient[:object_id_start] == 0).all())
        and bool((input_gradient[object_id_start + 1] == 0).all())
        and all(bool(input_gradient[i].abs().sum() > 0) for i in (object_id_start, object_id_start + 2))
    )
    output_gradient = has_gradient(language_model.get_output_embeddings().weight)

    encoder.eval()
    language_model.eval()
    with torch.no_grad():
        representations, prefix, _ = assemble()
        mask = torch.ones(prefix.shape[:2], dtype=torch.long)
        logits = language_model(inputs_embeds=prefix, attention_mask=mask).logits
        repeated = language_model(inputs_embeds=prefix, attention_mask=mask).logits
        means_used = all(torch.equal(branch.tokens, branch.mean) for branch in (
            representations.geometry, representations.appearance
        ))
        repeatable = bool(torch.isfinite(logits).all()) and torch.equal(logits, repeated)

    checks = {
        "response_only_supervision": bool((labels[:, :prefix.shape[1]] == -100).all()),
        "dop_mean_and_variance_heads_receive_gradients": both_branches,
        "lora_receives_gradients": adapter_gradients,
        "frozen_backbone_has_no_gradients": frozen_backbone,
        "object_identifier_input_gradients_only": input_routing,
        "output_head_receives_gradients": output_gradient,
        "default_gradient_checkpointing_enabled": language_model.is_gradient_checkpointing,
        "evaluation_uses_means": means_used,
        "evaluation_forward_is_repeatable": repeatable,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Interface checks failed: {checks}")
    print(json.dumps({
        "checks": checks,
        "valid_objects": int(valid_objects.sum()),
        "visual_token_count": visual_length,
        "language_model": "small randomly initialized Llama",
        "synthetic_inputs": True,
        "task_benchmark_evaluation": False,
    }, indent=2))


if __name__ == "__main__":
    main()
