"""Behavioral checks for object representations and language-model adaptation."""

from pathlib import Path
import unittest

import torch
from torch import nn

from consensus3d import Consensus3DObjectEncoder, DistributionalObjectProjector
from consensus3d.configuration import load_configuration
from consensus3d.finetuning import prepare_language_model, selective_input_embeddings


class ObjectEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_staged_sampling_and_independent_branches(self):
        model = Consensus3DObjectEncoder(8, 8, 16)
        geometry, appearance = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        early = model(geometry, appearance, mask, global_step=1999)
        self.assertTrue(torch.equal(early.geometry.tokens, early.geometry.mean))
        self.assertTrue(torch.all(early.geometry.log_variance == -9))
        later = model(geometry, appearance, mask, global_step=2000)
        self.assertFalse(torch.equal(later.geometry.tokens, later.geometry.mean))
        model.eval()
        evaluation = model(geometry, appearance, mask, global_step=3000)
        self.assertTrue(torch.equal(evaluation.geometry.tokens, evaluation.geometry.mean))
        self.assertTrue(torch.equal(evaluation.appearance.tokens, evaluation.appearance.mean))
        self.assertFalse(set(model.geometry_projector.parameters()) & set(model.appearance_projector.parameters()))

    def test_sampling_trains_both_distribution_heads(self):
        projector = DistributionalObjectProjector(8, 16)
        out = projector(torch.randn(4, 8))
        out.tokens.square().sum().backward()
        self.assertGreater(projector.mean_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(projector.log_variance_head.weight.grad.abs().sum().item(), 0)

    def test_token_order_padding_and_zero_modality(self):
        model = Consensus3DObjectEncoder(8, 8, 16).eval()
        geometry = torch.randn(1, 3, 8)
        appearance = torch.zeros_like(geometry)
        mask = torch.tensor([[True, False, True]])
        ids = torch.randn(1, 3, 16, requires_grad=True)
        out = model(geometry, appearance, mask)
        sequence = model.object_token_sequences(out, ids)[0]
        self.assertEqual(tuple(sequence.shape), (6, 16))
        self.assertTrue(torch.equal(sequence[0], ids[0, 0]))
        self.assertTrue(torch.equal(sequence[1], out.geometry.tokens[0, 0]))
        self.assertTrue(torch.equal(sequence[2], out.appearance.tokens[0, 0]))
        self.assertTrue(torch.equal(sequence[3], ids[0, 2]))
        self.assertTrue(torch.isfinite(sequence).all())
        sequence.sum().backward()
        self.assertTrue(torch.all(ids.grad[0, 1] == 0))
        self.assertGreater(ids.grad[0, 0].abs().sum().item(), 0)

    def test_configuration_has_consistent_scaled_learning_rate(self):
        config = load_configuration(Path(__file__).resolve().parents[1] / "configs/consensus3d_7b.json")
        self.assertEqual(config["architecture"]["language_checkpoint"], "lmsys/vicuna-7b-v1.5")
        self.assertEqual(config["training"]["global_batch_size"], 32)
        self.assertEqual(config["training"]["epochs"], 3)


class LanguageAdaptationTests(unittest.TestCase):
    def test_selective_input_gradient_and_system_prompt(self):
        embedding = nn.Embedding(12, 8)
        ids = torch.tensor([[1, 10, 2, 11]])
        out = selective_input_embeddings(embedding, ids, object_id_start=10, object_id_end=12)
        out.sum().backward()
        self.assertTrue(torch.all(embedding.weight.grad[:10] == 0))
        self.assertTrue(torch.all(embedding.weight.grad[10:] != 0))
        prompt = selective_input_embeddings(embedding, ids, object_id_start=10, object_id_end=12, system_prompt=True)
        self.assertFalse(prompt.requires_grad)

    def test_actual_lora_policy_and_backward_on_small_random_llama(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(42)
        backbone = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, tie_word_embeddings=False,
        ))
        model = prepare_language_model(backbone, gradient_checkpointing=False)
        for name, parameter in model.named_parameters():
            expected = "lora_" in name or "embed_tokens" in name or "lm_head" in name
            self.assertEqual(parameter.requires_grad, expected, name)
        ids = torch.tensor([[1, 30, 2, 31, 3]])
        embeddings = selective_input_embeddings(model.get_input_embeddings(), ids, object_id_start=30, object_id_end=32)
        loss = model(inputs_embeds=embeddings, labels=ids).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.get_output_embeddings().weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.all(model.get_input_embeddings().weight.grad[:30] == 0))
        self.assertGreater(model.get_input_embeddings().weight.grad[30:].abs().sum().item(), 0)
        adapter_gradients = [p.grad for n,p in model.named_parameters() if "lora_B" in n]
        self.assertTrue(any(g is not None and g.abs().sum().item() > 0 for g in adapter_gradients))


if __name__ == "__main__":
    unittest.main()
