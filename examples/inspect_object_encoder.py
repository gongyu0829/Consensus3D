"""Inspect DOP mean inference and object tokens with synthetic features on CPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from consensus3d import Consensus3DObjectEncoder
from consensus3d.configuration import load_configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-dimensions", action="store_true",
        help="Use the projector dimensions from the manuscript reference configuration.",
    )
    args = parser.parse_args()
    config = load_configuration(Path(__file__).resolve().parents[1] / "configs/consensus3d_7b.json")
    torch.manual_seed(config["training"]["seed"])
    architecture = config["architecture"]
    geometry_dim = architecture["geometry_feature_size"] if args.full_dimensions else 16
    appearance_dim = architecture["appearance_feature_size"] if args.full_dimensions else 16
    hidden_dim = architecture["language_hidden_size"] if args.full_dimensions else 32
    model = Consensus3DObjectEncoder(
        geometry_dim, appearance_dim, hidden_dim,
        deterministic_warmup_steps=config["training"]["deterministic_dop_warmup_steps"],
        log_variance_bias=architecture["log_variance_head_bias_init"],
        log_variance_offset=architecture["log_variance_forward_offset"],
        log_variance_max=architecture["log_variance_upper_bound"],
    )
    geometry = torch.randn(2, 4, geometry_dim)
    appearance = torch.randn(2, 4, appearance_dim)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    identifiers = torch.randn(2, 4, hidden_dim)
    model.eval()
    with torch.no_grad():
        representations = model(geometry, appearance, valid)
        sequences = model.object_token_sequences(representations, identifiers)
    print(json.dumps({
        "input_shape": list(geometry.shape),
        "projected_shape": list(representations.geometry.tokens.shape),
        "object_sequence_lengths": [len(s) for s in sequences],
        "initial_log_variance": representations.geometry.log_variance.mean().item(),
        "evaluation_uses_means": torch.equal(representations.geometry.tokens, representations.geometry.mean),
        "synthetic_inputs": True,
        "task_benchmark_evaluation": False,
    }, indent=2))


if __name__ == "__main__":
    main()

