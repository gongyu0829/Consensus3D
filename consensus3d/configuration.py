"""Load and validate Consensus3D architecture and training settings."""

import json
import math
from pathlib import Path


def load_configuration(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    training = config["training"]
    batch = training["samples_per_gpu"] * training["gpu_count"] * training["gradient_accumulation"]
    if batch != training["global_batch_size"]:
        raise ValueError("Global batch size is inconsistent with the device configuration.")
    effective_lr = training["learning_rate_argument"] * batch
    if not math.isclose(effective_lr, training["effective_base_learning_rate"]):
        raise ValueError("Effective learning rate is inconsistent with the scaling rule.")
    if config["architecture"]["language_hidden_size"] != 4096:
        raise ValueError("This reference configuration targets the 7B language interface.")
    if training["epochs"] != 2:
        raise ValueError("The manuscript reference configuration uses two epochs.")
    return config

