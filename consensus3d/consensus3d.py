"""Probabilistic object representations and their language-token interface."""


from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ObjectDistribution:
    tokens: Tensor
    mean: Tensor
    log_variance: Tensor


@dataclass
class PairedObjectRepresentations:
    geometry: ObjectDistribution
    appearance: ObjectDistribution
    valid_objects: Tensor


class DistributionalObjectProjector(nn.Module):
    """Map normalized object features to a diagonal Gaussian in LLM space.

    Input normalization is performed by Consensus3DObjectEncoder. The two
    initialization terms below reproduce the effective initial log variance
    of -9: a head bias of -4.5 plus a forward offset of -4.5.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        log_variance_bias: float = -4.5,
        log_variance_offset: float = -4.5,
        log_variance_max: float = -1.5,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")
        self.encoder = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU())
        self.mean_head = nn.Linear(output_dim, output_dim)
        self.log_variance_head = nn.Linear(output_dim, output_dim)
        self.log_variance_offset = log_variance_offset
        self.log_variance_max = log_variance_max
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, log_variance_bias)

    def forward(self, features: Tensor, *, deterministic: bool = False) -> ObjectDistribution:
        if features.shape[-1] != self.encoder[0].in_features:
            raise ValueError("Input feature dimension does not match the projector.")
        hidden = self.encoder(features)
        mean = self.mean_head(hidden)
        log_variance = torch.clamp(
            self.log_variance_head(hidden) + self.log_variance_offset,
            max=self.log_variance_max,
        )
        if self.training and not deterministic:
            standard_deviation = torch.exp(0.5 * log_variance)
            tokens = mean + torch.randn_like(standard_deviation) * standard_deviation
        else:
            tokens = mean
        return ObjectDistribution(tokens, mean, log_variance)


class Consensus3DObjectEncoder(nn.Module):
    """Construct paired object representations before language-model input.

    The two branches have independent parameters. Padding is carried as an
    explicit object mask; an absent modality is represented by a zero feature
    for an otherwise valid object, rather than by discarding that object.
    """

    def __init__(
        self,
        geometry_dim: int = 1024,
        appearance_dim: int = 1024,
        language_dim: int = 4096,
        deterministic_warmup_steps: int = 2000,
        **projector_options,
    ):
        super().__init__()
        if deterministic_warmup_steps < 0:
            raise ValueError("Warm-up steps must be nonnegative.")
        self.deterministic_warmup_steps = deterministic_warmup_steps
        self.geometry_projector = DistributionalObjectProjector(
            geometry_dim, language_dim, **projector_options
        )
        self.appearance_projector = DistributionalObjectProjector(
            appearance_dim, language_dim, **projector_options
        )

    def forward(
        self,
        geometry_features: Tensor,
        appearance_features: Tensor,
        valid_objects: Tensor,
        *,
        global_step: int = 0,
    ) -> PairedObjectRepresentations:
        if geometry_features.ndim != 3 or appearance_features.ndim != 3:
            raise ValueError("Features must have shape [batch, objects, features].")
        if geometry_features.shape[:2] != appearance_features.shape[:2]:
            raise ValueError("Both modalities must use the same object ordering.")
        if valid_objects.shape != geometry_features.shape[:2] or valid_objects.dtype != torch.bool:
            raise ValueError("valid_objects must be a boolean [batch, objects] mask.")
        if global_step < 0:
            raise ValueError("global_step must be nonnegative.")
        deterministic = global_step < self.deterministic_warmup_steps
        geometry = self.geometry_projector(
            F.normalize(geometry_features, dim=-1), deterministic=deterministic
        )
        appearance = self.appearance_projector(
            F.normalize(appearance_features, dim=-1), deterministic=deterministic
        )
        return PairedObjectRepresentations(geometry, appearance, valid_objects)

    @staticmethod
    def object_token_sequences(
        representations: PairedObjectRepresentations,
        identifier_embeddings: Tensor,
    ) -> List[Tensor]:
        """Interleave [object identifier, geometry token, appearance token].

        The caller supplies identifiers in the same order as the paired
        features, after any shared object-ID permutation. Text instructions
        and response embeddings are assembled by the downstream task code.
        """
        geometry = representations.geometry.tokens
        appearance = representations.appearance.tokens
        if identifier_embeddings.shape != geometry.shape:
            raise ValueError("Identifier embeddings must match the projected feature shape.")
        interleaved = torch.stack((identifier_embeddings, geometry, appearance), dim=2)
        return [
            scene[mask].flatten(0, 1)
            for scene, mask in zip(interleaved, representations.valid_objects)
        ]

