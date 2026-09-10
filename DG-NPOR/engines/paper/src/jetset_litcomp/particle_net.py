"""A compact, mask-aware PyTorch implementation of ParticleNet.

The EdgeConv channel layouts, k values, global pooling, dense layer, and
dropout follow Qu & Gouskos, Phys. Rev. D 101 (2020) 056019. The input adapter
is necessarily JetSet-specific: 19 reconstructed track fields, two relative
angular coordinates, and four reconstructed jet-context variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on installations without torch
    torch = None
    nn = None


@dataclass(frozen=True)
class ParticleNetConfig:
    name: str
    k: int
    channels: tuple[tuple[int, ...], ...]
    dense_units: int
    dropout: float

    def to_dict(self):
        result = asdict(self)
        result["channels"] = [list(value) for value in self.channels]
        return result


CONFIGS = {
    "full": ParticleNetConfig(
        name="ParticleNet",
        k=16,
        channels=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
        dense_units=256,
        dropout=0.10,
    ),
    "lite": ParticleNetConfig(
        name="ParticleNet-Lite",
        k=7,
        channels=((32, 32, 32), (64, 64, 64)),
        dense_units=128,
        dropout=0.10,
    ),
}


def require_torch():
    if torch is None:
        raise RuntimeError(
            "ParticleNet training requires PyTorch. Install "
            "requirements.txt in the active environment."
        )


def select_device(requested="auto"):
    require_torch()
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if nn is not None:

    def _gather_neighbors(features, indices):
        """Gather BxCxN features at BxNxK neighbor indices."""
        batch, channels, nodes = features.shape
        k = indices.shape[2]
        base = torch.arange(batch, device=features.device).view(batch, 1, 1) * nodes
        flat_index = (indices + base).reshape(-1)
        flat_features = features.transpose(1, 2).contiguous().reshape(
            batch * nodes, channels
        )
        gathered = flat_features[flat_index].reshape(batch, nodes, k, channels)
        return gathered.permute(0, 3, 1, 2).contiguous()


    def _knn(coordinates, mask, k):
        """Masked kNN including the central point, as in standard DGCNN kNN."""
        points = coordinates.transpose(1, 2).contiguous()
        distance = torch.cdist(points, points, p=2).square()
        valid_keys = mask[:, None, :].expand_as(distance)
        distance = distance.masked_fill(~valid_keys, torch.finfo(distance.dtype).max)
        k_eff = min(int(k), int(distance.shape[2]))
        values, indices = torch.topk(distance, k=k_eff, dim=2, largest=False)
        neighbor_valid = torch.isfinite(values) & (
            values < torch.finfo(values.dtype).max / 2
        )
        neighbor_valid &= mask[:, :, None]
        return indices, neighbor_valid


    class EdgeConvBlock(nn.Module):
        def __init__(self, input_channels, output_channels, k):
            super().__init__()
            self.k = int(k)
            layers = []
            current = 2 * int(input_channels)
            for channels in output_channels:
                layers.extend(
                    [
                        nn.Conv2d(current, int(channels), kernel_size=1, bias=False),
                        nn.BatchNorm2d(int(channels)),
                        nn.ReLU(inplace=True),
                    ]
                )
                current = int(channels)
            self.edge_mlp = nn.Sequential(*layers)
            final = int(output_channels[-1])
            if int(input_channels) == final:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(int(input_channels), final, kernel_size=1, bias=False),
                    nn.BatchNorm1d(final),
                )
            self.activation = nn.ReLU(inplace=True)

        def forward(self, features, coordinates, mask):
            indices, neighbor_valid = _knn(coordinates, mask, self.k)
            neighbors = _gather_neighbors(features, indices)
            central = features.unsqueeze(-1).expand_as(neighbors)
            edge = torch.cat([central, neighbors - central], dim=1)
            message = self.edge_mlp(edge)
            minimum = torch.finfo(message.dtype).min
            message = message.masked_fill(~neighbor_valid[:, None, :, :], minimum)
            message = message.max(dim=3).values
            message = torch.where(mask[:, None, :], message, torch.zeros_like(message))
            residual = self.shortcut(features)
            output = self.activation(message + residual)
            return output * mask[:, None, :].to(output.dtype)


    class ParticleNet(nn.Module):
        """ParticleNet adapted to reconstructed ATLAS JetSet tracks."""

        def __init__(
            self,
            input_features=19,
            context_features=4,
            coordinate_indices=(3, 2),
            architecture="full",
        ):
            super().__init__()
            if architecture not in CONFIGS:
                raise ValueError(f"Unknown ParticleNet architecture: {architecture}")
            self.architecture = str(architecture)
            self.config = CONFIGS[self.architecture]
            self.coordinate_indices = tuple(int(value) for value in coordinate_indices)
            blocks = []
            current = int(input_features)
            for channels in self.config.channels:
                blocks.append(EdgeConvBlock(current, channels, self.config.k))
                current = int(channels[-1])
            self.blocks = nn.ModuleList(blocks)
            self.classifier = nn.Sequential(
                nn.Linear(current + int(context_features), self.config.dense_units),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.dense_units, 1),
            )

        def forward(self, tracks, mask, context):
            if tracks.ndim != 3 or mask.ndim != 2:
                raise ValueError("Expected tracks BxNxF and mask BxN.")
            mask = mask.bool()
            original = tracks
            features = tracks.transpose(1, 2).contiguous()
            coordinates = original[:, :, self.coordinate_indices].transpose(1, 2)
            for block_index, block in enumerate(self.blocks):
                relation = coordinates if block_index == 0 else features
                features = block(features, relation, mask)
            weights = mask[:, None, :].to(features.dtype)
            pooled = (features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            return self.classifier(torch.cat([pooled, context], dim=1)).squeeze(1)

        def summary(self):
            return {
                "architecture": self.architecture,
                "paper_name": self.config.name,
                "paper_edgeconv_k": int(self.config.k),
                "paper_edgeconv_channels": [
                    list(value) for value in self.config.channels
                ],
                "paper_dense_units": int(self.config.dense_units),
                "paper_dropout": float(self.config.dropout),
                "jetset_track_feature_count": 19,
                "jet_context_feature_count": 4,
                "first_knn_coordinates": ["track_deta", "track_dphi"],
                "parameter_count": int(
                    sum(parameter.numel() for parameter in self.parameters())
                ),
            }

else:  # pragma: no cover

    class ParticleNet:
        def __init__(self, *args, **kwargs):
            require_torch()
