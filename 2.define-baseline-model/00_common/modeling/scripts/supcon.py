"""Two-view supervised contrastive learning (Khosla et al., Eq. 2)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cnn_backbone import IQClassifier


class SupConEncoder(nn.Module):
    def __init__(self, class_count, architecture, projection_dim=128):
        super().__init__()
        if type(projection_dim) is not int or projection_dim < 1:
            raise ValueError('Projection dimension must be a positive integer.')
        self.encoder = IQClassifier(class_count, **architecture)
        self.encoder.classifier.requires_grad_(False)
        width = architecture['feature_dim']
        self.projection = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                        nn.Linear(width, projection_dim))

    def features(self, iq):
        return self.encoder.features(iq)

    def forward(self, iq):
        return F.normalize(self.projection(self.features(iq)), dim=1)


def supervised_contrastive_loss(features, labels, temperature=.07, base_temperature=.07):
    if (features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < 2
            or features.shape[2] < 1 or not torch.isfinite(features).all()
            or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(base_temperature) or base_temperature <= 0):
        raise ValueError('SupCon requires finite [batch, views>=2, dimension] features and positive temperatures.')
    if (labels.ndim != 1 or len(labels) != len(features) or labels.dtype != torch.int64
            or labels.device != features.device or (labels < 0).any()):
        raise ValueError('Only known int64 labels matching the feature batch are allowed.')
    if (torch.linalg.vector_norm(features, dim=2) <= 1e-12).any():
        raise ValueError('Zero projection vectors cannot define cosine similarity.')
    normalized = F.normalize(features, dim=2)
    vectors = normalized.transpose(0, 1).reshape(-1, features.shape[2])
    targets = labels.repeat(features.shape[1])
    diagonal = torch.eye(len(vectors), dtype=torch.bool, device=features.device)
    positive = (targets[:, None] == targets[None, :]) & ~diagonal
    logits = vectors @ vectors.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    positive_mean = log_probability.masked_fill(~positive, 0.).sum(1) / positive.sum(1)
    return -(temperature / base_temperature) * positive_mean.mean()


def augment_iq(iq, generator, phase_degrees=10., noise_snr_db=30.):
    if (iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 256) or not iq.is_floating_point()
            or not torch.isfinite(iq).all() or iq.device.type != 'cpu'
            or not math.isfinite(phase_degrees) or not 0 <= phase_degrees <= 180
            or not math.isfinite(noise_snr_db) or not 0 <= noise_snr_db <= 100):
        raise ValueError('Augmentation requires finite CPU I/Q and bounded phase/noise settings.')
    power = iq.square().sum(1).mean(1)
    if (power <= 0).any():
        raise ValueError('Zero-power signals cannot be normalized.')
    angle = (2 * torch.rand((len(iq), 1), generator=generator, dtype=iq.dtype) - 1) * math.radians(phase_degrees)
    cosine, sine = angle.cos(), angle.sin()
    rotated = torch.stack([iq[:, 0] * cosine - iq[:, 1] * sine,
                           iq[:, 0] * sine + iq[:, 1] * cosine], dim=1)
    sigma = (power / (2 * 10 ** (noise_snr_db / 10))).sqrt()[:, None, None]
    noise = torch.randn(iq.shape, generator=generator, dtype=iq.dtype) * sigma
    augmented = rotated + noise
    rms = augmented.square().sum(1).mean(1).sqrt()[:, None, None]
    return augmented / rms
