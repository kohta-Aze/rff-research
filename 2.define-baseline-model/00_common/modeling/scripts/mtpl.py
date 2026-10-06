"""Common-CNN MTPL comparison variant: classification, reconstruction, centers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cnn_backbone import IQClassifier


class MultiTaskPrototype(nn.Module):
    def __init__(self, class_count, architecture, prototype_std=.01):
        super().__init__()
        if not math.isfinite(prototype_std) or prototype_std <= 0:
            raise ValueError('Prototype initialization scale must be positive and finite.')
        self.encoder = IQClassifier(class_count, **architecture)
        channels = architecture['channels']
        self.decoder_length = 256 // (2 ** len(channels))
        self.decoder_input = nn.Sequential(nn.Linear(architecture['feature_dim'], channels[-1] * self.decoder_length), nn.ReLU())
        blocks = []
        previous = channels[-1]
        for width in [*reversed(channels[:-1]), 2]:
            blocks.append(nn.ConvTranspose1d(previous, width, 4, stride=2, padding=1))
            if width != 2:
                blocks.extend([nn.BatchNorm1d(width), nn.ReLU()])
            previous = width
        self.decoder = nn.Sequential(*blocks)
        self.prototypes = nn.Parameter(torch.randn(class_count, architecture['feature_dim']) * prototype_std)

    def features(self, iq):
        return self.encoder.features(iq)

    def forward(self, iq):
        features = self.features(iq)
        logits = self.encoder.classifier(self.encoder.dropout(features))
        latent = self.decoder_input(features).reshape(len(iq), -1, self.decoder_length)
        reconstruction = self.decoder(latent)
        if reconstruction.shape != iq.shape:
            raise ValueError('Decoder reconstruction shape differs from input.')
        return logits, reconstruction, features


def multitask_loss(logits, reconstruction, features, prototypes, iq, labels,
                   reconstruction_weight=.1, prototype_weight=.05):
    if (labels.dtype != torch.int64 or labels.ndim != 1 or len(labels) != len(iq)
            or (labels < 0).any() or (labels >= len(prototypes)).any()):
        raise ValueError('Only known labels can enter MTPL optimization.')
    if (not len(iq) or reconstruction.shape != iq.shape or features.ndim != 2
            or prototypes.ndim != 2 or features.shape[1] != prototypes.shape[1]
            or len(features) != len(iq) or logits.shape != (len(iq), len(prototypes))
            or any(not torch.isfinite(tensor).all() for tensor in [logits, reconstruction, features, prototypes, iq])
            or not math.isfinite(reconstruction_weight) or reconstruction_weight < 0
            or not math.isfinite(prototype_weight) or prototype_weight < 0):
        raise ValueError('Invalid MTPL tensors or loss weights.')
    classification = F.cross_entropy(logits, labels)
    # Equations (4) and (5) sum coordinates, then average samples (not pixel MSE).
    reconstruction_loss = (reconstruction - iq).square().flatten(1).sum(1).mean()
    prototype_loss = (features - prototypes[labels]).square().sum(1).mean()
    total = classification + reconstruction_weight * reconstruction_loss + prototype_weight * prototype_loss
    return total, {'classification': classification, 'reconstruction': reconstruction_loss,
                   'prototype': prototype_loss}
