"""Shared I/Q CNN for the classification-based baselines."""
from __future__ import annotations

import torch
from torch import nn


class IQClassifier(nn.Module):
    def __init__(self, class_count: int, channels: list[int], kernel_sizes: list[int],
                 feature_dim: int, dropout: float):
        super().__init__()
        if len(channels) != len(kernel_sizes) or not channels:
            raise ValueError("Each convolution requires a channel count and kernel size.")
        if class_count < 2 or feature_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid classifier dimensions or dropout.")
        blocks = []
        previous = 2
        length = 256
        for width, kernel in zip(channels, kernel_sizes):
            if width < 1 or kernel < 1 or kernel % 2 != 1 or length < 2:
                raise ValueError("Convolutions require positive widths and odd kernels.")
            blocks.extend([
                nn.Conv1d(previous, width, kernel, padding=kernel // 2),
                nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2),
            ])
            previous = width
            length //= 2
        self.convolutions = nn.Sequential(*blocks)
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Linear(previous * length, feature_dim), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, class_count)

    def features(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 256):
            raise ValueError("CNN input must have shape [batch, 2, 256].")
        return self.embedding(self.convolutions(iq))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(iq)))
