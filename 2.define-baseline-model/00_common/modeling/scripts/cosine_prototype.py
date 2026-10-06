"""Class means of raw embeddings, normalized for cosine classification."""
from __future__ import annotations

import torch
from torch.nn import functional as F


class CosinePrototype:
    def __init__(self, class_count: int, epsilon: float = 1e-12):
        if type(class_count) is not int or class_count < 2 or epsilon <= 0:
            raise ValueError('Invalid class count or normalization epsilon.')
        self.class_count = class_count
        self.epsilon = epsilon
        self.prototypes = None
        self.raw_means = None
        self.counts = None

    def _features(self, features):
        features = torch.as_tensor(features, dtype=torch.float64, device='cpu')
        if features.ndim != 2 or not len(features) or features.shape[1] < 1:
            raise ValueError('Features must be a nonempty [sample, feature] matrix.')
        if not torch.isfinite(features).all():
            raise ValueError('Features must be finite.')
        if (torch.linalg.vector_norm(features, dim=1) <= self.epsilon).any():
            raise ValueError('Zero or near-zero features cannot define a cosine direction.')
        return features

    def fit(self, features, labels):
        features = self._features(features)
        labels = torch.as_tensor(labels, device='cpu')
        if labels.ndim != 1 or len(labels) != len(features) or labels.dtype != torch.int64:
            raise ValueError('Labels must be an int64 vector matching the features.')
        if (labels < 0).any() or (labels >= self.class_count).any():
            raise ValueError('Only known training labels can build prototypes.')
        counts = torch.bincount(labels, minlength=self.class_count)
        if (counts == 0).any():
            raise ValueError('Every known class requires training features.')
        means = torch.stack([features[labels == label].mean(dim=0)
                             for label in range(self.class_count)])
        if (torch.linalg.vector_norm(means, dim=1) <= self.epsilon).any():
            raise ValueError('A zero or near-zero class mean cannot define a prototype.')
        self.raw_means = means
        self.prototypes = F.normalize(means, p=2, dim=1, eps=self.epsilon)
        self.counts = counts
        return self

    def similarities(self, features):
        if self.prototypes is None:
            raise ValueError('Fit prototypes before inference.')
        features = self._features(features)
        if features.shape[1] != self.prototypes.shape[1]:
            raise ValueError('Feature dimensions differ from the prototypes.')
        normalized = F.normalize(features, p=2, dim=1, eps=self.epsilon)
        return (normalized @ self.prototypes.T).clamp(-1., 1.)
