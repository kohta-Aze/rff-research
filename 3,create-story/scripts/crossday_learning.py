"""Balanced source-only sampling and a cross-day supervised contrastive loss."""
import torch
from torch.nn import functional as F


def balanced_order(labels, days, generator):
    keys = sorted(set(zip(labels.tolist(), days.tolist())))
    groups = [torch.flatnonzero((labels == label) & (days == day)) for label, day in keys]
    if len(set(map(len, groups))) != 1:
        raise ValueError("Source label/day cells must have equal counts.")
    groups = [group[torch.randperm(len(group), generator=generator)] for group in groups]
    order = torch.randperm(len(groups), generator=generator).tolist()
    return torch.stack([groups[i] for i in order], dim=1).reshape(-1)


def crossday_loss(projected, labels, days, temperature):
    if temperature <= 0 or projected.ndim != 2 or len(projected) != len(labels) or len(labels) != len(days):
        raise ValueError("Invalid cross-day contrastive input.")
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=projected.device)
    positive = (labels[:, None] == labels[None, :]) & (days[:, None] != days[None, :])
    if (positive.sum(1) == 0).any():
        raise ValueError("Every anchor needs a same-transmitter, different-day positive.")
    vectors = F.normalize(projected, dim=1)
    logits = vectors @ vectors.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    return -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
