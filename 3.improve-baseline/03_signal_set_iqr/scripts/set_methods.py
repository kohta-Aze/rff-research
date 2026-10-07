"""Label-free set inference and deterministic packet partitions."""
from __future__ import annotations

import hashlib
import numpy as np


def finite_matrix(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Expected a nonempty finite feature matrix.')
    return matrix


class MedianIQR:
    def __init__(self, class_count, floor_fraction=0.1):
        if type(class_count) is not int or class_count < 2 or not 0 < floor_fraction <= 1:
            raise ValueError('Invalid class count or scale floor.')
        self.class_count = class_count
        self.floor_fraction = floor_fraction
        self.centers = self.scales = self.class_iqr = self.floor = None

    def fit(self, features, labels):
        x = finite_matrix(features)
        y = np.asarray(labels)
        if y.shape != (len(x),) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= self.class_count):
            raise ValueError('Only known training labels can fit prototypes.')
        if set(y.tolist()) != set(range(self.class_count)):
            raise ValueError('Every known class requires training samples.')
        pooled_iqr = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        positive = pooled_iqr[pooled_iqr > 0]
        if not len(positive):
            raise ValueError('All training features are constant; distance scale is undefined.')
        self.floor = self.floor_fraction * np.where(pooled_iqr > 0, pooled_iqr, np.median(positive))
        quantiles = np.stack([np.quantile(x[y == k], [0.25, 0.5, 0.75], axis=0)
                              for k in range(self.class_count)])
        self.centers = quantiles[:, 1]
        self.class_iqr = quantiles[:, 2] - quantiles[:, 0]
        self.scales = np.maximum(self.class_iqr, self.floor)
        return self

    def infer(self, sets):
        """Receive only feature sets; no identity, date, receiver, or target label."""
        if self.centers is None:
            raise ValueError('Fit training prototypes before inference.')
        representatives = np.stack([np.median(finite_matrix(x), axis=0) for x in sets])
        if representatives.shape[1] != self.centers.shape[1]:
            raise ValueError('Feature dimension differs from training.')
        distances = np.sqrt(np.mean(((representatives[:, None] - self.centers)
                                     / self.scales) ** 2, axis=-1))
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite distances.')
        predictions = distances.argmin(axis=1)
        scores = -distances[np.arange(len(predictions)), predictions]
        return representatives, predictions, scores, distances


def vote(final_predictions, class_count):
    """Unique plurality among known and unknown (-1); all ties reject."""
    values = np.asarray(final_predictions)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu' or np.any(values < -1) or np.any(values >= class_count):
        raise ValueError('Invalid final packet predictions.')
    counts = np.bincount(values + 1, minlength=class_count + 1)
    winners = np.flatnonzero(counts == counts.max())
    return (int(winners[0]) - 1 if len(winners) == 1 else -1), len(winners) > 1


def partition(rows, size, seed):
    """Labels only define/verify anonymous pure-Tx groups and score outcomes."""
    if type(size) is not int or size < 1 or type(seed) is not int:
        raise ValueError('Invalid partition parameters.')
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row['relative_path'], []).append(index)
    result = []
    for path in sorted(groups):
        indices = groups[path]
        if len(indices) % size:
            raise ValueError('Set size must divide each group; no dropping packets.')
        if len({(rows[i]['tx_id'], rows[i]['rx_id'], rows[i]['capture_date']) for i in indices}) != 1:
            raise ValueError('Mixed identities in one assumed source group.')
        signal_ids = [int(rows[i]['signal_index']) for i in indices]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError('Duplicate signals in one source group.')
        ordered = sorted(indices, key=lambda i: hashlib.sha256(
            f"signal-set-v1|{seed}|{path}|{rows[i]['signal_index']}".encode()).digest())
        anonymous = hashlib.sha256(f'{seed}|{path}'.encode()).hexdigest()[:16]
        for start in range(0, len(ordered), size):
            result.append({'set_id': f'{anonymous}-n{size}-{start // size:02d}',
                           'indices': ordered[start:start + size]})
    if sorted(i for group in result for i in group['indices']) != list(range(len(rows))):
        raise ValueError('Partition must cover every packet exactly once.')
    return result


def outcome_metrics(labels, final_predictions):
    y, p = np.asarray(labels), np.asarray(final_predictions)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or y.dtype.kind not in 'iu' or p.dtype.kind not in 'iu' or np.any(y < -1) or np.any(p < -1):
        raise ValueError('Invalid outcome arrays.')
    known, unknown = y >= 0, y < 0
    accepted, correct = p >= 0, p == y
    rate = lambda mask, population: float(mask[population].mean()) if population.any() else None
    ccr = rate(accepted & correct, known)
    reject = rate(~accepted, unknown)
    return {
        'set_count': len(y), 'known_count': int(known.sum()), 'unknown_count': int(unknown.sum()),
        'known_correct_accept_count': int((known & accepted & correct).sum()),
        'known_false_reject_count': int((known & ~accepted).sum()),
        'known_misidentification_count': int((known & accepted & ~correct).sum()),
        'unknown_false_accept_count': int((unknown & accepted).sum()),
        'unknown_reject_count': int((unknown & ~accepted).sum()),
        'known_correct_accept_rate': ccr,
        'known_false_reject_rate': rate(~accepted, known),
        'known_misidentification_rate': rate(accepted & ~correct, known),
        'unknown_false_accept_rate': rate(accepted, unknown),
        'unknown_reject_rate': reject,
        'balanced_open_set_accuracy': (ccr + reject) / 2 if ccr is not None and reject is not None else None,
    }
