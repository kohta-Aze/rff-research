"""Known-positive scores, tied thresholds, and open-set classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validated_arrays(labels, predictions, scores):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != predictions.shape or labels.shape != scores.shape:
        raise ValueError("Labels, predictions, and scores must be matching vectors.")
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Metrics require nonempty, finite scores.")
    if labels.dtype.kind not in 'iu' or predictions.dtype.kind not in 'iu':
        raise ValueError("Labels and predictions must be integers.")
    if np.any(labels < -1) or np.any(predictions < 0):
        raise ValueError("Unknown labels must be -1; predictions must be known classes.")
    return labels, predictions, scores


def operating_curve(labels, predictions, scores):
    labels, predictions, scores = validated_arrays(labels, predictions, scores)
    known = labels >= 0
    n_known = int(known.sum())
    n_unknown = len(labels) - n_known
    if not n_known or not n_unknown:
        raise ValueError("Operating curves require both known and unknown samples.")
    order = np.argsort(-scores, kind='stable')
    ordered_scores = scores[order]
    # Include all tied scores together, matching the acceptance rule score >= threshold.
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    thresholds = np.r_[np.nextafter(ordered_scores[0], np.inf), ordered_scores[ends]]
    correct = known & (predictions == labels)
    ccr = np.r_[0., np.cumsum(correct[order])[ends] / n_known]
    far = np.r_[0., np.cumsum((~known)[order])[ends] / n_unknown]
    known_accept = np.r_[0., np.cumsum(known[order])[ends] / n_known]
    return {
        'threshold': thresholds, 'known_correct_accept_rate': ccr,
        'unknown_false_accept_rate': far, 'known_accept_rate': known_accept,
    }


def select_threshold(labels, predictions, scores):
    curve = operating_curve(labels, predictions, scores)
    objective = .5 * (curve['known_correct_accept_rate'] + 1 - curve['unknown_false_accept_rate'])
    # Descending thresholds make the first optimum the most conservative tied optimum.
    best = int(np.flatnonzero(objective == objective.max())[0])
    return {
        'threshold': float(curve['threshold'][best]),
        'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
        'tie_break': 'highest_threshold',
        'validation_balanced_open_set_accuracy': float(objective[best]),
    }


def evaluate_open_set(labels, predictions, scores, threshold: float):
    labels, predictions, scores = validated_arrays(labels, predictions, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    known = labels >= 0
    unknown = ~known
    accepted = scores >= threshold
    correct = predictions == labels
    n_known, n_unknown = int(known.sum()), int(unknown.sum())
    rate = lambda mask, population: float(mask[population].mean()) if population.any() else None
    ccr = rate(accepted & correct, known)
    unknown_reject = rate(~accepted, unknown)
    both = bool(n_known and n_unknown)
    curve = operating_curve(labels, predictions, scores) if both else None
    return {
        'sample_count': len(labels), 'known_count': n_known, 'unknown_count': n_unknown,
        'threshold': float(threshold),
        'known_closed_set_accuracy': rate(correct, known),
        'known_correct_accept_rate': ccr,
        'known_false_reject_rate': rate(~accepted, known),
        'known_misidentification_rate': rate(accepted & ~correct, known),
        'unknown_false_accept_rate': rate(accepted, unknown),
        'unknown_reject_rate': unknown_reject,
        'balanced_open_set_accuracy': .5 * (ccr + unknown_reject) if both else None,
        'auroc_known_positive': float(roc_auc_score(known, scores)) if both else None,
        'oscr_auc': float(np.trapezoid(curve['known_correct_accept_rate'],
                                       curve['unknown_false_accept_rate'])) if both else None,
    }
