"""Global GPD fit on true-class training distances (MTPL Algorithm 2)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import genpareto

from open_set_metrics import operating_curve, evaluate_open_set


def squared_distances(features, prototypes):
    features = np.asarray(features, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    if (features.ndim != 2 or prototypes.ndim != 2 or not len(features) or len(prototypes) < 2
            or features.shape[1] != prototypes.shape[1] or features.shape[1] < 1
            or not np.isfinite(features).all() or not np.isfinite(prototypes).all()):
        raise ValueError('Finite feature and prototype matrices with matching dimensions required.')
    distances = np.square(features[:, None, :] - prototypes[None, :, :]).sum(2)
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite squared distances.')
    return distances


def select_gpd_threshold(labels, predictions, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('GPD scores must be probabilities.')
    curve = operating_curve(labels, predictions, scores)
    valid = (curve['threshold'] > 0) & (curve['threshold'] <= 1)
    thresholds = np.r_[1., curve['threshold'][valid]]
    objective = np.r_[evaluate_open_set(labels, predictions, scores, 1.)['balanced_open_set_accuracy'],
                      .5 * (curve['known_correct_accept_rate'][valid] + 1 - curve['unknown_false_accept_rate'][valid])]
    best = int(np.argmax(objective))
    return {'threshold': float(thresholds[best]),
            'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
            'tie_break': 'highest_threshold', 'allowed_range': '0 < delta <= 1',
            'validation_balanced_open_set_accuracy': float(objective[best]),
            'selected_from': ['validation_known', 'validation_unknown']}


class GlobalGPD:
    def __init__(self, quantile=.9, minimum_tail=20):
        if not np.isfinite(quantile) or not 0 < quantile < 1 or type(minimum_tail) is not int or minimum_tail < 3:
            raise ValueError('Invalid tail quantile or minimum tail count.')
        self.quantile, self.minimum_tail = quantile, minimum_tail
        self.threshold = self.shape = self.scale = None
        self.training_distances = self.excesses = None

    def fit(self, features, labels, prototypes):
        distances = squared_distances(features, prototypes)
        labels = np.asarray(labels)
        if (labels.ndim != 1 or len(labels) != len(distances) or labels.dtype.kind not in 'iu'
                or np.any(labels < 0) or np.any(labels >= distances.shape[1])):
            raise ValueError('Only known training labels may fit GPD.')
        self.training_distances = distances[np.arange(len(labels)), labels]
        self.threshold = float(np.quantile(self.training_distances, self.quantile, method='linear'))
        self.excesses = self.training_distances[self.training_distances > self.threshold] - self.threshold
        if len(self.excesses) < self.minimum_tail or np.ptp(self.excesses) <= 0:
            raise ValueError('Insufficient or degenerate training tail.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            shape, location, scale = genpareto.fit(self.excesses, floc=0.)
        if (not np.isfinite([shape, location, scale]).all() or location != 0 or scale <= 0
                or not np.isfinite(genpareto.logpdf(self.excesses, shape, loc=0., scale=scale)).all()):
            raise ValueError('Invalid GPD MLE fit or tail support.')
        self.shape, self.scale = float(shape), float(scale)
        return self

    def scores(self, minimum_distances):
        if self.shape is None:
            raise ValueError('Fit GPD before scoring.')
        distances = np.asarray(minimum_distances, dtype=np.float64)
        if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or np.any(distances < 0):
            raise ValueError('Nonnegative finite minimum squared distances required.')
        scores = np.ones_like(distances)
        tail = distances > self.threshold
        scores[tail] = genpareto.sf(distances[tail] - self.threshold, self.shape, loc=0., scale=self.scale)
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError('Invalid GPD survival probabilities.')
        return scores

    def state(self):
        if self.shape is None:
            raise ValueError('Fit GPD before saving state.')
        return {'quantile': self.quantile, 'quantile_method': 'linear', 'omega': self.threshold,
                'shape': self.shape, 'scale': self.scale, 'location': 0., 'fit': 'scipy_genpareto_mle_floc_zero',
                'training_count': len(self.training_distances), 'tail_count': len(self.excesses),
                'score': 'one_in_body_else_conditional_gpd_survival_probability'}
