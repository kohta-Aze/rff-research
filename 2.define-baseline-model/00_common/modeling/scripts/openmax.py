"""Single-channel OpenMax on logits, using native libMR FitHigh tails."""
from __future__ import annotations

import libmr
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

from open_set_metrics import operating_curve


def openmax_scores(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 2 or probabilities.shape[1] < 3 or not len(probabilities)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12)):
        raise ValueError('OpenMax requires normalized known-plus-unknown probabilities.')
    predictions = probabilities[:, :-1].argmax(axis=1)
    known_max = probabilities[:, :-1].max(axis=1)
    unknown = probabilities[:, -1]
    unknown_wins = unknown >= known_max
    # A positive threshold preserves both original OpenMax rejection conditions.
    scores = np.where(unknown_wins, -unknown, known_max)
    return predictions, scores, unknown_wins


def select_openmax_threshold(labels, predictions, scores):
    curve = operating_curve(labels, predictions, scores)
    indices = np.flatnonzero(curve['threshold'] > 0)
    objective = .5 * (curve['known_correct_accept_rate'] + 1 - curve['unknown_false_accept_rate'])
    if len(indices):
        best = int(indices[np.argmax(objective[indices])])
        threshold, value = float(curve['threshold'][best]), float(objective[best])
    else:
        threshold, value = float(np.nextafter(0., 1.)), .5
    return {'threshold': threshold, 'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
            'tie_break': 'highest_positive_threshold',
            'validation_balanced_open_set_accuracy': value,
            'selected_from': ['validation_known', 'validation_unknown']}


class OpenMax:
    def __init__(self, class_count, tail_size, distance='eucos', euclidean_scale=200.):
        if type(class_count) is not int or class_count < 2 or type(tail_size) is not int or tail_size < 2:
            raise ValueError('Class count and tail size must be integers >= 2.')
        if distance not in ['eucos', 'euclidean', 'cosine'] or not np.isfinite(euclidean_scale) or euclidean_scale <= 0:
            raise ValueError('Invalid distance or Euclidean scale.')
        self.class_count, self.tail_size = class_count, tail_size
        self.distance, self.euclidean_scale = distance, euclidean_scale
        self.means = None
        self.models = []
        self.tails = None
        self.correct_mask = None
        self.class_counts = None

    def _activations(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.class_count or not len(values) or not np.isfinite(values).all():
            raise ValueError('Activation vectors must be a finite [sample, class] matrix.')
        if self.distance in ['eucos', 'cosine'] and np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError('Cosine distance cannot use zero activation vectors.')
        return values

    def distances(self, activations):
        values = self._activations(activations)
        if self.means is None:
            raise ValueError('Fit mean activation vectors first.')
        distances = np.zeros((len(values), self.class_count), dtype=np.float64)
        if self.distance in ['eucos', 'euclidean']:
            distances += cdist(values, self.means, metric='euclidean') / self.euclidean_scale
        if self.distance in ['eucos', 'cosine']:
            distances += cdist(values, self.means, metric='cosine').clip(0., 2.)
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite distance from class means.')
        return distances

    def fit(self, activations, labels):
        values = self._activations(activations)
        labels = np.asarray(labels)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in 'iu'
                or np.any(labels < 0) or np.any(labels >= self.class_count)):
            raise ValueError('Only known training labels may calibrate OpenMax.')
        correct = values.argmax(axis=1) == labels
        counts = np.bincount(labels[correct], minlength=self.class_count)
        if np.any(counts < self.tail_size):
            raise ValueError('Insufficient correctly classified training signals for a tail.')
        means = np.stack([values[correct & (labels == label)].mean(axis=0)
                          for label in range(self.class_count)])
        if not np.isfinite(means).all() or np.any(np.linalg.norm(means, axis=1) == 0):
            raise ValueError('Invalid mean activation vector.')
        self.means = means
        distances = self.distances(values)
        models, tails = [], []
        for label in range(self.class_count):
            tail = np.sort(distances[correct & (labels == label), label])[-self.tail_size:]
            model = libmr.MR()
            model.fit_high(tail, self.tail_size)
            params = np.asarray(model.get_params(), dtype=np.float64)
            if not model.is_valid or not np.isfinite(params).all() or np.any(params[:2] <= 0):
                raise ValueError(f'libMR tail fit failed for class {label}.')
            models.append(model)
            tails.append(tail)
        self.models, self.tails = models, np.stack(tails)
        self.correct_mask, self.class_counts = correct, counts
        return self

    def probabilities(self, activations, alpha_rank):
        values = self._activations(activations)
        if len(self.models) != self.class_count:
            raise ValueError('Fit Weibull models before inference.')
        if type(alpha_rank) is not int or not 1 <= alpha_rank <= self.class_count:
            raise ValueError('Alpha rank must be between 1 and the known class count.')
        distances = self.distances(values)
        outlier = np.column_stack([model.w_score_vector(np.ascontiguousarray(distances[:, label]))
                                   for label, model in enumerate(self.models)])
        if not np.isfinite(outlier).all() or np.any(outlier < 0) or np.any(outlier > 1):
            raise ValueError('Invalid libMR outlier probabilities.')
        ranked = np.argsort(-values, axis=1, kind='stable')[:, :alpha_rank]
        weights = np.zeros_like(values)
        weights[np.arange(len(values))[:, None], ranked] = np.arange(alpha_rank, 0, -1) / alpha_rank
        revised = values * (1 - weights * outlier)
        unknown_activation = (values - revised).sum(axis=1, keepdims=True)
        probabilities = softmax(np.concatenate([revised, unknown_activation], axis=1), axis=1)
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite OpenMax probabilities.')
        return probabilities

    def state(self):
        return {'class_count': self.class_count, 'tail_size': self.tail_size,
                'distance': self.distance, 'euclidean_scale': self.euclidean_scale,
                'mean_activations': self.means.tolist(), 'tail_distances': self.tails.tolist(),
                'correct_training_class_counts': self.class_counts.tolist(),
                'weibull_params': [list(model.get_params()) for model in self.models],
                'weibull_model_strings': [str(model) for model in self.models]}
