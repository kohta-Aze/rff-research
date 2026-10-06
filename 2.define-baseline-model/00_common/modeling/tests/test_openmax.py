import sys
import unittest
from pathlib import Path

import libmr
import numpy as np
from scipy.special import softmax
from scipy.stats import weibull_min

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from openmax import OpenMax, openmax_scores, select_openmax_threshold


class ConstantTail:
    def w_score_vector(self, distances):
        return np.full(len(distances), .5)


class OpenMaxTests(unittest.TestCase):
    def fixture(self):
        first = np.column_stack([np.arange(6., 16.), np.linspace(.1, 1., 10)])
        return np.concatenate([first, first[:, ::-1]]), np.repeat([0, 1], 10)

    def test_only_correct_training_activations_define_means(self):
        values, labels = self.fixture()
        model = OpenMax(2, 3).fit(np.vstack([values, [0., 40.]]), np.r_[labels, 0])
        np.testing.assert_allclose(model.means, [values[:10].mean(axis=0), values[10:].mean(axis=0)])
        self.assertEqual(model.class_counts.tolist(), [10, 10])
        self.assertFalse(model.correct_mask[-1])

    def test_libmr_translation_and_serialization(self):
        values, labels = self.fixture()
        model = OpenMax(2, 5).fit(values, labels)
        distances = np.ascontiguousarray([0., .1, .3, .9, 2.])
        for fitted in model.models:
            scale, shape, sign, translate, small = fitted.get_params()
            expected = weibull_min.cdf(distances * sign + translate - small, shape, scale=scale)
            np.testing.assert_allclose(fitted.w_score_vector(distances), expected, atol=1e-12)
            restored = libmr.load_from_string(str(fitted))
            np.testing.assert_allclose(restored.w_score_vector(distances), expected, atol=1e-12)

    def test_rank_weights_and_unknown_activation_conservation(self):
        model = OpenMax(2, 2, distance='euclidean')
        model.means = np.eye(2)
        model.models = [ConstantTail(), ConstantTail()]
        np.testing.assert_allclose(model.probabilities([[4., 2.]], 1), [[1/3, 1/3, 1/3]])
        np.testing.assert_allclose(model.probabilities([[4., 2.]], 2), [softmax([2., 1.5, 2.5])])
        np.testing.assert_allclose(model.probabilities([[4., -2.]], 2), [softmax([2., -1.5, 1.5])])

    def test_large_activations_remain_normalized(self):
        model = OpenMax(2, 2, distance='euclidean')
        model.means = np.eye(2)
        model.models = [ConstantTail(), ConstantTail()]
        probabilities = model.probabilities([[1000., -1000.]], 2)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.])

    def test_unknown_winner_and_unknown_tie_are_rejected(self):
        predictions, scores, reject = openmax_scores([[.6, .1, .3], [.2, .1, .7], [.4, .2, .4]])
        np.testing.assert_allclose(scores, [.6, -.7, -.4])
        self.assertEqual(reject.tolist(), [False, True, True])
        self.assertEqual(predictions.tolist(), [0, 0, 0])

    def test_threshold_cannot_accept_an_unknown_winner(self):
        choice = select_openmax_threshold([0, -1, 0, -1], [0, 0, 0, 0], [.8, .2, -.5, -.9])
        self.assertGreater(choice['threshold'], 0.)
        self.assertEqual(choice['threshold'], .8)
        choice = select_openmax_threshold([0, -1], [0, 0], [-.6, -.7])
        self.assertGreater(choice['threshold'], 0.)
        self.assertEqual(choice['validation_balanced_open_set_accuracy'], .5)

    def test_invalid_calibration_stops(self):
        values, labels = self.fixture()
        for bad_labels, tail in [(np.r_[[-1], labels[1:]], 3), (np.zeros(20, dtype=int), 3), (labels, 11)]:
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                OpenMax(2, tail).fit(values, bad_labels)
        with self.assertRaises(ValueError):
            OpenMax(2, 3).fit(np.zeros_like(values), labels)


if __name__ == '__main__':
    unittest.main()
