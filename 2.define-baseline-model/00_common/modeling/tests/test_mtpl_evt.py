from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import genpareto
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cnn_backbone import IQClassifier
from mtpl import MultiTaskPrototype, multitask_loss
from gpd_evt import GlobalGPD, squared_distances, select_gpd_threshold


class MTPLEVTTests(unittest.TestCase):
    def test_loss_sums_coordinates_and_updates_all_tasks(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        iq = torch.zeros(2, 2, 256)
        reconstruction = torch.ones_like(iq, requires_grad=True)
        features = torch.ones(2, 3, requires_grad=True)
        prototypes = torch.zeros(2, 3, requires_grad=True)
        total, parts = multitask_loss(logits, reconstruction, features, prototypes, iq, torch.tensor([0, 1]))
        self.assertAlmostEqual(float(parts['classification'].detach()), np.log(2), places=6)
        self.assertEqual(float(parts['reconstruction'].detach()), 512.)
        self.assertEqual(float(parts['prototype'].detach()), 3.)
        self.assertAlmostEqual(float(total.detach()), np.log(2) + .1 * 512 + .05 * 3, places=5)
        total.backward()
        for tensor in [logits, reconstruction, features, prototypes]:
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            multitask_loss(logits, reconstruction, features, prototypes, iq, torch.tensor([0, -1]))

    def test_common_initial_encoder_decoder_shape_and_gradients(self):
        architecture = dict(channels=[4, 8, 16], kernel_sizes=[7, 5, 3], feature_dim=8, dropout=.2)
        torch.manual_seed(42); baseline = IQClassifier(2, **architecture)
        torch.manual_seed(42); model = MultiTaskPrototype(2, architecture)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.encoder.state_dict()[key])
        iq = torch.randn(4, 2, 256)
        logits, reconstruction, features = model(iq)
        self.assertEqual(reconstruction.shape, iq.shape)
        loss, _ = multitask_loss(logits, reconstruction, features, model.prototypes, iq, torch.tensor([0, 1, 0, 1]))
        loss.backward()
        for parameter in [model.encoder.convolutions[0].weight, model.encoder.classifier.weight,
                          model.decoder_input[0].weight, model.decoder[0].weight, model.prototypes]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_squared_distance_not_cosine_and_zero_features_allowed(self):
        np.testing.assert_array_equal(squared_distances([[0., 0.], [2., 1.]], [[0., 0.], [1., 1.]]), [[0., 2.], [5., 1.]])
        with self.assertRaises(ValueError):
            squared_distances([[np.nan]], [[0.], [1.]])

    def fixture(self):
        values = np.linspace(.1, 4., 200)
        return values[:, None], np.zeros(200, dtype=np.int64), np.array([[0.], [100.]])

    def test_tail_uses_true_class_training_not_nearest_prototype(self):
        features, labels, prototypes = self.fixture()
        labels[-1] = 1
        model = GlobalGPD(.9, 10).fit(features, labels, prototypes)
        expected = np.square(features[:, 0] - prototypes[labels, 0])
        np.testing.assert_array_equal(model.training_distances, expected)
        self.assertEqual(model.threshold, float(np.quantile(expected, .9)))
        np.testing.assert_array_equal(model.excesses, expected[expected > model.threshold] - model.threshold)
        shape, location, scale = genpareto.fit(model.excesses, floc=0.)
        self.assertEqual((model.shape, model.scale), (shape, scale))
        self.assertEqual(location, 0.)

    def test_body_exponential_tail_and_negative_shape_endpoint(self):
        model = GlobalGPD(); model.threshold, model.shape, model.scale = 2., 0., 3.
        np.testing.assert_allclose(model.scores([0., 2., 5.]), [1., 1., np.exp(-1)])
        model.shape = -.5
        np.testing.assert_allclose(model.scores([2., 5., 8., 10.]), [1., .25, 0., 0.])

    def test_positive_shape_and_monotonicity(self):
        model = GlobalGPD(); model.threshold, model.shape, model.scale = 1., .5, 2.
        distances = np.arange(1., 20.)
        expected = (1 + .5 * (distances - 1) / 2.) ** -2
        np.testing.assert_allclose(model.scores(distances), expected)
        self.assertTrue(np.all(np.diff(model.scores(distances)) < 0))

    def test_operational_threshold_preserves_body_acceptance(self):
        choice = select_gpd_threshold([0, -1, -1], [0, 0, 0], [1., 1., .2])
        self.assertEqual(choice['threshold'], 1.)
        choice = select_gpd_threshold([0, -1], [1, 0], [1., 1.])
        self.assertEqual(choice['threshold'], 1.)
        self.assertEqual(choice['validation_balanced_open_set_accuracy'], 0.)
        choice = select_gpd_threshold([0, -1], [0, 0], [.9, .1])
        self.assertEqual(choice['threshold'], .9)

    def test_invalid_or_degenerate_calibration_stops(self):
        features, labels, prototypes = self.fixture()
        with self.assertRaises(ValueError):
            GlobalGPD().fit(features, np.full_like(labels, -1), prototypes)
        with self.assertRaises(ValueError):
            GlobalGPD().fit(np.zeros_like(features), labels, prototypes)
        with self.assertRaises(ValueError):
            GlobalGPD().scores([1.])
        model = GlobalGPD(.9, 10).fit(features, labels, prototypes)
        with self.assertRaises(ValueError):
            model.scores([-1.])


if __name__ == '__main__':
    unittest.main()
