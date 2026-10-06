import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cnn_backbone import IQClassifier
from supcon import SupConEncoder, augment_iq, supervised_contrastive_loss


class SupConTests(unittest.TestCase):
    def test_loss_matches_independent_pair_enumeration(self):
        x = torch.randn((4, 2, 5), generator=torch.Generator().manual_seed(4), dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 2])
        vectors = F.normalize(x, dim=2).transpose(0, 1).reshape(8, 5)
        targets = labels.repeat(2)
        losses = []
        for i in range(8):
            contrasts = [j for j in range(8) if i != j]
            positives = [j for j in contrasts if targets[i] == targets[j]]
            denominator = torch.logsumexp(torch.stack([vectors[i] @ vectors[j] / .11 for j in contrasts]), 0)
            losses.append(torch.stack([denominator - vectors[i] @ vectors[j] / .11 for j in positives]).mean())
        expected = torch.stack(losses).mean() * .11 / .07
        actual = supervised_contrastive_loss(x, labels, .11, .07)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_self_excluded_and_counterpart_positive_exists(self):
        x = torch.ones(1, 2, 3)
        self.assertAlmostEqual(float(supervised_contrastive_loss(x, torch.tensor([0]))), 0.)
        x = torch.ones(3, 2, 3)
        self.assertAlmostEqual(float(supervised_contrastive_loss(x, torch.tensor([0, 1, 2]))), math.log(5), places=5)

    def test_separated_classes_have_lower_loss(self):
        separated = torch.eye(2).unsqueeze(1).repeat(1, 2, 1)
        collapsed = torch.ones_like(separated)
        labels = torch.tensor([0, 1])
        self.assertLess(float(supervised_contrastive_loss(separated, labels)), float(supervised_contrastive_loss(collapsed, labels)))

    def test_invalid_loss_inputs_stop(self):
        x = torch.ones(2, 2, 3)
        for labels, temperature in [(torch.tensor([0, -1]), .07), (torch.tensor([0., 1.]), .07), (torch.tensor([0, 1]), 0.)]:
            with self.assertRaises(ValueError):
                supervised_contrastive_loss(x, labels, temperature)
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(torch.zeros_like(x), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(x[:, :1], torch.tensor([0, 1]))

    def test_augmentation_is_reproducible_independent_and_unit_power(self):
        iq = torch.randn(8, 2, 256, generator=torch.Generator().manual_seed(2))
        saved = iq.clone()
        generator = torch.Generator().manual_seed(5)
        first = augment_iq(iq, generator)
        second = augment_iq(iq, generator)
        torch.testing.assert_close(first, augment_iq(iq, torch.Generator().manual_seed(5)))
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(first.square().sum(1).mean(1), torch.ones(8))
        torch.testing.assert_close(iq, saved)
        with self.assertRaises(ValueError):
            augment_iq(torch.zeros_like(iq), generator)

    def test_noise_and_rotation_are_bounded_as_configured(self):
        iq = torch.zeros(256, 2, 256)
        iq[:, 0] = 1.
        rotated = augment_iq(iq, torch.Generator().manual_seed(6), 10., 100.)
        angles = torch.atan2(rotated[:, 1].mean(1), rotated[:, 0].mean(1))
        self.assertLessEqual(float(angles.abs().max()), math.radians(10) + 1e-5)
        noisy = augment_iq(iq, torch.Generator().manual_seed(6), 0., 30.)
        noise_power = float((noisy - iq).square().sum(1).mean())
        self.assertAlmostEqual(noise_power, .001, delta=.00005)

    def test_shared_encoder_initialization_and_projection_gradient(self):
        architecture = dict(channels=[4], kernel_sizes=[3], feature_dim=8, dropout=.2)
        torch.manual_seed(42)
        baseline = IQClassifier(2, **architecture)
        torch.manual_seed(42)
        model = SupConEncoder(2, architecture, 6)
        for k, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.encoder.state_dict()[k])
        iq = torch.randn(4, 2, 256)
        projection = model(iq)
        self.assertEqual(tuple(projection.shape), (4, 6))
        torch.testing.assert_close(projection.norm(dim=1), torch.ones(4))
        loss = supervised_contrastive_loss(projection.reshape(2, 2, 6), torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(model.encoder.convolutions[0].weight.grad)
        self.assertIsNotNone(model.projection[0].weight.grad)
        self.assertIsNone(model.encoder.classifier.weight.grad)


if __name__ == '__main__':
    unittest.main()
