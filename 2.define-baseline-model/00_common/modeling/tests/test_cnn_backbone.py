import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cnn_backbone import IQClassifier


class ClassifierTests(unittest.TestCase):
    def test_six_class_forward_features_and_gradient(self):
        model = IQClassifier(6, [8, 16], [7, 3], 12, .2)
        iq = torch.randn(4, 2, 256)
        self.assertEqual(model.features(iq).shape, (4, 12))
        logits = model(iq)
        self.assertEqual(logits.shape, (4, 6))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 5])).backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_iq_layout_must_be_explicit(self):
        model = IQClassifier(6, [8], [3], 12, 0.)
        with self.assertRaises(ValueError):
            model(torch.randn(4, 256, 2))


if __name__ == '__main__':
    unittest.main()
