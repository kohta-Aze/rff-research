import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cosine_prototype import CosinePrototype


class CosinePrototypeTests(unittest.TestCase):
    def test_raw_mean_precedes_normalization(self):
        model = CosinePrototype(2).fit([[10., 0.], [0., 1.], [-1., 0.]], [0, 0, 1])
        expected = torch.tensor([10., 1.], dtype=torch.float64) / (101. ** .5)
        torch.testing.assert_close(model.prototypes[0], expected)
        self.assertEqual(model.counts.tolist(), [2, 1])

    def test_cosine_is_scale_invariant_and_can_be_negative(self):
        model = CosinePrototype(2).fit([[1., 0.], [0., 1.]], [0, 1])
        similarities = model.similarities([[5., 0.], [1., 0.], [-1., 0.]])
        torch.testing.assert_close(similarities, torch.tensor([[1., 0.], [1., 0.], [-1., 0.]], dtype=torch.float64))
        self.assertEqual(similarities.max(dim=1).indices.tolist(), [0, 0, 1])

    def test_unknown_and_missing_classes_are_rejected(self):
        for labels in [[0, -1], [0, 0], [0, 2]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                CosinePrototype(2).fit([[1., 0.], [0., 1.]], labels)

    def test_undefined_directions_and_nonfinite_features_stop(self):
        for features, labels in [([[0., 0.], [0., 1.]], [0, 1]),
                                  ([[float('nan'), 0.], [0., 1.]], [0, 1]),
                                  ([[1., 0.], [-1., 0.], [0., 1.]], [0, 0, 1])]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                CosinePrototype(2).fit(features, labels)

    def test_unfitted_and_mismatched_inference_stop(self):
        with self.assertRaises(ValueError):
            CosinePrototype(2).similarities([[1., 0.]])
        model = CosinePrototype(2).fit([[1., 0.], [0., 1.]], [0, 1])
        with self.assertRaises(ValueError):
            model.similarities([[1., 0., 0.]])


if __name__ == '__main__':
    unittest.main()
