import sys
from pathlib import Path
import unittest
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from set_methods import MedianIQR, outcome_metrics, partition, vote
from run_set_experiment import validate_config


class SetMethodTests(unittest.TestCase):
    def test_config_cannot_claim_a_different_method(self):
        config = json.loads((Path(__file__).resolve().parents[1] / '実験条件.json').read_text(encoding='utf-8'))
        validate_config(config)
        with self.assertRaises(ValueError):
            validate_config({**config, 'features': 'raw_IQ'})

    def test_sparse_relu_features_remain_finite_and_separate(self):
        x = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 10, 0], [0, 11, 0], [0, 12, 0]])
        model = MedianIQR(2).fit(x, np.array([0, 0, 0, 1, 1, 1]))
        self.assertTrue(np.all(model.scales > 0))
        _, predictions, scores, _ = model.infer([np.array([[0, 1, 0], [0, 1, 0], [0, 1000, 0]]), x[3:]])
        np.testing.assert_array_equal(predictions, [0, 1])
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[0], 0)

    def test_reject_unknown_training_and_constant_features(self):
        with self.assertRaises(ValueError):
            MedianIQR(2).fit([[1, 2], [3, 4]], np.array([0, -1]))
        with self.assertRaises(ValueError):
            MedianIQR(2).fit(np.ones((4, 2)), np.array([0, 0, 1, 1]))

    def test_vote_counts_unknown_and_rejects_all_ties(self):
        self.assertEqual(vote(np.array([1, 1, 0, -1]), 2), (1, False))
        self.assertEqual(vote(np.array([0, 1]), 2), (-1, True))
        self.assertEqual(vote(np.array([-1, -1, 0]), 2), (-1, False))
        for label in (-1, 0, 1):
            self.assertEqual(vote(np.array([label]), 2), (label, False))

    def test_partition_is_disjoint_nested_and_label_independent(self):
        rows = [{'relative_path': 'a.npz', 'signal_index': i, 'tx_id': 'a',
                 'rx_id': 'r', 'capture_date': 'd', 'true_label': 0} for i in range(40)]
        small, large = partition(rows, 5, 42), partition(rows, 20, 42)
        self.assertEqual([i for s in small[:4] for i in s['indices']], large[0]['indices'])
        changed = [{**r, 'true_label': -1} for r in rows]
        self.assertEqual(large, partition(changed, 20, 42))
        self.assertEqual(sorted(i for s in large for i in s['indices']), list(range(40)))
        with self.assertRaises(ValueError):
            partition(rows, 30, 42)

    def test_metrics_separate_false_reject_misidentification_and_unknown(self):
        metrics = outcome_metrics(np.array([0, 0, 1, -1, -1]), np.array([0, -1, 0, -1, 1]))
        self.assertAlmostEqual(metrics['known_correct_accept_rate'], 1 / 3)
        self.assertEqual(metrics['known_false_reject_count'], 1)
        self.assertEqual(metrics['known_misidentification_count'], 1)
        self.assertEqual(metrics['unknown_false_accept_rate'], .5)
        self.assertAlmostEqual(metrics['balanced_open_set_accuracy'], (1 / 3 + .5) / 2)


if __name__ == '__main__':
    unittest.main()
