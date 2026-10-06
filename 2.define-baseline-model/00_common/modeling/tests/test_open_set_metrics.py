"""Small cases with independently known open-set results."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from open_set_metrics import evaluate_open_set, operating_curve, select_threshold


class OpenSetMetricsTests(unittest.TestCase):
    def test_perfect_separation(self):
        metrics = evaluate_open_set([0, 1, -1, -1], [0, 1, 0, 1], [.9, .8, .3, .2], .8)
        for key in ['auroc_known_positive', 'oscr_auc', 'balanced_open_set_accuracy']:
            self.assertEqual(metrics[key], 1.)
        self.assertEqual(metrics['unknown_false_accept_rate'], 0.)

    def test_tied_scores_are_chance_not_order_dependent(self):
        metrics = evaluate_open_set([0, -1, 1, -1], [0, 0, 1, 1], [.5] * 4, .5)
        self.assertEqual(metrics['auroc_known_positive'], .5)
        self.assertEqual(metrics['oscr_auc'], .5)
        self.assertEqual(metrics['unknown_false_accept_rate'], 1.)
        self.assertEqual(len(operating_curve([0, -1], [0, 0], [.5, .5])['threshold']), 2)

    def test_classification_errors_reduce_oscr(self):
        metrics = evaluate_open_set([0, 1, -1, -1], [0, 0, 0, 0], [.9, .8, .3, .2], .8)
        self.assertEqual(metrics['auroc_known_positive'], 1.)
        self.assertEqual(metrics['oscr_auc'], .5)
        self.assertEqual(metrics['known_misidentification_rate'], .5)
        self.assertEqual(metrics['balanced_open_set_accuracy'], .75)

    def test_reversed_ranking(self):
        metrics = evaluate_open_set([0, -1], [0, 0], [.1, .9], .5)
        self.assertEqual(metrics['auroc_known_positive'], 0.)
        self.assertEqual(metrics['oscr_auc'], 0.)
        self.assertEqual(metrics['balanced_open_set_accuracy'], 0.)

    def test_threshold_selects_highest_optimum(self):
        choice = select_threshold([0, 1, -1, -1], [0, 1, 0, 1], [.9, .8, .3, .2])
        self.assertEqual(choice['threshold'], .8)
        metrics = evaluate_open_set([0, 1, -1, -1], [0, 1, 0, 1], [.9, .8, .3, .2], choice['threshold'])
        self.assertEqual(metrics['balanced_open_set_accuracy'], choice['validation_balanced_open_set_accuracy'])

    def test_reject_all_threshold_is_finite_and_rejects_score_one(self):
        choice = select_threshold([0, -1], [1, 0], np.array([1., 1.], dtype=np.float32))
        self.assertTrue(np.isfinite(choice['threshold']))
        metrics = evaluate_open_set([0, -1], [1, 0], [1., 1.], choice['threshold'])
        self.assertEqual(metrics['known_false_reject_rate'], 1.)
        self.assertEqual(metrics['unknown_false_accept_rate'], 0.)

    def test_single_population_has_null_detection_metrics(self):
        metrics = evaluate_open_set([0, 1], [0, 1], [.9, .8], .85)
        self.assertIsNone(metrics['auroc_known_positive'])
        self.assertIsNone(metrics['oscr_auc'])
        self.assertEqual(metrics['known_correct_accept_rate'], .5)
        self.assertIsNone(metrics['unknown_false_accept_rate'])

    def test_invalid_inputs_stop(self):
        for labels, predictions, scores in [([], [], []), ([0], [0], [np.nan]),
                                            ([0], [-1], [.5]), ([-2], [0], [.5])]:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    evaluate_open_set(labels, predictions, scores, .5)


if __name__ == '__main__':
    unittest.main()
