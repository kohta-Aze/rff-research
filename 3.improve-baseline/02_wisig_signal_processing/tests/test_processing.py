import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from transforms import relative_l2, residual_emphasis, select_candidate
from run_frozen import restore_calibration, validate_config
from paths import STUDY, read_json
from openmax import OpenMax


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.iq = np.random.default_rng(42).normal(size=(3, 256, 2)).astype(np.float32)
        self.iq /= np.sqrt(np.mean(np.sum(self.iq ** 2, axis=-1), axis=-1))[:, None, None]
        self.config = read_json(STUDY / 'configs/residual_grid.json')

    def test_identity_is_bitwise_equal_and_does_not_alias(self):
        result = residual_emphasis(self.iq, 0)
        np.testing.assert_array_equal(result, self.iq)
        self.assertFalse(np.shares_memory(result, self.iq))
        np.testing.assert_array_equal(relative_l2(self.iq, result), np.zeros(3))

    def test_windows_are_independent_and_order_is_preserved(self):
        batched = residual_emphasis(self.iq, .25)
        separate = np.concatenate([residual_emphasis(row[None], .25) for row in self.iq])
        np.testing.assert_array_equal(batched, separate)
        np.testing.assert_array_equal(batched[::-1], residual_emphasis(self.iq[::-1], .25))

    def test_constant_waveform_survives_and_positive_gain_is_removed(self):
        constant = np.ones((1, 256, 2), dtype=np.float32) / np.sqrt(2)
        np.testing.assert_allclose(residual_emphasis(constant, .5), constant, atol=1e-7)
        np.testing.assert_allclose(residual_emphasis(self.iq * 3, .25), residual_emphasis(self.iq, .25), atol=1e-7)

    def test_processed_power_is_one_and_input_is_unchanged(self):
        before = self.iq.copy()
        result = residual_emphasis(self.iq, .25)
        np.testing.assert_allclose(np.mean(np.sum(result ** 2, axis=-1), axis=-1), 1, atol=1e-7)
        np.testing.assert_array_equal(before, self.iq)
        self.assertTrue((relative_l2(self.iq, result) > 0).all())

    def test_invalid_inputs_stop(self):
        for value in [self.iq[:, :255], self.iq[:0], np.zeros_like(self.iq), self.iq * np.nan]:
            with self.assertRaises(ValueError):
                residual_emphasis(value, 0)
        for strength in [np.nan, np.inf, .6, -.6]:
            with self.assertRaises(ValueError):
                residual_emphasis(self.iq, strength)

    def test_validation_guard_rejects_better_oscr_with_worse_far(self):
        baseline = dict(strength=0., relative_l2_max=0., metrics=dict(oscr_auc=.5, unknown_false_accept_rate=.1, known_closed_set_accuracy=.9))
        unsafe = dict(strength=.25, relative_l2_max=.05, metrics=dict(oscr_auc=.9, unknown_false_accept_rate=.2, known_closed_set_accuracy=.9))
        self.assertIs(select_candidate([baseline, unsafe], self.config), baseline)
        unsafe['metrics']['unknown_false_accept_rate'] = .1
        unsafe['relative_l2_max'] = .11
        self.assertIs(select_candidate([baseline, unsafe], self.config), baseline)

    def test_grid_order_keeps_identity_on_tie(self):
        first = dict(strength=0., relative_l2_max=0., metrics=dict(oscr_auc=.5, unknown_false_accept_rate=.1, known_closed_set_accuracy=.9))
        second = dict(first, strength=.1)
        self.assertIs(select_candidate([first, second], self.config), first)

    def test_saved_openmax_probabilities_are_preserved(self):
        first = np.column_stack([np.arange(6., 16.), np.linspace(.1, 1., 10)])
        logits, labels = np.vstack([first, first[:, ::-1]]), np.repeat([0, 1], 10)
        calibration = OpenMax(2, 5).fit(logits, labels)
        restored = restore_calibration(calibration.state())
        np.testing.assert_allclose(restored.probabilities(logits, 2), calibration.probabilities(logits, 2), atol=1e-12)

    def test_protocol_rejects_changed_weight_or_calibration_mode(self):
        validate_config(self.config)
        for key in ['cnn_retrained', 'openmax_recalibrated']:
            with self.assertRaises(ValueError):
                validate_config(dict(self.config, **{key: True}))


if __name__ == '__main__':
    unittest.main()
