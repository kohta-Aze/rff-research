"""Behavioral guarantees, leakage boundaries and delayed-detection limitations."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from attempt_monitor import Attempt, AttemptMonitor, FeatureReference, Policy, waveform_digest


def temporary_output():
    root = (Path(__file__).resolve().parents[1] / 'output' / 'test_tmp').resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = tempfile.TemporaryDirectory(dir=root)
    if not Path(folder.name).resolve().is_relative_to(root):
        raise ValueError('Temporary test output escaped its workspace directory.')
    return folder


def reference():
    value = FeatureReference()
    value.centers = np.eye(3)
    value.thresholds = np.full(3, 0.2)
    value.jump_threshold = 0.2
    return value


def attempt(t, claim=0, feature=None, token='receiver-track', digest=None):
    return Attempt(token, float(t), claim, np.eye(3)[claim] if feature is None else feature,
                   digest if digest is not None else f'{int(t):064x}')


class MonitorTests(unittest.TestCase):
    def test_id_sweep_uses_observation_id_not_claim_as_key(self):
        monitor = AttemptMonitor(reference(), Policy())
        self.assertFalse(monitor.observe(attempt(1, 0))['on_watchlist'])
        self.assertFalse(monitor.observe(attempt(2, 1))['on_watchlist'])
        result = monitor.observe(attempt(3, 2))
        self.assertIn('id_sweep', result['trigger_reasons'])
        self.assertEqual(set(monitor.tracks), {'receiver-track'})
        self.assertFalse(monitor.observe(attempt(4, 0, token='another-receiver-track'))['on_watchlist'])

    def test_single_success_can_precede_listing(self):
        monitor = AttemptMonitor(reference(), Policy())
        rows = [monitor.observe(attempt(t, digest='a' * 64)) for t in [1, 2, 3]]
        self.assertEqual([row['on_watchlist'] for row in rows], [False, False, True])
        self.assertEqual([True and not row['on_watchlist'] for row in rows], [True, True, False])

    def test_fresh_tracking_tokens_evade_history_rules(self):
        monitor = AttemptMonitor(reference(), Policy())
        for t in range(1, 20):
            self.assertFalse(monitor.observe(attempt(t, t % 3, token=f'track-{t}', digest='a' * 64))['on_watchlist'])

    def test_feature_changes_and_reference_mismatch_are_distinct(self):
        mismatch = AttemptMonitor(reference(), Policy())
        for t in range(1, 4):
            result = mismatch.observe(attempt(t, feature=np.eye(3)[1]))
        self.assertIn('reference_mismatch', result['trigger_reasons'])
        self.assertNotIn('feature_changes', result['trigger_reasons'])
        changes = AttemptMonitor(reference(), Policy(feature_window=4, feature_hit_limit=3))
        for t in range(1, 5):
            result = changes.observe(attempt(t, feature=np.eye(3)[t % 2]))
        self.assertIn('feature_changes', result['trigger_reasons'])

    def test_normal_noise_does_not_immediately_list_sender(self):
        monitor = AttemptMonitor(reference(), Policy())
        for t in range(1, 30):
            value = np.array([1., .01 * (-1) ** t, .01])
            self.assertFalse(monitor.observe(attempt(t, feature=value))['on_watchlist'])

    def test_history_window_and_list_expiry(self):
        monitor = AttemptMonitor(reference(), Policy(window_seconds=5, watch_seconds=10))
        monitor.observe(attempt(1, 0))
        monitor.observe(attempt(2, 1))
        self.assertFalse(monitor.observe(attempt(8, 2))['on_watchlist'])
        monitor.observe(attempt(9, 0))
        self.assertTrue(monitor.observe(attempt(10, 1))['on_watchlist'])
        self.assertTrue(monitor.observe(attempt(16, 1))['on_watchlist'])
        result = monitor.observe(attempt(21, 1))
        self.assertFalse(result['on_watchlist'])
        self.assertEqual(result['active_reasons'], '')

    def test_feature_window_removes_old_outliers(self):
        monitor = AttemptMonitor(reference(), Policy(feature_window=3, feature_hit_limit=3))
        for t in [1, 2]:
            monitor.observe(attempt(t, feature=np.eye(3)[1]))
        for t in [3, 4, 5, 6]:
            result = monitor.observe(attempt(t))
        self.assertEqual(result['reference_hits'], 0)
        self.assertFalse(result['on_watchlist'])

    def test_bad_input_does_not_update_state(self):
        monitor = AttemptMonitor(reference(), Policy())
        monitor.observe(attempt(1))
        before = copy.deepcopy(monitor.tracks['receiver-track'])
        with self.assertRaises(ValueError):
            monitor.observe(attempt(2, claim=3, feature=np.ones(3)))
        self.assertEqual(monitor.last_timestamp, 1.)
        self.assertEqual(list(monitor.tracks['receiver-track'].history), list(before.history))
        with self.assertRaises(ValueError):
            monitor.observe(attempt(0))
        with self.assertRaises(ValueError):
            monitor.observe(attempt(2, feature=np.zeros(3)))

    def test_waveform_age_is_not_observable(self):
        fresh, old = AttemptMonitor(reference(), Policy()), AttemptMonitor(reference(), Policy())
        # Metadata names and alleged age are deliberately absent from Attempt.
        for t in range(1, 10):
            self.assertEqual(fresh.observe(attempt(t)), old.observe(attempt(t)))

    def test_exact_digest_is_not_robust_to_changed_receive_samples(self):
        iq = np.ones((2, 256), dtype=np.float32)
        original = waveform_digest(iq)
        self.assertEqual(original, waveform_digest(iq.copy()))
        iq[0, 0] += .001
        self.assertNotEqual(original, waveform_digest(iq))

    def test_references_fit_known_only_and_roundtrip(self):
        training = np.array([[1., .01], [1., -.01], [.01, 1.], [-.01, 1.]])
        labels = np.array([0, 0, 1, 1])
        fitted = FeatureReference().fit(training, labels, training, labels, ['r'] * 4, .9)
        with temporary_output() as folder:
            path = Path(folder) / 'reference.npz'
            np.savez(path, centers=fitted.centers, thresholds=fitted.thresholds)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['centers'], fitted.centers)
        with self.assertRaises(ValueError):
            FeatureReference().fit(training, np.array([0, -1, 1, 1]), training, labels, ['r'] * 4, .9)

    def test_invalid_policies(self):
        for values in [{'window_seconds': 0}, {'watch_seconds': float('nan')},
                       {'distinct_claim_limit': True}, {'feature_window': 2, 'feature_hit_limit': 3}]:
            with self.assertRaises(ValueError):
                Policy(**values)


class RunnerTests(unittest.TestCase):
    def test_claim_matching_and_offline_blocking_counts(self):
        from run_monitor_experiment import evaluate_streams
        batch = {'features': np.tile([1., 0., 0.], (3, 1)),
                 'iq': np.ones((3, 2, 256), dtype=np.float32),
                 'scores': np.array([.9, .9, .9]), 'predictions': np.array([0, 0, 0]),
                 'rows': [{'tx_id': 'unknown', 'rx_id': 'r', 'capture_date': 'old',
                           'signal_index': i, 'view': 'test', 'relative_path': 'fixture.npz'} for i in range(3)]}
        streams = [('unknown_fixed_claim', 'fixture', 'unknown',
                    [(batch, i, 0, True, False) for i in range(3)])]
        with temporary_output() as folder:
            result = evaluate_streams(streams, reference(), Policy(), {'attempt_interval_seconds': 1.},
                                      {'registered-a': 0, 'registered-b': 1, 'registered-c': 2}, .5, Path(folder))
            summary = result['scenarios'][0]
            self.assertEqual(summary['baseline_accepts'], 3)
            self.assertEqual(summary['accepts_if_list_blocks'], 2)
            self.assertEqual(summary['streams_with_success_if_list_blocks'], 1)
            self.assertTrue((Path(folder) / 'watchlist.csv').exists())
        # A high score for a different registered ID must not authenticate this claim.
        streams = [('unknown_fixed_claim', 'fixture', 'unknown', [(batch, 0, 1, True, False)])]
        with temporary_output() as folder:
            result = evaluate_streams(streams, reference(), Policy(), {'attempt_interval_seconds': 1.},
                                      {'registered-a': 0, 'registered-b': 1, 'registered-c': 2}, .5, Path(folder))
            self.assertEqual(result['scenarios'][0]['baseline_accepts'], 0)

    def test_starting_claim_can_succeed_before_detection(self):
        from run_monitor_experiment import evaluate_streams
        batch = {'features': np.tile([0., 0., 1.], (3, 1)),
                 'iq': np.stack([np.full((2, 256), i + 1, dtype=np.float32) for i in range(3)]),
                 'scores': np.full(3, .9), 'predictions': np.full(3, 2),
                 'rows': [{'tx_id': 'unknown', 'rx_id': 'r', 'capture_date': 'day', 'signal_index': i,
                           'view': 'test', 'relative_path': 'fixture.npz'} for i in range(3)]}
        streams = [('unknown_fixed_claim', f'start_{offset}', 'unknown',
                    [(batch, i, (i + offset) % 3, True, False) for i in range(3)])
                   for offset in range(3)]
        with temporary_output() as folder:
            result = evaluate_streams(streams, reference(), Policy(), {'attempt_interval_seconds': 1.},
                                      {'a': 0, 'b': 1, 'c': 2}, .5, Path(folder))
            summary = result['scenarios'][0]
            self.assertEqual(summary['listed_streams'], 3)
            self.assertEqual(summary['baseline_accepts'], 3)
            self.assertEqual(summary['accepts_if_list_blocks'], 2)
            self.assertEqual(summary['streams_with_success_if_list_blocks'], 2)


if __name__ == '__main__':
    unittest.main()
