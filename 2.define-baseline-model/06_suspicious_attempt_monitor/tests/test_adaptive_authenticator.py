"""Causality, first-attempt limits, source isolation, and selection constraints."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adaptive_authenticator import AuthObservation, AdaptiveAuthenticator, CountLimitAuthenticator


def obs(t, score=.9, claim=0, predicted=0, token='receiver-track', distance=.1, jump=0.):
    return AuthObservation(token, float(t), claim, predicted, score, distance, .2, jump, .2)


class AdaptiveTests(unittest.TestCase):
    def test_first_attempt_unchanged_even_with_large_current_feature_mismatch(self):
        control = AdaptiveAuthenticator(.5, 8.)
        result = control.step(obs(1, distance=1.5, jump=1.5))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['effective_threshold'], .5)
        self.assertGreater(result['risk_after'], 0.)
        self.assertGreater(control.step(obs(2))['effective_threshold'], .5)

    def test_failure_tightens_next_attempt_and_id_change_keeps_source_history(self):
        control = AdaptiveAuthenticator(.5, 1.)
        self.assertFalse(control.step(obs(1, score=.4))['accepted'])
        result = control.step(obs(2, score=.6, claim=1, predicted=1))
        self.assertFalse(result['accepted'])
        self.assertTrue(result['claim_switched'])
        self.assertGreater(result['effective_threshold'], .6)
        self.assertEqual(set(control.tracks), {'receiver-track'})

    def test_failure_severity_changes_future_threshold(self):
        mild, severe = AdaptiveAuthenticator(.5, .3), AdaptiveAuthenticator(.5, .3)
        mild.step(obs(1, score=.49))
        severe.step(obs(1, score=.1))
        self.assertGreater(severe.step(obs(2))['effective_threshold'], mild.step(obs(2))['effective_threshold'])

    def test_own_stricter_rejection_is_not_new_failure_evidence(self):
        control = AdaptiveAuthenticator(.5, 1., evidence=False)
        control.step(obs(1, score=.4))
        result = control.step(obs(2, score=.6))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['failure_evidence'], 0.)
        self.assertEqual(result['risk_after'], result['risk_before'])

    def test_own_accept_does_not_erase_risk(self):
        control = AdaptiveAuthenticator(.5, .1, evidence=False)
        control.step(obs(1, score=.4))
        result = control.step(obs(2, score=1.))
        self.assertTrue(result['accepted'])
        self.assertGreater(result['risk_after'], 0.)

    def test_risk_decays_over_receiver_time(self):
        control = AdaptiveAuthenticator(.5, 1., evidence=False, half_life_seconds=10.)
        control.step(obs(1, score=.4))
        result = control.step(obs(11))
        self.assertAlmostEqual(result['risk_before'], .5)
        later = control.step(obs(101))
        self.assertLess(later['effective_threshold'], result['effective_threshold'])

    def test_source_isolation_and_changed_tokens_evade_history(self):
        control = AdaptiveAuthenticator(.5, 8.)
        control.step(obs(1, score=.4, distance=1.))
        result = control.step(obs(2, claim=1, predicted=1, token='new-track'))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['effective_threshold'], .5)
        self.assertFalse(result['claim_switched'])

    def test_failure_only_ablation_does_not_use_feature_or_id_switch(self):
        control = AdaptiveAuthenticator(.5, 8., evidence=False)
        control.step(obs(1, score=1., distance=1.))
        result = control.step(obs(2, claim=1, predicted=1, score=.6))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['effective_threshold'], .5)

    def test_very_high_wrong_accepts_can_survive_stricter_threshold(self):
        control = AdaptiveAuthenticator(.5, 8.)
        control.step(obs(1, score=.1))
        self.assertTrue(control.step(obs(2, score=1.))['accepted'])

    def test_invalid_observation_does_not_mutate_state(self):
        control = AdaptiveAuthenticator(.5, 1.)
        control.step(obs(1, score=.4))
        before = copy.deepcopy(control.tracks)
        for invalid in [replace(obs(2), score=float('nan')), obs(0), replace(obs(2), claim_id=-1)]:
            with self.assertRaises(ValueError):
                control.step(invalid)
        self.assertEqual(control.tracks, before)
        self.assertEqual(control.last_timestamp, 1.)

    def test_strength_zero_reproduces_baseline(self):
        control = AdaptiveAuthenticator(.5, 0.)
        events = [obs(1, score=.1), obs(2, claim=1, predicted=1, score=.6), obs(3, score=.9)]
        self.assertEqual([control.step(o)['accepted'] for o in events], [False, True, True])

    def test_negative_openmax_unknown_winner_score_remains_rejected(self):
        for control in [AdaptiveAuthenticator(.5, 1.), CountLimitAuthenticator(.5, 2)]:
            self.assertFalse(control.step(obs(1, score=-.99))['accepted'])
        control = AdaptiveAuthenticator(.5, 1.)
        result = control.step(obs(1, score=-.99))
        self.assertGreater(result['failure_evidence'], 1.)


class LimitTests(unittest.TestCase):
    def test_attempt_cap_two_blocks_third_even_after_successes(self):
        control = CountLimitAuthenticator(.5, 2, kind='attempts')
        self.assertEqual([control.step(obs(t))['accepted'] for t in [1, 2, 3]], [True, True, False])

    def test_one_failure_does_not_permanently_ban_and_cooldown_expires(self):
        control = CountLimitAuthenticator(.5, 2, cooldown_seconds=10.)
        self.assertFalse(control.step(obs(1, score=.4))['blocked'])
        self.assertTrue(control.step(obs(2))['accepted'])
        self.assertTrue(control.step(obs(3, score=.4))['blocked'])
        self.assertFalse(control.step(obs(4))['accepted'])
        self.assertTrue(control.step(obs(65))['accepted'])

    def test_failure_counter_follows_source_across_claim_changes(self):
        control = CountLimitAuthenticator(.5, 2)
        control.step(obs(1, claim=1))
        control.step(obs(2, claim=2))
        self.assertFalse(control.step(obs(3))['accepted'])
        self.assertTrue(control.step(obs(4, token='another-track'))['accepted'])


class SelectionTests(unittest.TestCase):
    def test_only_feasible_parameters_are_selected_without_attack_results(self):
        from run_adaptive_experiment import select_controls
        entries = [{'observation': obs(t, score=.4 if t == 1 else .6),
                    'ordinal': t, 'evaluated': True, 'hard_watchlist': False,
                    'audit': {'tx_id': 'normal', 'rx_id': 'r', 'capture_date': 'reference'}} for t in [1, 2, 3]]
        validation = [{'scenario': 'validation_normal', 'category': 'normal',
                       'stream_id': 'fixture', 'entries': entries}]
        config = {'normal_validation_extra_reject_budget': 0., 'strength_candidates': [0., 8.],
                  'failure_limit_candidates': [1, 2], 'half_life_seconds': 60.,
                  'switch_weight': 1., 'feature_weight': .5, 'max_risk': 64.}
        selection = select_controls(validation, .5, config)
        policies = {p['name']: p for p in selection['policies']}
        self.assertEqual(policies['adaptive_evidence']['strength'], 0.)
        self.assertEqual(policies['adaptive_failures']['strength'], 0.)
        self.assertEqual(policies['failure_limit_calibrated']['limit'], 2)
        for row in selection['selected_validation']:
            if row['policy'] in ['adaptive_evidence', 'adaptive_failures', 'failure_limit_calibrated']:
                self.assertEqual(row['extra_reject_rate'], 0.)
        validation[0]['category'] = 'unknown'
        with self.assertRaises(ValueError):
            select_controls(validation, .5, config)


if __name__ == '__main__':
    unittest.main()
