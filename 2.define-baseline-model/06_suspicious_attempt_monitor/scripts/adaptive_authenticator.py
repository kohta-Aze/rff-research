"""Causal, receiver-track-based controls over a frozen authentication score."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class AuthObservation:
    observation_id: str
    timestamp: float
    claim_id: int
    predicted_id: int
    score: float
    reference_distance: float
    reference_threshold: float
    feature_jump: float | None
    jump_threshold: float


@dataclass
class State:
    last_seen: float
    previous_claim: int | None = None
    risk: float = 0.
    events: deque = field(default_factory=deque)
    blocked_until: float = -math.inf


class TrackControl:
    def __init__(self, threshold):
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Need a frozen score threshold strictly between zero and one.')
        self.threshold = float(threshold)
        self.tracks = {}
        self.last_timestamp = -math.inf

    def validate(self, observation):
        o = observation
        if (not isinstance(o.observation_id, str) or not o.observation_id
                or isinstance(o.timestamp, bool) or not math.isfinite(o.timestamp)
                or o.timestamp < self.last_timestamp
                or type(o.claim_id) is not int or o.claim_id < 0
                or type(o.predicted_id) is not int or o.predicted_id < 0
                # Frozen OpenMax encodes an unknown-class winner as -P(unknown).
                or not math.isfinite(o.score) or not -1 <= o.score <= 1
                or not all(math.isfinite(x) and 0 <= x <= 2 for x in
                           [o.reference_distance, o.reference_threshold, o.jump_threshold])
                or (o.feature_jump is not None and
                    (not math.isfinite(o.feature_jump) or not 0 <= o.feature_jump <= 2))):
            raise ValueError('Invalid observation; ground-truth labels are not inputs.')

    def baseline_accepts(self, observation):
        return observation.score >= self.threshold and observation.predicted_id == observation.claim_id


class AdaptiveAuthenticator(TrackControl):
    """Use past evidence and the current claimed-ID switch before deciding.

    Current score/feature evidence updates the NEXT decision. A rejection caused
    only by our stricter threshold is not fed back as a new baseline failure.
    Own accepts do not erase risk: they are not independent proof of identity.
    """
    def __init__(self, threshold, strength, evidence=True, half_life_seconds=60.,
                 switch_weight=1., feature_weight=.5, max_risk=64.):
        super().__init__(threshold)
        if (type(evidence) is not bool
                or not all(math.isfinite(x) and x >= 0 for x in [strength, switch_weight, feature_weight])
                or not math.isfinite(half_life_seconds) or half_life_seconds <= 0
                or not math.isfinite(max_risk) or max_risk <= 0):
            raise ValueError('Invalid adaptive policy.')
        self.strength, self.evidence = float(strength), evidence
        self.half_life_seconds = float(half_life_seconds)
        self.switch_weight, self.feature_weight = float(switch_weight), float(feature_weight)
        self.max_risk = float(max_risk)

    @staticmethod
    def excess(value, reference):
        return min(3., max(0., value / max(reference, 1e-12) - 1.))

    def step(self, observation):
        self.validate(observation)
        o = observation
        state = self.tracks.get(o.observation_id, State(o.timestamp))
        risk = state.risk * 2. ** (-(o.timestamp - state.last_seen) / self.half_life_seconds)
        switched = state.previous_claim is not None and state.previous_claim != o.claim_id
        if self.evidence and switched:
            risk += self.switch_weight
        risk = min(self.max_risk, risk)
        threshold = max(self.threshold, min(math.nextafter(1., 0.),
            1. - (1. - self.threshold) * math.exp(-self.strength * risk)))
        baseline = self.baseline_accepts(o)
        accepted = baseline and o.score >= threshold
        failure_evidence = float(not baseline)
        feature_evidence = 0.
        if self.evidence:
            # Score deficit is a bounded severity measure, not a probability.
            if not baseline:
                failure_evidence += min(3., max(0.,
                    (self.threshold - o.score) / (1. - self.threshold)))
                failure_evidence += float(o.predicted_id != o.claim_id)
            feature_evidence = self.feature_weight * self.excess(o.reference_distance, o.reference_threshold)
            if o.feature_jump is not None:
                feature_evidence += self.feature_weight * self.excess(o.feature_jump, o.jump_threshold)
        risk_after = min(self.max_risk, risk + failure_evidence + feature_evidence)
        state.risk, state.previous_claim, state.last_seen = risk_after, o.claim_id, o.timestamp
        self.tracks[o.observation_id] = state
        self.last_timestamp = o.timestamp
        return {'accepted': bool(accepted), 'effective_threshold': threshold,
                'risk_before': risk, 'risk_after': risk_after, 'claim_switched': switched,
                'failure_evidence': failure_evidence, 'feature_evidence': feature_evidence,
                'blocked': False, 'count_in_window': 0}


class CountLimitAuthenticator(TrackControl):
    """A simple control: count total attempts or baseline failures per source."""
    def __init__(self, threshold, limit, kind='failures', window_seconds=60., cooldown_seconds=300.):
        super().__init__(threshold)
        if (type(limit) is not int or limit < 1 or kind not in ['attempts', 'failures']
                or not all(math.isfinite(x) and x > 0 for x in [window_seconds, cooldown_seconds])):
            raise ValueError('Invalid count-limit policy.')
        self.limit, self.kind = limit, kind
        self.window_seconds, self.cooldown_seconds = window_seconds, cooldown_seconds

    def step(self, observation):
        self.validate(observation)
        o = observation
        state = self.tracks.get(o.observation_id, State(o.timestamp))
        baseline = self.baseline_accepts(o)
        while state.events and o.timestamp - state.events[0] > self.window_seconds:
            state.events.popleft()
        if self.kind == 'attempts' or not baseline:
            state.events.append(o.timestamp)
        reached = len(state.events) > self.limit if self.kind == 'attempts' else len(state.events) >= self.limit
        if reached:
            state.blocked_until = max(state.blocked_until, o.timestamp + self.cooldown_seconds)
        blocked = o.timestamp < state.blocked_until
        state.last_seen = o.timestamp
        self.tracks[o.observation_id] = state
        self.last_timestamp = o.timestamp
        return {'accepted': bool(baseline and not blocked), 'effective_threshold': self.threshold,
                'risk_before': 0., 'risk_after': 0., 'claim_switched': False,
                'failure_evidence': float(not baseline), 'feature_evidence': 0.,
                'blocked': blocked, 'count_in_window': len(state.events)}
