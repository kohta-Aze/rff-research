"""Stateful review candidates; no transmitter truth, dates, or locations as inputs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib

import numpy as np


def unit_features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Features must be a nonempty finite matrix.')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Zero features cannot be compared using cosine distance.')
    return values / norms


def waveform_digest(iq):
    """Exact normalized sample equality, not an over-the-air replay detector."""
    values = np.asarray(iq, dtype='<f4')
    if values.shape != (2, 256) or not np.isfinite(values).all():
        raise ValueError('Expected a finite normalized [2, 256] waveform.')
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class FeatureReference:
    def fit(self, training, labels, validation, validation_labels, validation_groups, quantile):
        training, validation = unit_features(training), unit_features(validation)
        labels, validation_labels = np.asarray(labels), np.asarray(validation_labels)
        if (labels.dtype.kind not in 'iu' or validation_labels.dtype.kind not in 'iu'
                or len(labels) != len(training) or len(validation_labels) != len(validation)
                or np.any(labels < 0) or np.any(validation_labels < 0)
                or len(validation_groups) != len(validation) or not 0 < quantile < 1
                or training.shape[1] != validation.shape[1]):
            raise ValueError('Only matched known training/validation data may fit references.')
        classes = np.unique(labels)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError('Known classes must be contiguous from zero.')
        if not np.array_equal(np.unique(validation_labels), classes):
            raise ValueError('Validation must cover exactly the training classes.')
        self.centers = unit_features(np.stack([training[labels == k].mean(0) for k in classes]))
        self.thresholds = np.asarray([
            np.quantile(1 - validation[validation_labels == k] @ self.centers[k],
                        quantile, method='higher') for k in classes
        ]).clip(0, 2)
        jumps = []
        groups = {}
        for i, (label, group) in enumerate(zip(validation_labels, validation_groups)):
            groups.setdefault((int(label), str(group)), []).append(i)
        for indices in groups.values():
            values = validation[indices]
            if len(values) > 1:
                jumps.extend((1 - np.sum(values[1:] * values[:-1], axis=1)).clip(0, 2))
        if not jumps:
            raise ValueError('Need repeated known validation samples to calibrate changes.')
        self.jump_threshold = float(np.quantile(jumps, quantile, method='higher'))
        self.quantile = float(quantile)
        return self

    def compare(self, feature, claim_id, previous):
        feature = unit_features(np.asarray(feature)[None])[0]
        if feature.shape != self.centers.shape[1:]:
            raise ValueError('Feature dimensions differ from the frozen reference.')
        if type(claim_id) is not int or not 0 <= claim_id < len(self.centers):
            raise ValueError('Claim must be a registered class ID.')
        distance = float(np.clip(1 - feature @ self.centers[claim_id], 0, 2))
        jump = None if previous is None else float(np.clip(1 - feature @ previous, 0, 2))
        return feature, distance, jump


@dataclass(frozen=True)
class Policy:
    window_seconds: float = 60.
    distinct_claim_limit: int = 3
    feature_window: int = 10
    feature_hit_limit: int = 3
    duplicate_limit: int = 3
    watch_seconds: float = 300.

    def __post_init__(self):
        for value in [self.window_seconds, self.watch_seconds]:
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError('Time windows must be positive and finite.')
        for value in [self.distinct_claim_limit, self.feature_window,
                      self.feature_hit_limit, self.duplicate_limit]:
            if type(value) is not int or value < 2:
                raise ValueError('History limits must be integers >= 2.')
        if self.feature_hit_limit > self.feature_window:
            raise ValueError('Feature hit count exceeds its window.')


@dataclass(frozen=True)
class Attempt:
    observation_id: str
    timestamp: float
    claim_id: int
    feature: np.ndarray
    waveform_hash: str


@dataclass
class Track:
    history: deque = field(default_factory=deque)
    feature_hits: deque = field(default_factory=deque)
    previous: np.ndarray | None = None
    last_seen: float = -np.inf
    watch_until: float = -np.inf
    watch_reasons: set = field(default_factory=set)


class AttemptMonitor:
    """An observation ID must be provided by the receiver, not the claimed ID.

    Expiry and processing time use receiver-side timestamps. A changed tracking
    token breaks continuity; this class does not solve RF-based source linking.
    """
    def __init__(self, reference: FeatureReference, policy: Policy):
        self.reference, self.policy = reference, policy
        self.tracks = {}
        self.last_timestamp = -np.inf

    def observe(self, attempt: Attempt):
        if (not isinstance(attempt.observation_id, str) or not attempt.observation_id
                or isinstance(attempt.timestamp, bool) or not np.isfinite(attempt.timestamp)
                or not isinstance(attempt.waveform_hash, str) or len(attempt.waveform_hash) != 64
                or any(c not in '0123456789abcdef' for c in attempt.waveform_hash)):
            raise ValueError('Need a receiver observation ID, finite time and SHA-256 digest.')
        if attempt.timestamp < self.last_timestamp:
            raise ValueError('Attempts must arrive in receiver-time order.')
        track = self.tracks.get(attempt.observation_id, Track())
        reset = attempt.timestamp - track.last_seen > self.policy.window_seconds
        previous = None if reset else track.previous
        feature, distance, jump = self.reference.compare(attempt.feature, attempt.claim_id, previous)
        # Validate everything before updating state. Unknown truth is never passed here.
        self.last_timestamp = attempt.timestamp
        for key in list(self.tracks):
            old = self.tracks[key]
            if (attempt.timestamp - old.last_seen > self.policy.window_seconds
                    and attempt.timestamp >= old.watch_until):
                del self.tracks[key]
        self.tracks[attempt.observation_id] = track
        if reset:
            track.history.clear()
            track.feature_hits.clear()
        if attempt.timestamp >= track.watch_until:
            track.watch_reasons.clear()
        while track.history and attempt.timestamp - track.history[0][0] > self.policy.window_seconds:
            track.history.popleft()
        while track.feature_hits and attempt.timestamp - track.feature_hits[0][0] > self.policy.window_seconds:
            track.feature_hits.popleft()
        track.history.append((attempt.timestamp, attempt.claim_id, attempt.waveform_hash))
        track.feature_hits.append((attempt.timestamp, distance > self.reference.thresholds[attempt.claim_id],
                                   jump is not None and jump > self.reference.jump_threshold))
        while len(track.feature_hits) > self.policy.feature_window:
            track.feature_hits.popleft()
        distinct = len({entry[1] for entry in track.history})
        duplicates = sum(entry[2] == attempt.waveform_hash for entry in track.history)
        mismatch_hits = sum(entry[1] for entry in track.feature_hits)
        change_hits = sum(entry[2] for entry in track.feature_hits)
        reasons = []
        if distinct >= self.policy.distinct_claim_limit:
            reasons.append('id_sweep')
        if mismatch_hits >= self.policy.feature_hit_limit:
            reasons.append('reference_mismatch')
        if change_hits >= self.policy.feature_hit_limit:
            reasons.append('feature_changes')
        if duplicates >= self.policy.duplicate_limit:
            reasons.append('exact_waveform_reuse')
        already_listed = attempt.timestamp < track.watch_until
        if reasons:
            track.watch_until = max(track.watch_until, attempt.timestamp + self.policy.watch_seconds)
            track.watch_reasons.update(reasons)
        track.previous, track.last_seen = feature, attempt.timestamp
        return {
            'on_watchlist': bool(attempt.timestamp < track.watch_until),
            'newly_listed': bool(reasons and not already_listed),
            'trigger_reasons': '|'.join(reasons), 'active_reasons': '|'.join(sorted(track.watch_reasons)),
            'distinct_claims': distinct, 'exact_duplicates': duplicates,
            'reference_distance': distance, 'reference_threshold': float(self.reference.thresholds[attempt.claim_id]),
            'feature_jump': jump, 'jump_threshold': self.reference.jump_threshold,
            'reference_hits': mismatch_hits, 'change_hits': change_hits,
            'watch_until': float(track.watch_until) if track.watch_reasons else None,
        }
