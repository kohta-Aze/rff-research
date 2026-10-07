"""Label-blind per-window processing; no Tx, Rx or date inputs."""
import numpy as np


def validate_iq(iq):
    value = np.asarray(iq)
    if value.ndim != 3 or value.shape[1:] != (256, 2) or not len(value):
        raise ValueError('Expected nonempty [N, 256, 2] I/Q.')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite I/Q.')
    work = value.astype(np.float64)
    power = np.mean(np.sum(work * work, axis=-1), axis=-1)
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise ValueError('Invalid I/Q power.')
    return work, power


def residual_emphasis(iq, strength):
    work, _ = validate_iq(iq)
    if not np.isscalar(strength) or not np.isfinite(strength) or not -0.5 <= strength <= 0.5:
        raise ValueError('Strength must be finite in [-0.5, 0.5].')
    if strength == 0:
        return np.array(iq, dtype=np.float32, copy=True)
    padded = np.pad(work, ((0, 0), (1, 1), (0, 0)), mode='reflect')
    smooth = .25 * padded[:, :-2] + .5 * padded[:, 1:-1] + .25 * padded[:, 2:]
    changed = work + strength * (work - smooth)
    # Match the baseline's float64-power / float32-output normalization.
    rms = np.sqrt(np.mean(np.sum(changed * changed, axis=-1), axis=-1))
    return np.asarray(changed / rms[:, None, None], dtype=np.float32)


def relative_l2(original, changed):
    first, power = validate_iq(original)
    second, _ = validate_iq(changed)
    if first.shape != second.shape:
        raise ValueError('Paired I/Q shapes differ.')
    return np.sqrt(np.mean(np.sum((second - first) ** 2, axis=-1), axis=-1) / power)


def select_candidate(candidates, config):
    """Use validation results only; grid order breaks ties, including identity."""
    baseline = candidates[0]['metrics']
    if candidates[0]['strength'] != 0:
        raise ValueError('The first candidate must be the unchanged baseline.')
    eligible = [row for row in candidates if
                row['relative_l2_max'] <= config['max_relative_l2_per_signal'] and
                row['metrics']['unknown_false_accept_rate'] <= baseline['unknown_false_accept_rate'] + config['max_unknown_far_increase'] and
                row['metrics']['known_closed_set_accuracy'] >= baseline['known_closed_set_accuracy'] - config['max_known_accuracy_decrease']]
    if not eligible:
        raise ValueError('No eligible validation candidate, including identity.')
    return max(eligible, key=lambda row: row['metrics']['oscr_auc'])
