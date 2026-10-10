"""Offline behavior-monitor experiment on the frozen CNN + OpenMax baseline."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from attempt_monitor import Attempt, AttemptMonitor, FeatureReference, Policy, waveform_digest

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parent
JST = timezone(timedelta(hours=9))


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_ready(source):
    info = read_json(source / 'run.json')
    if info['status'] != 'completed' or info['method'] != 'CNN + OpenMax' or not info['test_executed']:
        raise ValueError('Source must be a completed CNN + OpenMax run with tests.')
    files = {'cnn_model.pt': 'checkpoint_sha256', f'fold_{info["fold"]:02d}.json': 'split_sha256',
             'threshold.json': 'threshold_sha256', 'calibration.json': 'calibration_sha256'}
    for filename, key in files.items():
        if sha(source / filename) != info[key]:
            raise ValueError(f'Frozen source hash mismatch: {filename}')
    needed = {'cnn_backbone.py', 'wisig_dataset.py', 'wisig_common.py',
              'wisig_coverage.py', 'wisig_protocol.py', 'openmax.py', 'open_set_metrics.py'}
    for relative, expected in info['source_files'].items():
        name = Path(relative).name
        if name in needed:
            if sha(source / 'source_snapshot' / name) != expected:
                raise ValueError(f'Frozen source code mismatch: {name}')
            needed.remove(name)
    if needed:
        raise ValueError(f'Missing frozen code: {needed}')
    return info


def restore_calibration(source, calibration_class):
    import libmr
    state = read_json(source / 'calibration.json')
    calibration = calibration_class(state['class_count'], state['tail_size'],
                                    state['distance'], state['euclidean_scale'])
    calibration.means = np.asarray(state['mean_activations'], dtype=np.float64)
    calibration.tails = np.asarray(state['tail_distances'], dtype=np.float64)
    for tail, expected in zip(calibration.tails, state['weibull_params']):
        model = libmr.MR()
        model.fit_high(np.ascontiguousarray(tail), state['tail_size'])
        if not model.is_valid or not np.allclose(model.get_params(), expected, rtol=1e-10, atol=1e-12):
            raise ValueError('Stored Weibull tails do not reproduce the baseline fit.')
        calibration.models.append(model)
    return calibration, state['alpha_rank']


def infer_iq(model, iq):
    features, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(iq), 128):
            values = model.features(torch.from_numpy(np.ascontiguousarray(iq[start:start + 128])))
            features.append(values.numpy())
            logits.append(model.classifier(values).numpy())
    return np.concatenate(features), np.concatenate(logits).astype(np.float64)


def load_view(model, dataset_class, prepared, split, view, calibration, alpha, scoring):
    dataset = dataset_class(prepared, split, view, cache_groups=128)
    batches = list(dataset.iter_batches(512))
    iq = np.concatenate([batch[0] for batch in batches]).transpose(0, 2, 1).copy()
    features, logits = infer_iq(model, iq)
    probabilities = calibration.probabilities(logits, alpha)
    predictions, scores, _ = scoring(probabilities)
    return {'iq': iq, 'features': features, 'logits': logits, 'scores': scores,
            'predictions': predictions, 'probabilities': probabilities,
            'labels': np.concatenate([batch[1] for batch in batches]),
            'rows': [row for batch in batches for row in batch[2]]}


def verify_saved(data, source, view, threshold, receipts):
    npz_path, csv_path = source / f'{view}_outputs.npz', source / f'{view}_predictions.csv'
    with np.load(npz_path, allow_pickle=False) as saved:
        if not np.array_equal(saved['labels'], data['labels']):
            raise ValueError(f'Saved label order differs: {view}')
        error = float(np.max(np.abs(saved['logits'] - data['logits'])))
        probability_error = float(np.max(np.abs(saved['probabilities'] - data['probabilities'])))
        if error > 5e-5 or probability_error > 5e-6:
            raise ValueError(f'Frozen inference differs: {view}, {error}, {probability_error}')
        saved_final = np.where(saved['scores'] >= threshold, saved['predictions'], -1)
        current_final = np.where(data['scores'] >= threshold, data['predictions'], -1)
        if not np.array_equal(saved_final, current_final):
            raise ValueError(f'Baseline operational decisions differ: {view}')
    with csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data['rows']):
        raise ValueError('Saved prediction length differs.')
    for i, (actual, saved) in enumerate(zip(data['rows'], rows)):
        if any(str(actual[key]) != saved[key] for key in
               ['tx_id', 'rx_id', 'capture_date', 'relative_path', 'signal_index']):
            raise ValueError('Saved prediction metadata order differs.')
        if int(saved['open_set_prediction']) != int(current_final[i]):
            raise ValueError('Saved CSV/NPZ decisions differ.')
    receipts[view] = {'npz_sha256': sha(npz_path), 'csv_sha256': sha(csv_path),
                      'max_logit_error': error, 'max_probability_error': probability_error,
                      'signal_count': len(rows), 'baseline_decisions_identical': True}


def groups(data, seed, known):
    result = {}
    for i, row in enumerate(data['rows']):
        if (data['labels'][i] >= 0) == known:
            result.setdefault((row['tx_id'], row['rx_id']), []).append(i)
    for indices in result.values():
        indices.sort(key=lambda i: hashlib.sha256(
            f'{seed}|{data["rows"][i]["relative_path"]}|{data["rows"][i]["signal_index"]}'.encode()).digest())
    return result


def alter_waveforms(data, model, calibration, alpha, scoring, config):
    # Digital phase perturbations are synthetic probes, not optimized RF attacks.
    angle = config['phase_rotation_radians'] + np.linspace(0, config['phase_ramp_radians'], 256)
    c, s = np.cos(angle), np.sin(angle)
    original = data['iq']
    iq = np.stack([original[:, 0] * c - original[:, 1] * s,
                   original[:, 0] * s + original[:, 1] * c], axis=1).astype(np.float32)
    features, logits = infer_iq(model, iq)
    probabilities = calibration.probabilities(logits, alpha)
    predictions, scores, _ = scoring(probabilities)
    return {**data, 'iq': iq, 'features': features, 'logits': logits,
            'probabilities': probabilities, 'predictions': predictions, 'scores': scores}


def scenario_streams(data, altered, plan, config):
    """Truth is used only to construct/audit synthetic sessions, never by observe()."""
    reference_date = plan['protocol']['reference_date']
    reference = data['test_' + reference_date]
    known = groups(reference, config['seed'], True)
    unknown = groups(reference, config['seed'], False)
    n = config['context_signals']
    claims = sorted(plan['class_map'].values())

    def event(batch, index, claim, evaluated=True, reset_token=False):
        return batch, index, claim, evaluated, reset_token

    for key, indices in known.items():
        claim = plan['class_map'][key[0]]
        yield 'normal_same_day', str(key), 'normal', [event(reference, i, claim) for i in indices]
        yield 'normal_first_seen', str(key), 'normal', [event(reference, i, claim) for i in indices[:n]]
        for date in plan['protocol']['test_dates']:
            if date == reference_date:
                continue
            later = data['test_' + date]
            later_indices = groups(later, config['seed'], True)[key]
            yield 'normal_day_change_' + date, str(key), 'normal', (
                [event(reference, i, claim, False) for i in indices[:n]]
                + [event(later, i, claim) for i in later_indices[:n]])
        yield 'synthetic_signal_change', str(key), 'synthetic_attack', (
            [event(reference, i, claim, False) for i in indices[:n]]
            + [event(altered, i, claim) for i in indices[n:2 * n]])
        # These samples are first-seen old recordings. Their age is not observable.
        yield 'old_recording_first_seen', str(key), 'synthetic_attack', [
            event(reference, i, claim) for i in indices[:n]]
        yield 'exact_recording_reuse', str(key), 'synthetic_attack', [
            event(reference, indices[0], claim) for _ in range(n)]
    for tx in sorted(plan['class_map']):
        receivers = sorted(rx for t, rx in known if t == tx)
        first, second = known[tx, receivers[0]], known[tx, receivers[1]]
        claim = plan['class_map'][tx]
        yield 'normal_receiver_change', tx, 'normal', (
            [event(reference, i, claim, False) for i in first[:n]]
            + [event(reference, i, claim) for i in second[:n]])
    for key, indices in unknown.items():
        for claim in claims:
            yield 'unknown_fixed_claim', f'{key}|{claim}', 'unknown', [
                event(reference, i, claim) for i in indices]
        # Balance every starting ID: an easy-to-imitate ID may be tried first.
        # Each signal is evaluated against every claim across these rotations.
        for offset in range(len(claims)):
            for reset in [False, True]:
                name = 'unknown_id_sweep_new_track' if reset else 'unknown_id_sweep'
                yield name, f'{key}|start_{offset}', 'unknown', [
                    event(reference, i, claims[(j + offset) % len(claims)], reset_token=reset)
                    for j, i in enumerate(indices)]


def evaluate_streams(streams, reference, policy, config, class_map, threshold, output):
    inverse = {value: key for key, value in class_map.items()}
    summaries, tracks, watchlist, pairs = {}, [], [], {}
    path = output / 'attempts.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = None
        for stream_number, (scenario, key, category, events) in enumerate(streams):
            monitor = AttemptMonitor(reference, policy)
            stream_id = f'stream_{stream_number:05d}'
            evaluated_rows = []
            for ordinal, (batch, index, claim, evaluated, reset) in enumerate(events, start=1):
                token = f'{stream_id}/observation_{ordinal if reset else 0}'
                metadata = batch['rows'][index]
                digest = waveform_digest(batch['iq'][index])
                observed = monitor.observe(Attempt(token, ordinal * config['attempt_interval_seconds'],
                    int(claim), batch['features'][index], digest))
                accepted = bool(batch['scores'][index] >= threshold and batch['predictions'][index] == claim)
                row = {'scenario': scenario, 'stream_id': stream_id, 'observation_id': token,
                       'attempt': ordinal, 'synthetic_time_seconds': ordinal * config['attempt_interval_seconds'],
                       'evaluated': evaluated, 'category': category,
                       'claimed_tx': inverse[claim], 'true_tx_for_audit_only': metadata['tx_id'],
                       'rx_for_audit_only': metadata['rx_id'], 'capture_date_for_audit_only': metadata['capture_date'],
                       'signal_index': metadata['signal_index'], 'source_view': metadata['view'],
                       'source_relative_path': metadata['relative_path'], 'waveform_sha256': digest,
                       'openmax_score': float(batch['scores'][index]),
                       'baseline_accepted': accepted, 'accepted_if_watchlist_blocks': accepted and not observed['on_watchlist'],
                       **observed}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if observed['newly_listed']:
                    # Never blacklist the registered claimed identity as the sender.
                    watchlist.append({k: row[k] for k in ['scenario', 'stream_id', 'observation_id', 'attempt',
                        'synthetic_time_seconds', 'claimed_tx', 'trigger_reasons', 'watch_until']})
                if evaluated:
                    evaluated_rows.append(row)
                    if scenario == 'unknown_fixed_claim':
                        pair_key = (metadata['tx_id'], inverse[claim])
                        pair = pairs.setdefault(pair_key, {'attempts': 0, 'baseline_accepts': 0, 'accepts_if_list_blocks': 0})
                        pair['attempts'] += 1
                        pair['baseline_accepts'] += int(accepted)
                        pair['accepts_if_list_blocks'] += int(row['accepted_if_watchlist_blocks'])
            count = len(evaluated_rows)
            if not count:
                raise ValueError(f'Scenario has no evaluated attempts: {scenario}, {key}')
            flagged = [r for r in evaluated_rows if r['on_watchlist']]
            track = {'scenario': scenario, 'stream_id': stream_id, 'category': category, 'evaluated_attempts': count,
                     'baseline_accepts': sum(r['baseline_accepted'] for r in evaluated_rows),
                     'accepts_if_list_blocks': sum(r['accepted_if_watchlist_blocks'] for r in evaluated_rows),
                     'review_attempts': len(flagged), 'listed_during_evaluation': bool(flagged),
                     'first_listed_attempt': flagged[0]['attempt'] if flagged else None,
                     'any_baseline_success': any(r['baseline_accepted'] for r in evaluated_rows),
                     'any_success_if_list_blocks': any(r['accepted_if_watchlist_blocks'] for r in evaluated_rows)}
            tracks.append(track)
            total = summaries.setdefault(scenario, {'scenario': scenario, 'category': category, 'streams': 0,
                'attempts': 0, 'baseline_accepts': 0, 'accepts_if_list_blocks': 0, 'review_attempts': 0,
                'listed_streams': 0, 'streams_with_baseline_success': 0, 'streams_with_success_if_list_blocks': 0})
            total['streams'] += 1
            for a, b in [('attempts', 'evaluated_attempts'), ('baseline_accepts', 'baseline_accepts'),
                         ('accepts_if_list_blocks', 'accepts_if_list_blocks'), ('review_attempts', 'review_attempts'),
                         ('listed_streams', 'listed_during_evaluation'), ('streams_with_baseline_success', 'any_baseline_success'),
                         ('streams_with_success_if_list_blocks', 'any_success_if_list_blocks')]:
                total[a] += int(track[b])
    for total in summaries.values():
        for rate, numerator in [('baseline_accept_rate', 'baseline_accepts'),
                                ('accept_rate_if_list_blocks', 'accepts_if_list_blocks'), ('review_attempt_rate', 'review_attempts')]:
            total[rate] = total[numerator] / total['attempts']
        total['listed_stream_rate'] = total['listed_streams'] / total['streams']
        total['any_success_stream_rate_if_list_blocks'] = total['streams_with_success_if_list_blocks'] / total['streams']
    pair_rows = [{'unknown_tx': tx, 'claimed_tx': claim, **values,
                  'baseline_accept_rate': values['baseline_accepts'] / values['attempts'],
                  'accept_rate_if_list_blocks': values['accepts_if_list_blocks'] / values['attempts']}
                 for (tx, claim), values in sorted(pairs.items())]
    save_csv(output / 'summary.csv', list(summaries.values()))
    save_csv(output / 'streams.csv', tracks)
    if watchlist:
        save_csv(output / 'watchlist.csv', watchlist)
    else:
        (output / 'watchlist.csv').write_text('scenario,stream_id,observation_id,attempt,synthetic_time_seconds,claimed_tx,trigger_reasons,watch_until\n', encoding='utf-8-sig')
    save_csv(output / 'unknown_claim_pairs.csv', pair_rows)
    return {'scenarios': list(summaries.values()), 'unknown_claim_pairs': pair_rows,
            'watchlist_entries': len(watchlist), 'baseline_retrained': False,
            'actual_action': 'review_list_only', 'blocking_is_offline_counterfactual': True,
            'normal_false_alerts_must_be_considered': True,
            'recording_age_and_physical_location_not_observed': True}


def run(args):
    config = read_json(args.config)
    if (config['schema_version'] != 'suspicious_attempt_monitor_v1'
            or config['tracking'] != 'assumed_receiver_assigned_stable_observation_id'
            or config['ordering'] != 'synthetic_sha256_order_not_capture_time'
            or config['action'] != 'review_list_only_with_offline_blocking_comparison'):
        raise ValueError('Unsupported monitor protocol.')
    policy = Policy(**config['policy'])
    if (type(config['seed']) is not int or type(config['context_signals']) is not int
            or config['context_signals'] < 2 or not 0 < config['validation_quantile'] < 1
            or not np.isfinite(config['attempt_interval_seconds']) or config['attempt_interval_seconds'] <= 0
            or not np.isfinite(config['phase_rotation_radians']) or not np.isfinite(config['phase_ramp_radians'])
            or args.threads < 1):
        raise ValueError('Invalid experiment configuration.')
    source = args.source_run.resolve()
    info = source_ready(source)
    split = source / f'fold_{info["fold"]:02d}.json'
    plan = read_json(split)
    if plan['class_map'] != info['class_map'] or plan['tx_roles'] != info['tx_roles']:
        raise ValueError('Source roles/class map differ from the frozen split.')
    if policy.distinct_claim_limit > len(plan['class_map']):
        raise ValueError('ID sweep limit exceeds the number of registered IDs.')
    sys.path.insert(0, str(source / 'source_snapshot'))
    from cnn_backbone import IQClassifier
    from wisig_dataset import WisigDataset
    from openmax import OpenMax, openmax_scores
    torch.set_num_threads(args.threads)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(source / 'cnn_model.pt', map_location='cpu', weights_only=True)
    model = IQClassifier(checkpoint['class_count'], **checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    calibration, alpha = restore_calibration(source, OpenMax)
    threshold = read_json(source / 'threshold.json')['threshold']
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Expected the positive frozen OpenMax threshold.')
    output = args.output.resolve() if args.output else HERE / 'output' / datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(path, snapshot / path.name)
    save_json(output / 'config.json', config)
    frozen_inputs = {str(source / f): sha(source / f) for f in
                     ['run.json', 'cnn_model.pt', split.name, 'calibration.json', 'threshold.json', 'metrics.json']}
    state = {'status': 'fitting_known_reference', 'started_at': now(), 'source_run': str(source),
             'config': config, 'config_sha256': sha(args.config), 'input_hashes': frozen_inputs,
             'checkpoint_sha256': info['checkpoint_sha256'], 'split_sha256': info['split_sha256'],
             'seed': info['seed'], 'scenario_seed': config['seed'], 'fold': info['fold'], 'receipts': {},
             'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'threads': args.threads},
             'source_files': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
             'scope': 'synthetic_order_and_assumed_source_linking_on_previously_explored_wisig',
             'real_attack_capture': False, 'age_location_or_tx_truth_given_to_monitor': False}
    save_json(output / 'run.json', state)
    started = time.perf_counter()
    try:
        def load(view):
            result = load_view(model, WisigDataset, args.prepared_root, split, view, calibration, alpha, openmax_scores)
            print(f'Loaded {view}: {len(result["labels"])}', flush=True)
            return result
        training, validation = load('train'), load('validation_known')
        order = [i for indices in groups(validation, config['seed'], True).values() for i in indices]
        reference = FeatureReference().fit(training['features'], training['labels'], validation['features'][order],
            validation['labels'][order], [validation['rows'][i]['rx_id'] for i in order], config['validation_quantile'])
        np.savez_compressed(output / 'feature_reference.npz', centers=reference.centers,
                            thresholds=reference.thresholds, jump_threshold=reference.jump_threshold)
        save_json(output / 'selection.json', {'frozen_at': now(), 'quantile': config['validation_quantile'],
            'training_count': len(training['labels']), 'validation_known_count': len(validation['labels']),
            'reference_thresholds': reference.thresholds.tolist(), 'jump_threshold': reference.jump_threshold,
            'fit_uses_test_or_unknown_signals': False, 'policy': config['policy'],
            'reference_sha256': sha(output / 'feature_reference.npz')})
        selection_hash = sha(output / 'selection.json')
        reference_hash = sha(output / 'feature_reference.npz')
        del training, validation
        state.update(status='choices_frozen_before_test', selection_sha256=selection_hash, reference_sha256=reference_hash)
        save_json(output / 'run.json', state)
        data = {}
        for date in plan['protocol']['test_dates']:
            view = 'test_' + date
            data[view] = load(view)
            verify_saved(data[view], source, view, threshold, state['receipts'])
        first = data['test_' + plan['protocol']['reference_date']]
        if any(len(indices) < 2 * config['context_signals'] for batch in data.values()
               for known in [True, False] for indices in groups(batch, config['seed'], known).values()):
            raise ValueError('Insufficient disjoint scenario samples for the requested context.')
        altered = alter_waveforms(first, model, calibration, alpha, openmax_scores, config)
        result = evaluate_streams(scenario_streams(data, altered, plan, config), reference, policy,
                                  config, plan['class_map'], threshold, output)
        save_json(output / 'results.json', result)
        for path, expected in {**frozen_inputs, str(output / 'selection.json'): selection_hash,
                               str(output / 'feature_reference.npz'): reference_hash,
                               str(args.config): state['config_sha256']}.items():
            if sha(path) != expected:
                raise ValueError(f'Frozen input/selection changed: {path}')
        state.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
                     baseline_decisions_reproduced=True, output=str(output))
        save_json(output / 'run.json', state)
        for row in result['scenarios']:
            print(f'{row["scenario"]}: listed streams={row["listed_stream_rate"]:.1%}; '
                  f'accept={row["baseline_accept_rate"]:.1%} -> {row["accept_rate_if_list_blocks"]:.1%}', flush=True)
        print(f'Completed: {output}', flush=True)
    except BaseException as exc:
        state.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=BASE / '03_cnn_openmax/output/20261006_primary_fold00_seed42')
    parser.add_argument('--prepared-root', type=Path, default=BASE / '00_common/data_preprocessing/output/prepared-wisig')
    parser.add_argument('--config', type=Path, default=HERE / 'experiments/monitor.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
