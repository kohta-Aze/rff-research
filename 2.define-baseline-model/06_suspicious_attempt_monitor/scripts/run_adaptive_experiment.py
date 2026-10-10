"""Compare fixed limits, the existing watchlist, and causal adaptive thresholds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from adaptive_authenticator import AuthObservation, AdaptiveAuthenticator, CountLimitAuthenticator
from attempt_monitor import Attempt, AttemptMonitor, FeatureReference, Policy, waveform_digest
from run_monitor_experiment import (HERE, BASE, JST, now, sha, read_json, save_json, save_csv,
    source_ready, restore_calibration, load_view, groups, scenario_streams, alter_waveforms, verify_saved)


def prepare_streams(streams, reference, monitor_policy, monitor_config):
    """Truth constructs/audits histories, but is excluded from policy observations."""
    prepared = []
    for number, (scenario, key, category, events) in enumerate(streams):
        monitor = AttemptMonitor(reference, monitor_policy)
        stream_id = f'stream_{number:05d}'
        entries = []
        for ordinal, (batch, index, claim, evaluated, reset) in enumerate(events, start=1):
            token = f'{stream_id}/observation_{ordinal if reset else 0}'
            timestamp = ordinal * monitor_config['attempt_interval_seconds']
            diagnostic = monitor.observe(Attempt(token, timestamp, int(claim), batch['features'][index],
                                                 waveform_digest(batch['iq'][index])))
            observation = AuthObservation(token, timestamp, int(claim), int(batch['predictions'][index]),
                float(batch['scores'][index]), diagnostic['reference_distance'], diagnostic['reference_threshold'],
                diagnostic['feature_jump'], diagnostic['jump_threshold'])
            entries.append({'observation': observation, 'ordinal': ordinal, 'evaluated': evaluated,
                            'hard_watchlist': diagnostic['on_watchlist'], 'audit': batch['rows'][index]})
        prepared.append({'scenario': scenario, 'stream_id': stream_id, 'key': key,
                         'category': category, 'entries': entries})
    return prepared


def controller(spec, threshold, config):
    if spec['kind'] == 'adaptive':
        return AdaptiveAuthenticator(threshold, spec['strength'], spec['evidence'],
            **{key: config[key] for key in ['half_life_seconds', 'switch_weight', 'feature_weight', 'max_risk']})
    if spec['kind'] in ['attempts', 'failures']:
        return CountLimitAuthenticator(threshold, spec['limit'], kind=spec['kind'])
    if spec['kind'] not in ['frozen', 'watchlist']:
        raise ValueError('Unknown control.')
    return None


def evaluate(prepared, spec, threshold, config, writer=None):
    summaries, stream_rows, pairs, subgroups = {}, [], {}, {}
    for stream in prepared:
        control = controller(spec, threshold, config)
        counts = {'attempts': 0, 'baseline_accepts': 0, 'accepts': 0, 'extra_rejects': 0,
                  'initial_attempts': 0, 'initial_accepts': 0, 'blocked_attempts': 0}
        for entry in stream['entries']:
            o = entry['observation']
            baseline = o.score >= threshold and o.predicted_id == o.claim_id
            if control is None:
                blocked = spec['kind'] == 'watchlist' and entry['hard_watchlist']
                decision = {'accepted': baseline and not blocked, 'effective_threshold': threshold,
                            'risk_before': 0., 'risk_after': 0., 'claim_switched': False,
                            'failure_evidence': 0., 'feature_evidence': 0.,
                            'blocked': blocked, 'count_in_window': 0}
            else:
                decision = control.step(o)
            accepted = decision['accepted']
            if accepted and not baseline:
                raise AssertionError('A control must not loosen the frozen authentication decision.')
            if not entry['evaluated']:
                continue
            counts['attempts'] += 1
            counts['baseline_accepts'] += int(baseline)
            counts['accepts'] += int(accepted)
            counts['extra_rejects'] += int(baseline and not accepted)
            counts['blocked_attempts'] += int(decision['blocked'])
            if entry['ordinal'] == 1:
                counts['initial_attempts'] += 1
                counts['initial_accepts'] += int(accepted)
            audit = entry['audit']
            if stream['scenario'] == 'unknown_fixed_claim':
                for table, key in [(pairs, (audit['tx_id'], o.claim_id)),
                                   (subgroups, (audit['tx_id'], o.claim_id, audit['rx_id']))]:
                    group = table.setdefault(key, {'attempts': 0, 'baseline_accepts': 0, 'accepts': 0})
                    group['attempts'] += 1
                    group['baseline_accepts'] += int(baseline)
                    group['accepts'] += int(accepted)
            if writer is not None:
                writer.writerow({'policy': spec['name'], 'scenario': stream['scenario'],
                    'stream_id': stream['stream_id'], 'observation_id': o.observation_id,
                    'attempt': entry['ordinal'], 'synthetic_time_seconds': o.timestamp,
                    'claimed_id': o.claim_id, 'predicted_id': o.predicted_id, 'openmax_score': o.score,
                    'baseline_accepted': baseline, **decision, 'hard_watchlist': entry['hard_watchlist'],
                    'reference_distance': o.reference_distance, 'reference_threshold': o.reference_threshold,
                    'feature_jump': o.feature_jump, 'jump_threshold': o.jump_threshold,
                    'true_tx_for_audit_only': audit['tx_id'], 'rx_for_audit_only': audit['rx_id'],
                    'capture_date_for_audit_only': audit['capture_date'], 'signal_index': audit['signal_index']})
        row = {'policy': spec['name'], 'scenario': stream['scenario'], 'stream_id': stream['stream_id'],
               'category': stream['category'], **counts, 'any_baseline_success': counts['baseline_accepts'] > 0,
               'any_success': counts['accepts'] > 0, 'affected': counts['extra_rejects'] > 0}
        stream_rows.append(row)
        summary = summaries.setdefault(stream['scenario'], {'policy': spec['name'], 'scenario': stream['scenario'],
            'category': stream['category'], 'streams': 0, **{key: 0 for key in counts},
            'streams_with_baseline_success': 0, 'streams_with_success': 0, 'affected_streams': 0})
        summary['streams'] += 1
        for key in counts:
            summary[key] += counts[key]
        summary['streams_with_baseline_success'] += int(row['any_baseline_success'])
        summary['streams_with_success'] += int(row['any_success'])
        summary['affected_streams'] += int(row['affected'])
    for row in summaries.values():
        row['baseline_accept_rate'] = row['baseline_accepts'] / row['attempts']
        row['accept_rate'] = row['accepts'] / row['attempts']
        row['extra_reject_rate'] = row['extra_rejects'] / row['attempts']
        row['any_success_stream_rate'] = row['streams_with_success'] / row['streams']
        row['affected_stream_rate'] = row['affected_streams'] / row['streams']
    pair_rows = [{'policy': spec['name'], 'unknown_tx': key[0], 'claimed_id': key[1], **value,
                  'accept_rate': value['accepts'] / value['attempts']} for key, value in sorted(pairs.items())]
    subgroup_rows = [{'policy': spec['name'], 'unknown_tx': key[0], 'claimed_id': key[1], 'rx': key[2], **value,
                     'accept_rate': value['accepts'] / value['attempts']} for key, value in sorted(subgroups.items())]
    return {'summaries': list(summaries.values()), 'streams': stream_rows,
            'pairs': pair_rows, 'subgroups': subgroup_rows}


def select_controls(validation, threshold, config):
    if not validation or any(s['scenario'] != 'validation_normal' or s['category'] != 'normal' for s in validation):
        raise ValueError('Control selection accepts known normal validation histories only.')
    budget = config['normal_validation_extra_reject_budget']
    rows, chosen = [], []
    for name, evidence in [('adaptive_failures', False), ('adaptive_evidence', True)]:
        feasible = []
        for strength in config['strength_candidates']:
            spec = {'name': name, 'kind': 'adaptive', 'strength': strength, 'evidence': evidence}
            summary = evaluate(validation, spec, threshold, config)['summaries'][0]
            ok = summary['extra_reject_rate'] <= budget + 1e-12
            rows.append({**spec, **summary, 'feasible': ok})
            if ok:
                feasible.append(spec)
        if not feasible:
            raise ValueError('No feasible adaptive control; include strength zero.')
        chosen.append(max(feasible, key=lambda spec: spec['strength']))
    feasible = []
    for limit in config['failure_limit_candidates']:
        spec = {'name': 'failure_limit_calibrated', 'kind': 'failures', 'limit': limit}
        summary = evaluate(validation, spec, threshold, config)['summaries'][0]
        ok = summary['extra_reject_rate'] <= budget + 1e-12
        rows.append({**spec, **summary, 'feasible': ok})
        if ok:
            feasible.append(spec)
    if not feasible:
        raise ValueError('No feasible failure limit; expand the declared grid before testing.')
    chosen.insert(0, min(feasible, key=lambda spec: spec['limit']))
    specs = [{'name': 'frozen', 'kind': 'frozen'},
             {'name': 'attempt_limit_2', 'kind': 'attempts', 'limit': 2},
             chosen[0], {'name': 'watchlist', 'kind': 'watchlist'}, *chosen[1:]]
    selected_validation = [evaluate(validation, spec, threshold, config)['summaries'][0] for spec in specs]
    return {'frozen_at': now(), 'selection_data': 'reference_day_validation_known_only',
            'uses_unknown_or_later_day_signals': False, 'budget': budget,
            'budget_is_validation_upper_bound_not_exact_test_matching': True,
            'selection_rule': 'max_strength_or_min_failure_limit_within_budget',
            'candidates': rows, 'policies': specs, 'selected_validation': selected_validation}


def verify_previous(result, path):
    if not path.exists():
        return {'available': False}
    previous = read_json(path / 'results.json')
    old = {row['scenario']: row for row in previous['scenarios']}
    for row in result['summaries']:
        prior = old[row['scenario']]
        expected = prior['baseline_accepts'] if row['policy'] == 'frozen' else prior['accepts_if_list_blocks']
        if row['policy'] in ['frozen', 'watchlist'] and (row['attempts'] != prior['attempts'] or row['accepts'] != expected):
            raise ValueError('Original watchlist/frozen comparison was not reproduced.')
    return {'available': True, 'identical_frozen_and_watchlist_counts': True,
            'results_sha256': sha(path / 'results.json')}


def run(args):
    config, monitor_config = read_json(args.config), read_json(args.monitor_config)
    if (config['schema_version'] != 'adaptive_authentication_comparison_v1'
            or config['selection'] != 'strongest_feasible_on_known_validation_only'
            or config['risk_feedback'] != 'frozen_baseline_failure_not_own_stricter_rejection'
            or config['action'] != 'offline_counterfactual_no_live_authentication_change'
            or not 0 <= config['normal_validation_extra_reject_budget'] < 1
            or monitor_config['schema_version'] != 'suspicious_attempt_monitor_v1'
            or args.threads < 1):
        raise ValueError('Unsupported comparison protocol.')
    source = args.source_run.resolve()
    info = source_ready(source)
    split = source / f'fold_{info["fold"]:02d}.json'
    plan = read_json(split)
    if plan['class_map'] != info['class_map'] or plan['tx_roles'] != info['tx_roles']:
        raise ValueError('Source role map differs.')
    sys.path.insert(0, str(source / 'source_snapshot'))
    from cnn_backbone import IQClassifier
    from wisig_dataset import WisigDataset
    from openmax import OpenMax, openmax_scores
    torch.set_num_threads(args.threads)
    torch.manual_seed(monitor_config['seed'])
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(source / 'cnn_model.pt', map_location='cpu', weights_only=True)
    model = IQClassifier(checkpoint['class_count'], **checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    calibration, alpha = restore_calibration(source, OpenMax)
    threshold = read_json(source / 'threshold.json')['threshold']
    monitor_policy = Policy(**monitor_config['policy'])
    output = args.output.resolve() if args.output else HERE / 'output' / datetime.now(JST).strftime('%Y%m%d_%H%M%S_adaptive')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    save_json(output / 'config.json', config)
    save_json(output / 'monitor_config.json', monitor_config)
    frozen_inputs = {str(source / name): sha(source / name) for name in
        ['run.json', 'cnn_model.pt', split.name, 'calibration.json', 'threshold.json']}
    state = {'status': 'calibrating_known_only', 'started_at': now(), 'source_run': str(source),
        'checkpoint_sha256': info['checkpoint_sha256'], 'split_sha256': info['split_sha256'],
        'fold': info['fold'], 'seed': info['seed'], 'baseline_retrained': False,
        'config_sha256': sha(args.config), 'monitor_config_sha256': sha(args.monitor_config),
        'input_hashes': frozen_inputs, 'source_files': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'receipts': {}, 'scope': 'exploratory_previously_explored_wisig_synthetic_histories',
        'real_attack_capture': False, 'actual_action': 'offline_counterfactual',
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'threads': args.threads}}
    save_json(output / 'run.json', state)
    started = time.perf_counter()
    try:
        def load(view):
            batch = load_view(model, WisigDataset, args.prepared_root, split, view, calibration, alpha, openmax_scores)
            print(f'Loaded {view}: {len(batch["labels"])}', flush=True)
            return batch
        training, validation = load('train'), load('validation_known')
        order = [i for indices in groups(validation, monitor_config['seed'], True).values() for i in indices]
        reference = FeatureReference().fit(training['features'], training['labels'], validation['features'][order],
            validation['labels'][order], [validation['rows'][i]['rx_id'] for i in order], monitor_config['validation_quantile'])
        np.savez_compressed(output / 'feature_reference.npz', centers=reference.centers,
                            thresholds=reference.thresholds, jump_threshold=reference.jump_threshold)
        validation_streams = [('validation_normal', str(key), 'normal',
            [(validation, i, plan['class_map'][key[0]], True, False) for i in indices])
            for key, indices in groups(validation, monitor_config['seed'], True).items()]
        prepared_validation = prepare_streams(validation_streams, reference, monitor_policy, monitor_config)
        selection = select_controls(prepared_validation, threshold, config)
        selection.update(reference_sha256=sha(output / 'feature_reference.npz'),
            known_validation_signals=len(validation['labels']), training_signals=len(training['labels']))
        save_json(output / 'selection.json', selection)
        selection_hash = sha(output / 'selection.json')
        state.update(status='choices_frozen_before_test', selection_sha256=selection_hash)
        save_json(output / 'run.json', state)
        print('Frozen policies: ' + str(selection['policies']), flush=True)
        del training, validation, prepared_validation, validation_streams
        data = {}
        for date in plan['protocol']['test_dates']:
            view = 'test_' + date
            data[view] = load(view)
            verify_saved(data[view], source, view, threshold, state['receipts'])
        first = data['test_' + plan['protocol']['reference_date']]
        if any(len(indices) < 2 * monitor_config['context_signals'] for batch in data.values()
               for known in [True, False] for indices in groups(batch, monitor_config['seed'], known).values()):
            raise ValueError('Insufficient held-out scenario samples.')
        altered = alter_waveforms(first, model, calibration, alpha, openmax_scores, monitor_config)
        prepared = prepare_streams(scenario_streams(data, altered, plan, monitor_config),
                                   reference, monitor_policy, monitor_config)
        fields = ['policy', 'scenario', 'stream_id', 'observation_id', 'attempt', 'synthetic_time_seconds',
            'claimed_id', 'predicted_id', 'openmax_score', 'baseline_accepted', 'accepted', 'effective_threshold',
            'risk_before', 'risk_after', 'claim_switched', 'failure_evidence', 'feature_evidence', 'blocked',
            'count_in_window', 'hard_watchlist', 'reference_distance', 'reference_threshold', 'feature_jump',
            'jump_threshold', 'true_tx_for_audit_only', 'rx_for_audit_only', 'capture_date_for_audit_only', 'signal_index']
        results = {'summaries': [], 'streams': [], 'pairs': [], 'subgroups': []}
        with (output / 'attempts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for spec in selection['policies']:
                result = evaluate(prepared, spec, threshold, config, writer)
                for key in results:
                    results[key].extend(result[key])
                print(f'Evaluated {spec["name"]}', flush=True)
        comparison = verify_previous(results, args.previous_run)
        inverse = {value: key for key, value in plan['class_map'].items()}
        for row in results['pairs'] + results['subgroups']:
            row['claimed_tx'] = inverse[row['claimed_id']]
        save_csv(output / 'summary.csv', results['summaries'])
        save_csv(output / 'streams.csv', results['streams'])
        save_csv(output / 'unknown_claim_pairs.csv', results['pairs'])
        save_csv(output / 'unknown_claim_receivers.csv', results['subgroups'])
        compact = {'schema_version': config['schema_version'], 'recorded_on_jst': now()[:10],
            'human_review_status': 'pending', 'source_run': str(source), 'detailed_output': str(output),
            'checkpoint_sha256': info['checkpoint_sha256'], 'split_sha256': info['split_sha256'],
            'config_sha256': state['config_sha256'], 'source_files': state['source_files'],
            'calibration_grid_notes': config.get('calibration_grid_notes'),
            'selection_sha256': selection_hash, 'policies': selection['policies'],
            'validation_budget': selection['budget'], 'validation_results': selection['selected_validation'],
            'summaries': results['summaries'], 'unknown_claim_pairs': results['pairs'],
            'fully_accepted_fixed_claim_receiver_groups': [r for r in results['subgroups'] if r['accepts'] == r['attempts']],
            'previous_monitor_reproduced': comparison, 'limitations': [
                'Frozen CNN/OpenMax; adaptive thresholds cannot undo a first-attempt success.',
                'Policy strength selected on same-day known validation only, before this run reads test signals.',
                'The 1-point budget is not guaranteed on test or later days; actual losses are reported.',
                'Previously explored fold 0 seed 42; not a fresh independent confirmation.',
                'Two test-unknown transmitters; histories and six starting IDs share recorded signals.',
                'Stable receiver source linking is assumed and not implemented; timestamps/order are synthetic.',
                'Synthetic exact-waveform reuse is not over-the-air replay; age/location are unobserved.',
                'All control actions are offline counterfactuals; no live authentication is changed.']}
        save_json(output / 'results.json', compact)
        checks = {**frozen_inputs, str(args.config): state['config_sha256'],
                  str(args.monitor_config): state['monitor_config_sha256'],
                  str(output / 'selection.json'): selection_hash,
                  str(output / 'feature_reference.npz'): selection['reference_sha256']}
        for path, expected in checks.items():
            if sha(path) != expected:
                raise ValueError(f'Frozen input changed: {path}')
        state.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
            baseline_decisions_reproduced=True, previous_monitor_reproduced=comparison,
            output=str(output), evaluated_attempt_rows=sum(r['attempts'] for r in results['summaries']))
        save_json(output / 'run.json', state)
        for row in results['summaries']:
            if row['scenario'] in ['normal_same_day', 'unknown_id_sweep', 'unknown_fixed_claim']:
                print(f'{row["policy"]} / {row["scenario"]}: accept={row["accept_rate"]:.3%}, '
                      f'any-success={row["any_success_stream_rate"]:.3%}, extra-reject={row["extra_reject_rate"]:.3%}', flush=True)
        print(f'Completed: {output}', flush=True)
    except BaseException as exc:
        state.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=BASE / '03_cnn_openmax/output/20261006_primary_fold00_seed42')
    parser.add_argument('--previous-run', type=Path, default=HERE / 'output/20261010_balanced_claim_orders_fold00_seed42')
    parser.add_argument('--prepared-root', type=Path, default=BASE / '00_common/data_preprocessing/output/prepared-wisig')
    parser.add_argument('--monitor-config', type=Path, default=HERE / 'experiments/monitor.json')
    parser.add_argument('--config', type=Path, default=HERE / 'experiments/adaptive.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
