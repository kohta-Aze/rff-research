"""Compare frozen OpenMax voting with median/IQR set classification."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
JST = timezone(timedelta(hours=9))
from set_methods import MedianIQR, outcome_metrics, partition, vote


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty result table.')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_config(config):
    expected = {
        'schema_version': 'signal_set_iqr_v1', 'primary_set_size': 20, 'set_sizes': [1, 5, 10, 20],
        'features': 'frozen_cnn_128_dimension_relu_embedding',
        'prototype': 'coordinatewise_median_of_known_training_signals_pooled_over_receivers',
        'scale': 'coordinatewise_training_class_IQR_with_floor', 'scale_floor_fraction': 0.1,
        'scale_floor': '0.1_times_pooled_training_dimension_IQR_or_median_positive_pooled_IQR_for_constant_dimension',
        'distance': 'sqrt_mean_squared_class_IQR_standardized_difference',
        'set_representative': 'coordinatewise_median',
        'threshold': 'per_set_size_max_validation_balanced_open_set_accuracy_highest_known_positive_threshold_tie_break',
        'vote': 'unique_plurality_of_frozen_openmax_final_labels_including_unknown_ties_reject',
        'validation': ['validation_known', 'validation_unknown'],
        'grouping': 'same_transmitter_receiver_date_assumed_available_without_transmitter_identity',
        'partition': 'sha256_order_disjoint_no_replacement_no_dropped_packets',
        'inferencer_receives_true_tx_rx_or_date': False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Configuration describes a method this runner does not implement.')
    if type(config.get('group_seed')) is not int:
        raise ValueError('Grouping seed must be an integer.')


def infer_view(model, dataset_class, prepared, split, view, batch_size=128):
    dataset = dataset_class(prepared, split, view, cache_groups=128)
    features, logits, labels, metadata = [], [], [], []
    with torch.inference_mode():
        for iq, target, rows in dataset.iter_batches(batch_size):
            embedding = model.features(torch.from_numpy(iq.transpose(0, 2, 1).copy()))
            features.append(embedding.numpy())
            logits.append(model.classifier(embedding).numpy())
            labels.append(target)
            metadata.extend(rows)
    print(f'{view}: {len(dataset)} signals', flush=True)
    return {'features': np.concatenate(features), 'logits': np.concatenate(logits),
            'labels': np.concatenate(labels), 'rows': metadata}


def combine(first, second):
    return {key: first[key] + second[key] if key == 'rows' else np.concatenate([first[key], second[key]])
            for key in first}


def match_saved(data, source, view, threshold, receipts):
    path = source / f'{view}_outputs.npz'
    csv_path = source / f'{view}_predictions.csv'
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['labels'], data['labels']):
            raise ValueError('Saved label order differs from the actual split.')
        maximum_error = float(np.max(np.abs(saved['logits'] - data['logits'])))
        if maximum_error > 5e-5:
            raise ValueError(f'Frozen model does not reproduce logits: {maximum_error}')
        final = np.where(saved['scores'] >= threshold, saved['predictions'], -1)
    with csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data['rows']):
        raise ValueError('Prediction CSV row count differs from the split.')
    for i, (saved_row, actual) in enumerate(zip(rows, data['rows'])):
        if any(str(actual[key]) != saved_row[key] for key in ['tx_id', 'rx_id', 'capture_date', 'relative_path', 'signal_index']):
            raise ValueError('Prediction CSV identities differ from actual signals.')
        if int(saved_row['open_set_prediction']) != int(final[i]):
            raise ValueError('CSV and NPZ operational predictions disagree.')
    data['packet_final'] = final.astype(np.int64)
    receipts[view] = {'npz_sha256': sha(path), 'csv_sha256': sha(csv_path),
                      'max_recomputed_logit_error': maximum_error, 'signal_count': len(final)}


def grouped(data, size, seed):
    groups = partition(data['rows'], size, seed)
    arrays, labels, metadata = [], [], []
    for group in groups:
        indices = group['indices']
        group_labels = data['labels'][indices]
        if len(set(group_labels.tolist())) != 1:
            raise ValueError('Mixed target labels in a pure-source set.')
        row = data['rows'][indices[0]]
        arrays.append(data['features'][indices])
        labels.append(int(group_labels[0]))
        metadata.append({
            'set_id': group['set_id'], 'tx_id': row['tx_id'], 'rx_id': row['rx_id'],
            'capture_date': row['capture_date'], 'true_label': int(group_labels[0]),
            'signal_indices': '|'.join(str(data['rows'][i]['signal_index']) for i in indices),
        })
    return groups, arrays, np.asarray(labels, dtype=np.int64), metadata


def source_ready(source):
    info = load_json(source / 'run.json')
    if info['status'] != 'completed' or info['method'] != 'CNN + OpenMax':
        raise ValueError('Source must be a completed CNN + OpenMax run.')
    for filename, key in [('cnn_model.pt', 'checkpoint_sha256'), ('fold_00.json', 'split_sha256'),
                          ('threshold.json', 'threshold_sha256'), ('calibration.json', 'calibration_sha256')]:
        if sha(source / filename) != info[key]:
            raise ValueError(f'Source hash mismatch: {filename}')
    names = {'cnn_backbone.py', 'wisig_dataset.py', 'wisig_common.py', 'wisig_coverage.py',
             'wisig_protocol.py', 'open_set_metrics.py'}
    for relative, expected in info['source_files'].items():
        name = Path(relative).name
        if name in names:
            if sha(source / 'source_snapshot' / name) != expected:
                raise ValueError(f'Frozen source mismatch: {name}')
            names.remove(name)
    if names:
        raise ValueError(f'Missing source verification for {names}')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / '実験条件.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    config = load_json(args.config)
    validate_config(config)
    if args.threads < 1:
        raise ValueError('Thread count must be positive.')
    source = ROOT / config['source_run']
    source_info = source_ready(source)
    sys.path.insert(0, str(source / 'source_snapshot'))
    from cnn_backbone import IQClassifier
    from wisig_dataset import WisigDataset
    from open_set_metrics import select_threshold
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve() if args.output else HERE / 'output' / datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(args.config, output / '実験条件.json')
    started = time.perf_counter()
    state = {'status': 'fitting_source_only', 'started_at': now(), 'config': config,
             'config_sha256': sha(args.config), 'source_run': str(source), 'output': str(output),
             'checkpoint_sha256': source_info['checkpoint_sha256'], 'split_sha256': source_info['split_sha256'],
             'openmax_threshold_sha256': source_info['threshold_sha256'],
             'source_run_sha256': sha(source / 'run.json'), 'receipts': {},
             'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                         'platform': platform.platform(), 'threads': args.threads},
             'source_files': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
             'inference_grouping_is_an_assumption': True, 'later_days_previously_explored': True}
    save_json(output / 'run.json', state)
    try:
        checkpoint = torch.load(source / 'cnn_model.pt', map_location='cpu', weights_only=True)
        model = IQClassifier(checkpoint['class_count'], **checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        split = source / 'fold_00.json'
        plan = load_json(split)
        prepared = ROOT / '2.define-baseline-model/00_common/data_preprocessing/output/prepared-wisig'
        original_threshold = load_json(source / 'threshold.json')['threshold']
        training = infer_view(model, WisigDataset, prepared, split, 'train')
        robust = MedianIQR(checkpoint['class_count'], config['scale_floor_fraction']).fit(training['features'], training['labels'])
        pca = PCA(n_components=2, svd_solver='full').fit(training['features'])
        np.savez_compressed(output / 'training_statistics.npz', centers=robust.centers,
                            class_iqr=robust.class_iqr, scales=robust.scales, floor=robust.floor,
                            pca_components=pca.components_, pca_mean=pca.mean_,
                            pca_explained_variance_ratio=pca.explained_variance_ratio_)
        state['training'] = {'signal_count': len(training['labels']), 'class_counts': np.bincount(training['labels']).tolist(),
                             'class_iqr_zero_fraction': float(np.mean(robust.class_iqr == 0)),
                             'scale_floor_used_fraction': float(np.mean(robust.class_iqr < robust.floor)),
                             'statistics_sha256': sha(output / 'training_statistics.npz')}
        validation = combine(infer_view(model, WisigDataset, prepared, split, 'validation_known'),
                             infer_view(model, WisigDataset, prepared, split, 'validation_unknown'))
        match_saved(validation, source, 'validation', original_threshold, state['receipts'])
        selections, validation_metrics = {}, []
        for size in config['set_sizes']:
            groups, inputs, labels, metadata = grouped(validation, size, config['group_seed'])
            _, predictions, scores, _ = robust.infer(inputs)
            selection = select_threshold(labels, predictions, scores)
            selections[str(size)] = {**selection, 'distance_threshold': -selection['threshold'],
                                      'validation_set_count': len(labels),
                                      'known_count': int((labels >= 0).sum()), 'unknown_count': int((labels < 0).sum())}
            final = np.where(scores >= selection['threshold'], predictions, -1)
            validation_metrics.append({'method': 'median_iqr', 'set_size': size,
                                       **outcome_metrics(labels, final)})
            votes = np.asarray([vote(validation['packet_final'][g['indices']], checkpoint['class_count'])[0] for g in groups])
            validation_metrics.append({'method': 'vote_openmax', 'set_size': size,
                                       **outcome_metrics(labels, votes)})
        # Persist every model/threshold choice before any target feature extraction.
        frozen = {'frozen_at': now(), 'config_sha256': state['config_sha256'],
                  'statistics_sha256': state['training']['statistics_sha256'],
                  'selections': selections, 'validation_metrics': validation_metrics,
                  'target_feature_extraction_started': False}
        save_json(output / 'selection.json', frozen)
        selection_hash = sha(output / 'selection.json')
        state.update(status='choices_frozen_before_target', selection_sha256=selection_hash)
        save_json(output / 'run.json', state)
        print('Choices frozen:', {n: s['distance_threshold'] for n, s in selections.items()}, flush=True)
        all_metrics, subgroup_metrics, set_predictions, point_rows = [], [], [], []
        original_metrics = load_json(source / 'metrics.json')
        inverse_map = {v: k for k, v in plan['class_map'].items()}
        for date in config['test_dates']:
            if sha(output / 'selection.json') != selection_hash:
                raise ValueError('Selection changed after test began.')
            view = 'test_' + date
            data = infer_view(model, WisigDataset, prepared, split, view)
            match_saved(data, source, view, original_threshold, state['receipts'])
            baseline = outcome_metrics(data['labels'], data['packet_final'])
            for key in ['known_correct_accept_rate', 'known_false_reject_rate', 'known_misidentification_rate', 'unknown_false_accept_rate', 'balanced_open_set_accuracy']:
                if abs(baseline[key] - original_metrics[view][key]) > 1e-12:
                    raise ValueError(f'Original metrics do not reproduce: {view}/{key}')
            all_metrics.append({'date': date, 'method': 'single_openmax', 'set_size': 1, 'vote_tie_count': 0, **baseline})
            primary_data = [('single_openmax', 1, data['labels'], data['packet_final'], data['rows'])]
            for size in config['set_sizes']:
                groups, inputs, labels, metadata = grouped(data, size, config['group_seed'])
                representatives, predictions, scores, distances = robust.infer(inputs)
                final = np.where(scores >= selections[str(size)]['threshold'], predictions, -1)
                vote_results = [vote(data['packet_final'][g['indices']], checkpoint['class_count']) for g in groups]
                votes = np.asarray([v[0] for v in vote_results], dtype=np.int64)
                tie_count = sum(v[1] for v in vote_results)
                if size == 1 and not np.array_equal(votes, data['packet_final'][[g['indices'][0] for g in groups]]):
                    raise ValueError('One-packet voting must equal the original decisions.')
                for name, finals, ties in [('vote_openmax', votes, tie_count), ('median_iqr', final, 0)]:
                    all_metrics.append({'date': date, 'method': name, 'set_size': size, 'vote_tie_count': ties,
                                        **outcome_metrics(labels, finals)})
                    if size == config['primary_set_size']:
                        primary_data.append((name, size, labels, finals, metadata))
                xy = pca.transform(representatives) if size == config['primary_set_size'] else None
                for i, row in enumerate(metadata):
                    exported = {**row, 'set_size': size, 'vote_prediction': int(votes[i]),
                                'vote_tied': bool(vote_results[i][1]), 'iqr_closed_prediction': int(predictions[i]),
                                'iqr_final_prediction': int(final[i]), 'iqr_distance': float(-scores[i]),
                                'iqr_distance_threshold': selections[str(size)]['distance_threshold']}
                    set_predictions.append(exported)
                    if xy is not None:
                        point_rows.append({**exported, 'pca_1': float(xy[i, 0]), 'pca_2': float(xy[i, 1]),
                                           'iqr_distances': distances[i].tolist()})
            for name, size, labels, finals, metadata in primary_data:
                for field in ['rx_id', 'tx_id']:
                    values = np.asarray([r[field] for r in metadata])
                    for value in sorted(set(values)):
                        mask = values == value
                        subgroup_metrics.append({'date': date, 'method': name, 'set_size': size,
                                                 'group_by': field, 'group': value,
                                                 **outcome_metrics(labels[mask], finals[mask])})
            print(date, [(r['method'], round(r['balanced_open_set_accuracy'], 4)) for r in all_metrics
                         if r['date'] == date and (r['method'] == 'single_openmax' or r['set_size'] == 20)], flush=True)
        if sha(output / 'selection.json') != selection_hash or sha(source / 'cnn_model.pt') != state['checkpoint_sha256']:
            raise ValueError('Frozen choices/model were modified.')
        save_csv(output / 'set_predictions.csv', set_predictions)
        save_csv(output / 'metrics.csv', all_metrics)
        save_csv(output / 'subgroups.csv', subgroup_metrics)
        save_json(output / 'set_points.json', {'classes': inverse_map, 'pca_training_count': len(training['labels']),
                  'pca_variance': pca.explained_variance_ratio_.tolist(),
                  'centers': pca.transform(robust.centers).tolist(), 'points': point_rows})
        result = {'generated_at': now(), 'config': config, 'source_checkpoint_sha256': state['checkpoint_sha256'],
                  'selection_sha256': selection_hash, 'output': str(output), 'training': state['training'],
                  'selections': selections, 'validation_metrics': validation_metrics, 'metrics': all_metrics,
                  'subgroups': subgroup_metrics, 'receipts': state['receipts']}
        save_json(output / 'results.json', result)
        result_dir = HERE / '実験結果'
        result_dir.mkdir(exist_ok=True)
        save_json(result_dir / '信号セットの比較結果.json', result)
        save_csv(result_dir / '結果.csv', all_metrics)
        save_csv(result_dir / '受信機別と送信機別.csv', subgroup_metrics)
        state.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
                     set_prediction_rows=len(set_predictions), selection_unchanged=True, cnn_unchanged=True)
        save_json(output / 'run.json', state)
        print('Completed:', output, flush=True)
    except BaseException as exc:
        state.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
