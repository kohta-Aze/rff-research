"""Evaluate train-only cosine prototypes using the frozen CNN from an MSP run."""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MODEL_ROOT.parent
MODELING = BASELINE_ROOT / '00_common/modeling'
MSP_SCRIPTS = BASELINE_ROOT / '01_cnn_msp/scripts'
sys.path.insert(0, str(MSP_SCRIPTS))
sys.path.insert(0, str(MODELING / 'scripts'))
from run_experiment import PREPROCESSING, JST, load_view, save_csv, save_json, now, git_output
from wisig_common import sha256_file
from cnn_backbone import IQClassifier
from cosine_prototype import CosinePrototype
from open_set_metrics import evaluate_open_set, operating_curve, select_threshold


def extract_features(model, iq, batch_size):
    model.eval()
    with torch.inference_mode():
        features = torch.cat([model.features(iq[start:start + batch_size])
                              for start in range(0, len(iq), batch_size)])
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite CNN features.')
    return features


def infer(prototype, features):
    similarities = prototype.similarities(features)
    scores, predictions = similarities.max(dim=1)
    return similarities.numpy(), predictions.numpy(), scores.numpy()


def validate_source(source_root, split_file, config):
    source = json.loads((source_root / 'run.json').read_text(encoding='utf-8'))
    if source['status'] != 'completed' or source['method'] != 'CNN + MSP' or not source['test_executed']:
        raise ValueError('Source must be a completed primary CNN + MSP run.')
    if source['split_sha256'] != sha256_file(split_file):
        raise ValueError('The source CNN uses a different split.')
    if source['checkpoint_sha256'] != sha256_file(source_root / 'best_model.pt'):
        raise ValueError('Source checkpoint hash mismatch.')
    if source['config']['training']['selection'] != 'minimum_validation_known_cross_entropy':
        raise ValueError('Source model must have been selected on known validation loss.')
    for relative, digest in source['source_files'].items():
        if sha256_file(BASELINE_ROOT / relative) != digest:
            raise ValueError(f'Source CNN/loader/evaluation code changed: {relative}')
    expected = {
        'schema_version': 'cosine_prototype_experiment_v1',
        'feature_layer': 'embedding_relu_before_dropout',
        'prototype': {'construction': 'normalize_mean_raw_training_embeddings',
                      'normalization_epsilon': 1e-12, 'arithmetic_dtype': 'float64'},
        'threshold': {'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
                      'tie_break': 'highest_threshold',
                      'acceptance': 'similarity_greater_than_or_equal_to_threshold'},
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError('Unsupported prototype configuration.')
    if config['inference']['prediction'] != 'maximum_cosine_similarity':
        raise ValueError('Unsupported prediction rule.')
    if type(config['inference']['batch_size']) is not int or config['inference']['batch_size'] < 1:
        raise ValueError('Batch size must be a positive integer.')
    return source


def export_view(prototype, features, labels, metadata, threshold, output, view, class_map):
    similarities, predictions, scores = infer(prototype, features)
    metrics = evaluate_open_set(labels, predictions, scores, threshold)
    inverse_map = {value: key for key, value in class_map.items()}
    accepted_labels = np.where(scores >= threshold, predictions, -1)
    save_csv(output / f'{view}_predictions.csv', [{
        **row, 'true_label': int(labels[i]), 'closed_set_prediction': int(predictions[i]),
        'closed_set_predicted_tx': inverse_map[int(predictions[i])],
        'cosine_similarity': float(scores[i]), 'open_set_prediction': int(accepted_labels[i]),
        'open_set_predicted_tx': inverse_map.get(int(accepted_labels[i]), 'unknown'),
    } for i, row in enumerate(metadata)])
    np.savez_compressed(output / f'{view}_features.npz', features=features.numpy(),
                        labels=labels, predictions=predictions, scores=scores, similarities=similarities)
    curve = operating_curve(labels, predictions, scores)
    save_csv(output / f'{view}_oscr.csv', [
        {key: float(value[i]) for key, value in curve.items()}
        for i in range(len(curve['threshold']))
    ])
    groups = []
    for field in ['tx_id', 'rx_id']:
        values = np.asarray([row[field] for row in metadata])
        for value in sorted(set(values)):
            mask = values == value
            groups.append({'view': view, 'group_by': field, 'group': value,
                           **evaluate_open_set(labels[mask], predictions[mask], scores[mask], threshold)})
    save_csv(output / f'{view}_subgroups.csv', groups)
    save_csv(output / f'{view}_confusion.csv', [{
        'true_label': true_label, 'predicted_label': predicted_label,
        'true_tx': inverse_map.get(true_label, 'unknown'),
        'predicted_tx': inverse_map.get(predicted_label, 'unknown'),
        'count': int(((labels == true_label) & (accepted_labels == predicted_label)).sum()),
    } for true_label in [-1, *range(len(class_map))]
      for predicted_label in [-1, *range(len(class_map))]])
    return metrics


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source_root = args.source_run.resolve()
    source_info = json.loads((source_root / 'run.json').read_text(encoding='utf-8'))
    split_file = args.splits_root.resolve() / f'fold_{source_info["fold"]:02d}.json'
    source = validate_source(source_root, split_file, config)
    plan = json.loads(split_file.read_text(encoding='utf-8'))
    if source['class_map'] != plan['class_map'] or source['tx_roles'] != plan['tx_roles']:
        raise ValueError('Source class map or transmitter roles differ.')
    readiness_path = PREPROCESSING / 'output/data_readiness.json'
    readiness = json.loads(readiness_path.read_text(encoding='utf-8'))
    if not readiness['verification_passed'] or readiness['source_sha256'] != source['data_provenance']['source_sha256']:
        raise ValueError('Data verification does not match the source CNN.')
    if args.threads < 1:
        raise ValueError('Threads must be positive.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(source['seed'])
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    mode = 'smoke' if args.validation_only else 'primary'
    output = args.output.resolve() if args.output else MODEL_ROOT / 'output' / f'{stamp}_{mode}_fold{source["fold"]:02d}_seed{source["seed"]}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint_path = source_root / 'best_model.pt'
    info = {
        'status': 'running', 'started_at': now(), 'method': 'CNN + Cosine Prototype',
        'scope': 'validation_only' if args.validation_only else 'single_fold_single_seed',
        'config': config, 'config_file_sha256': sha256_file(args.config),
        'seed': source['seed'], 'fold': source['fold'], 'class_map': plan['class_map'],
        'tx_roles': plan['tx_roles'], 'data_protocol': plan['protocol'],
        'data_provenance': plan['provenance'], 'split_sha256': sha256_file(split_file),
        'source_run': str(source_root), 'source_run_sha256': sha256_file(source_root / 'run.json'),
        'source_metrics_sha256': sha256_file(source_root / 'metrics.json'),
        'checkpoint_sha256': source['checkpoint_sha256'], 'cnn_retrained': False,
        'source_best_epoch': source['training_result']['best_epoch'],
        'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--short'),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'runtime': {'python': sys.version, 'device': 'cpu', 'threads': args.threads,
                    'packages': {name: version(name)
                                 for name in source['runtime']['packages']}},
        'source_files': {},
    }
    code_dir = output / 'source_snapshot'
    code_dir.mkdir()
    for path in [Path(__file__), MSP_SCRIPTS / 'run_experiment.py',
                 *sorted((MODELING / 'scripts').glob('*.py')),
                 *sorted((PREPROCESSING / 'scripts').glob('wisig_*.py'))]:
        info['source_files'][str(path.relative_to(BASELINE_ROOT))] = sha256_file(path)
        shutil.copy2(path, code_dir / path.name)
    shutil.copy2(checkpoint_path, output / 'cnn_model.pt')
    shutil.copy2(source_root / 'run.json', output / 'source_cnn_run.json')
    shutil.copy2(split_file, output / split_file.name)
    save_json(output / 'config.json', config)
    save_json(output / 'run.json', info)
    print(f'Output: {output}; frozen CNN: epoch {info["source_best_epoch"]}', flush=True)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['architecture'] != source['config']['architecture'] or checkpoint['class_count'] != len(plan['class_map']):
            raise ValueError('Checkpoint architecture or class count differs from source run.')
        model = IQClassifier(len(plan['class_map']), **checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'])
        model.requires_grad_(False)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        batch_size = config['inference']['batch_size']
        train_iq, labels, metadata = load_view(args.prepared_root, split_file, 'train')
        features = extract_features(model, train_iq, batch_size)
        prototype = CosinePrototype(len(plan['class_map']), config['prototype']['normalization_epsilon']).fit(features, labels)
        np.savez_compressed(output / 'train_features.npz', features=features.numpy(), labels=labels)
        save_csv(output / 'prototype_training_signals.csv', [{**row, 'label': int(labels[i])}
                                                           for i, row in enumerate(metadata)])
        np.savez_compressed(output / 'prototypes.npz', prototypes=prototype.prototypes.numpy(),
                            raw_means=prototype.raw_means.numpy(), class_counts=prototype.counts.numpy())
        info['prototype_training_count'] = len(labels)
        info['prototype_class_counts'] = prototype.counts.tolist()
        del train_iq, features, labels, metadata
        known = load_view(args.prepared_root, split_file, 'validation_known')
        unknown = load_view(args.prepared_root, split_file, 'validation_unknown')
        labels = np.concatenate([known[1], unknown[1]])
        metadata = known[2] + unknown[2]
        features = extract_features(model, torch.cat([known[0], unknown[0]]), batch_size)
        del known, unknown
        _, predictions, scores = infer(prototype, features)
        selection = select_threshold(labels, predictions, scores)
        selection['selected_from'] = ['validation_known', 'validation_unknown']
        save_json(output / 'threshold.json', selection)
        info['threshold_selection'] = selection
        results = {'validation': export_view(prototype, features, labels, metadata, selection['threshold'],
                                              output, 'validation', plan['class_map'])}
        del features, labels, metadata
        info.update(status='validation_complete', prototypes_sha256=sha256_file(output / 'prototypes.npz'))
        save_json(output / 'run.json', info)
        if not args.validation_only:
            prefixes = ['test_']
            if args.supplemental:
                prefixes += ['supplemental_test_', 'paired_supplemental_test_']
            for prefix in prefixes:
                for date in plan['protocol']['test_dates']:
                    view = prefix + date
                    iq, labels, metadata = load_view(args.prepared_root, split_file, view)
                    features = extract_features(model, iq, batch_size)
                    results[view] = export_view(prototype, features, labels, metadata, selection['threshold'],
                                                output, view, plan['class_map'])
                    del iq, features, labels, metadata
                    print(f'{view}: {json.dumps(results[view])}', flush=True)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state_before.items()):
            raise ValueError('CNN state changed during prototype inference.')
        if (sha256_file(checkpoint_path) != source['checkpoint_sha256']
                or sha256_file(output / 'cnn_model.pt') != source['checkpoint_sha256']
                or sha256_file(output / 'prototypes.npz') != info['prototypes_sha256']
                or sha256_file(source_root / 'metrics.json') != info['source_metrics_sha256']):
            raise ValueError('Frozen CNN or prototype artifact changed.')
        save_json(output / 'metrics.json', results)
        save_csv(output / 'metrics.csv', [{'view': view, **result} for view, result in results.items()])
        if not args.validation_only:
            msp_results = json.loads((source_root / 'metrics.json').read_text(encoding='utf-8'))
            comparison = []
            for view, result in results.items():
                if view == 'validation' or view not in msp_results:
                    continue
                previous = msp_results[view]
                for key in ['sample_count', 'known_count', 'unknown_count']:
                    if result[key] != previous[key]:
                        raise ValueError('Comparison sample counts differ.')
                for metric in ['known_closed_set_accuracy', 'known_correct_accept_rate',
                               'known_false_reject_rate', 'unknown_false_accept_rate',
                               'balanced_open_set_accuracy', 'auroc_known_positive', 'oscr_auc']:
                    comparison.append({'view': view, 'metric': metric, 'cnn_msp': previous[metric],
                                       'cnn_cosine_prototype': result[metric],
                                       'difference_prototype_minus_msp': result[metric] - previous[metric]})
            save_csv(output / 'comparison_with_msp.csv', comparison)
        info.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
                    test_executed=not args.validation_only, cnn_state_unchanged=True,
                    threshold_sha256=sha256_file(output / 'threshold.json'))
        save_json(output / 'run.json', info)
        print(f'Completed: {output}', flush=True)
    except BaseException as exc:
        info.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path,
                        default=BASELINE_ROOT / '01_cnn_msp/output/20261006_primary_fold00_seed42')
    parser.add_argument('--config', type=Path, default=MODEL_ROOT / 'experiments/cosine_prototype_baseline.json')
    parser.add_argument('--prepared-root', type=Path, default=PREPROCESSING / 'output/prepared-wisig')
    parser.add_argument('--splits-root', type=Path, default=PREPROCESSING / 'output/splits/day_shift_draft')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--supplemental', action='store_true')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
