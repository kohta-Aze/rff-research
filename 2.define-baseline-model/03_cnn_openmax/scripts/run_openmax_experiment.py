"""Calibrate native-libMR OpenMax on the frozen CNN and evaluate day shift."""
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
from run_experiment import PREPROCESSING, JST, load_view, infer, save_csv, save_json, now, git_output
from wisig_common import sha256_file
from cnn_backbone import IQClassifier
from openmax import OpenMax, openmax_scores, select_openmax_threshold
from open_set_metrics import evaluate_open_set
from open_set_outputs import export_scores


def validate_config(config, class_count):
    expected = {
        'schema_version': 'openmax_experiment_v1', 'activation': 'six_class_logits_single_channel',
        'calibration': 'correctly_classified_known_training_only', 'distance': 'eucos',
        'euclidean_scale': 200.,
        'candidate_selection': 'maximum_validation_balanced_open_set_accuracy_then_grid_order',
        'score': 'known_max_probability_if_known_wins_else_negative_unknown_probability',
        'threshold': {'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
                      'tie_break': 'highest_positive_threshold', 'unknown_probability_tie': 'reject'},
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError('Unsupported OpenMax configuration.')
    for key in ['tail_sizes', 'alpha_ranks']:
        values = config[key]
        if not values or len(values) != len(set(values)) or any(type(value) is not int for value in values):
            raise ValueError('Candidate grids must contain distinct integers.')
    if min(config['tail_sizes']) < 2 or min(config['alpha_ranks']) < 1 or max(config['alpha_ranks']) > class_count:
        raise ValueError('Invalid tail size or alpha rank.')
    if type(config['inference_batch_size']) is not int or config['inference_batch_size'] < 1:
        raise ValueError('Batch size must be a positive integer.')


def export_view(calibration, alpha_rank, logits, labels, metadata, threshold, output, view, class_map):
    probabilities = calibration.probabilities(logits, alpha_rank)
    predictions, scores, unknown_wins = openmax_scores(probabilities)
    if threshold <= 0:
        raise ValueError('OpenMax operational threshold must be positive.')
    result = export_scores(
        output, view, labels, predictions, scores, metadata, threshold, class_map,
        arrays={'logits': logits, 'probabilities': probabilities, 'unknown_wins': unknown_wins},
        per_signal={'known_max_probability': probabilities[:, :-1].max(axis=1),
                    'unknown_probability': probabilities[:, -1], 'unknown_wins': unknown_wins})
    result['unknown_winner_count'] = int(unknown_wins.sum())
    return result


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source_root = args.source_run.resolve()
    source = json.loads((source_root / 'run.json').read_text(encoding='utf-8'))
    if source['status'] != 'completed' or source['method'] != 'CNN + MSP' or not source['test_executed']:
        raise ValueError('Source must be a completed primary CNN + MSP run.')
    split_file = args.splits_root.resolve() / f'fold_{source["fold"]:02d}.json'
    if sha256_file(split_file) != source['split_sha256']:
        raise ValueError('Split differs from source CNN.')
    checkpoint_path = source_root / 'best_model.pt'
    if sha256_file(checkpoint_path) != source['checkpoint_sha256']:
        raise ValueError('Source checkpoint hash mismatch.')
    for relative, digest in source['source_files'].items():
        if sha256_file(BASELINE_ROOT / relative) != digest:
            raise ValueError(f'Source CNN/loader/metrics code changed: {relative}')
    plan = json.loads(split_file.read_text(encoding='utf-8'))
    if plan['class_map'] != source['class_map'] or plan['tx_roles'] != source['tx_roles']:
        raise ValueError('Transmitter roles or class map differ from source CNN.')
    validate_config(config, len(plan['class_map']))
    readiness_path = PREPROCESSING / 'output/data_readiness.json'
    readiness = json.loads(readiness_path.read_text(encoding='utf-8'))
    if not readiness['verification_passed'] or readiness['source_sha256'] != source['data_provenance']['source_sha256']:
        raise ValueError('Data verification does not match source CNN.')
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
    info = {
        'status': 'running', 'started_at': now(), 'method': 'CNN + OpenMax',
        'scope': 'validation_only' if args.validation_only else 'single_fold_single_seed',
        'config': config, 'config_file_sha256': sha256_file(args.config),
        'seed': source['seed'], 'fold': source['fold'], 'class_map': plan['class_map'],
        'tx_roles': plan['tx_roles'], 'data_protocol': plan['protocol'], 'data_provenance': plan['provenance'],
        'split_sha256': sha256_file(split_file), 'checkpoint_sha256': source['checkpoint_sha256'],
        'source_run': str(source_root), 'source_run_sha256': sha256_file(source_root / 'run.json'),
        'source_metrics_sha256': sha256_file(source_root / 'metrics.json'),
        'cnn_retrained': False, 'source_best_epoch': source['training_result']['best_epoch'],
        'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--short'),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'runtime': {'python': sys.version, 'device': 'cpu', 'threads': args.threads,
                    'packages': {name: version(name) for name in [*source['runtime']['packages'], 'libmr', 'Cython']}},
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
    print(f'Output: {output}; frozen CNN: epoch {info["source_best_epoch"]}; libMR {version("libmr")}', flush=True)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['architecture'] != source['config']['architecture'] or checkpoint['class_count'] != len(plan['class_map']):
            raise ValueError('Checkpoint architecture or class count differs from source run.')
        model = IQClassifier(len(plan['class_map']), **checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'])
        model.requires_grad_(False)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        batch_size = config['inference_batch_size']
        iq, train_labels, metadata = load_view(args.prepared_root, split_file, 'train')
        train_logits = infer(model, iq, train_labels, batch_size, 'cpu')[0].numpy().astype(np.float64)
        del iq
        correct = train_logits.argmax(axis=1) == train_labels
        np.savez_compressed(output / 'training_activations.npz', logits=train_logits,
                            labels=train_labels, correctly_classified=correct)
        save_csv(output / 'calibration_training_signals.csv', [
            {**row, 'label': int(train_labels[i]), 'correctly_classified': bool(correct[i])}
            for i, row in enumerate(metadata)])
        known = load_view(args.prepared_root, split_file, 'validation_known')
        unknown = load_view(args.prepared_root, split_file, 'validation_unknown')
        val_labels = np.concatenate([known[1], unknown[1]])
        val_metadata = known[2] + unknown[2]
        val_logits = infer(model, torch.cat([known[0], unknown[0]]), val_labels, batch_size, 'cpu')[0].numpy().astype(np.float64)
        del known, unknown
        candidates, best = [], None
        for tail_size in config['tail_sizes']:
            calibration = OpenMax(len(plan['class_map']), tail_size, config['distance'], config['euclidean_scale']).fit(train_logits, train_labels)
            for alpha_rank in config['alpha_ranks']:
                probabilities = calibration.probabilities(val_logits, alpha_rank)
                predictions, scores, _ = openmax_scores(probabilities)
                selection = select_openmax_threshold(val_labels, predictions, scores)
                metrics = evaluate_open_set(val_labels, predictions, scores, selection['threshold'])
                objective = metrics['balanced_open_set_accuracy']
                candidates.append({'tail_size': tail_size, 'alpha_rank': alpha_rank,
                                   'validation_objective': objective, 'threshold': selection['threshold'],
                                   'validation_auroc': metrics['auroc_known_positive'], 'validation_oscr': metrics['oscr_auc']})
                if best is None or objective > best['objective']:
                    best = {'calibration': calibration, 'alpha_rank': alpha_rank,
                            'threshold_selection': selection, 'objective': objective}
        calibration, alpha_rank = best['calibration'], best['alpha_rank']
        selection = best['threshold_selection']
        save_csv(output / 'validation_candidates.csv', candidates)
        save_json(output / 'calibration.json', {**calibration.state(), 'alpha_rank': alpha_rank})
        save_json(output / 'threshold.json', selection)
        info.update(candidate_count=len(candidates), selected_tail_size=calibration.tail_size,
                    selected_alpha_rank=alpha_rank, threshold_selection=selection,
                    correct_training_class_counts=calibration.class_counts.tolist(),
                    calibration_sha256=sha256_file(output / 'calibration.json'))
        results = {'validation': export_view(calibration, alpha_rank, val_logits, val_labels, val_metadata,
                                             selection['threshold'], output, 'validation', plan['class_map'])}
        del train_logits, train_labels, val_logits, val_labels, val_metadata, metadata
        # Save all choices before loading test signals.
        info['status'] = 'validation_complete'
        save_json(output / 'run.json', info)
        if not args.validation_only:
            for date in plan['protocol']['test_dates']:
                view = 'test_' + date
                iq, labels, metadata = load_view(args.prepared_root, split_file, view)
                logits = infer(model, iq, labels, batch_size, 'cpu')[0].numpy().astype(np.float64)
                results[view] = export_view(calibration, alpha_rank, logits, labels, metadata,
                                           selection['threshold'], output, view, plan['class_map'])
                del iq, logits, labels, metadata
                print(f'{view}: {json.dumps(results[view])}', flush=True)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state_before.items()):
            raise ValueError('CNN state changed during OpenMax calibration/inference.')
        for path, expected in [(checkpoint_path, source['checkpoint_sha256']),
                               (output / 'cnn_model.pt', source['checkpoint_sha256']),
                               (output / 'calibration.json', info['calibration_sha256']),
                               (source_root / 'metrics.json', info['source_metrics_sha256'])]:
            if sha256_file(path) != expected:
                raise ValueError(f'Frozen input or calibration changed: {path}')
        save_json(output / 'metrics.json', results)
        save_csv(output / 'metrics.csv', [{'view': view, **result} for view, result in results.items()])
        if not args.validation_only:
            references = {'cnn_msp': source_root,
                          'cnn_cosine_prototype': args.prototype_run.resolve()}
            comparison = []
            for method, root in references.items():
                previous_run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
                if previous_run['split_sha256'] != info['split_sha256'] or previous_run['checkpoint_sha256'] != info['checkpoint_sha256']:
                    raise ValueError('Comparison uses a different CNN or split.')
                previous = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
                for view, result in results.items():
                    if view == 'validation':
                        continue
                    for key in ['sample_count', 'known_count', 'unknown_count']:
                        if result[key] != previous[view][key]:
                            raise ValueError('Comparison sample counts differ.')
                    for metric in ['known_closed_set_accuracy', 'known_correct_accept_rate',
                                   'known_false_reject_rate', 'unknown_false_accept_rate',
                                   'balanced_open_set_accuracy', 'auroc_known_positive', 'oscr_auc']:
                        comparison.append({'view': view, 'reference_method': method, 'metric': metric,
                                           'reference_value': previous[view][metric], 'openmax': result[metric],
                                           'difference_openmax_minus_reference': result[metric] - previous[view][metric]})
            save_csv(output / 'comparison.csv', comparison)
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
    parser.add_argument('--source-run', type=Path, default=BASELINE_ROOT / '01_cnn_msp/output/20261006_primary_fold00_seed42')
    parser.add_argument('--prototype-run', type=Path, default=BASELINE_ROOT / '02_cnn_cosine_prototype/output/20261006_primary_fold00_seed42')
    parser.add_argument('--config', type=Path, default=MODEL_ROOT / 'experiments/openmax_baseline.json')
    parser.add_argument('--prepared-root', type=Path, default=PREPROCESSING / 'output/prepared-wisig')
    parser.add_argument('--splits-root', type=Path, default=PREPROCESSING / 'output/splits/day_shift_draft')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
