"""Train common-CNN MTPL, fit a train-only global GPD, freeze day-shift tests."""
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
PROTOTYPE_SCRIPTS = BASELINE_ROOT / '02_cnn_cosine_prototype/scripts'
sys.path.insert(0, str(MSP_SCRIPTS))
sys.path.insert(0, str(PROTOTYPE_SCRIPTS))
sys.path.insert(0, str(MODELING / 'scripts'))
from run_experiment import PREPROCESSING, JST, load_view, known_validation_loss, save_csv, save_json, now, git_output
from run_prototype_experiment import extract_features
from wisig_common import sha256_file
from mtpl import MultiTaskPrototype, multitask_loss
from gpd_evt import GlobalGPD, squared_distances, select_gpd_threshold
from open_set_metrics import evaluate_open_set
from open_set_outputs import export_scores


def validate_config(config, reference):
    fixed = {'schema_version': 'mtpl_evt_experiment_v1',
             'variant': 'common_real_cnn_not_paper_complex_cnn_reproduction',
             'decoder': 'linear_relu_reshape_128x32_convtranspose_64_32_2_kernel4_stride2_padding1',
             'secondary_ranking': 'negative_minimum_distance_to_expose_probability_ties_no_retuning'}
    if any(config[key] != value for key, value in fixed.items()):
        raise ValueError('Unsupported MTPL variant.')
    if config['architecture'] != reference['config']['architecture']:
        raise ValueError('CNN architecture must match the MSP comparison.')
    if config['prototype'] != {'construction': 'jointly_learned_not_posthoc_class_means',
                               'initialization': 'gaussian', 'initialization_std': .01, 'normalization': 'none'}:
        raise ValueError('Unsupported learned prototype convention.')
    settings = config['training']
    fixed_training = {'optimizer': 'Adam', 'selection': 'minimum_validation_known_classifier_cross_entropy',
                      'augmentation': 'none', 'norm_reduction': 'sum_coordinates_then_mean_samples',
                      'initialization': 'from_scratch_same_seed_as_msp'}
    if any(settings[key] != value for key, value in fixed_training.items()):
        raise ValueError('Unsupported MTPL training convention.')
    for key in ['batch_size', 'max_epochs', 'early_stopping_patience']:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError('Training counts must be positive integers.')
    if not np.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0:
        raise ValueError('Learning rate must be positive and finite.')
    for key in ['weight_decay', 'reconstruction_weight', 'prototype_weight']:
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError('Loss weights and weight decay must be nonnegative and finite.')
    if config['evt'] != {'fit': 'global_true_class_training_squared_euclidean_gpd_mle',
                         'quantile': .9, 'quantile_method': 'linear', 'location': 0.,
                         'minimum_tail_count': 20, 'body_score': 1., 'paper_delta': .01}:
        raise ValueError('Unsupported EVT convention.')
    if config['threshold'] != {'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
                               'tie_break': 'highest_threshold', 'allowed_range': '0 < delta <= 1',
                               'acceptance': 'gpd_survival_probability_greater_than_or_equal_to_delta'}:
        raise ValueError('Unsupported threshold convention.')
    if type(config['inference_batch_size']) is not int or config['inference_batch_size'] < 1:
        raise ValueError('Inference batch size must be positive.')


def train(model, training, validation, config, seed, output):
    settings = config['training']
    iq, labels, _ = training
    if np.any(labels < 0) or np.any(validation[1] < 0):
        raise ValueError('Unknown signals cannot enter training/checkpoint selection.')
    targets = torch.from_numpy(labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    for epoch in range(1, settings['max_epochs'] + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        totals = {'total': 0., 'classification': 0., 'reconstruction': 0., 'prototype': 0.}
        for start in range(0, len(labels), settings['batch_size']):
            indices = order[start:start + settings['batch_size']]
            batch, target = iq[indices], targets[indices]
            optimizer.zero_grad(set_to_none=True)
            logits, reconstruction, features = model(batch)
            loss, parts = multitask_loss(logits, reconstruction, features, model.prototypes, batch, target,
                                         settings['reconstruction_weight'], settings['prototype_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MTPL training loss.')
            loss.backward()
            optimizer.step()
            for key, value in {'total': loss, **parts}.items():
                totals[key] += float(value.detach()) * len(indices)
        model.eval()
        val_loss, val_accuracy = known_validation_loss(model.encoder, validation[0], validation[1],
                                                       config['inference_batch_size'], 'cpu')
        val_features = extract_features(model, validation[0], config['inference_batch_size'])
        distances = squared_distances(val_features.numpy(), model.prototypes.detach().numpy())
        prototype_accuracy = float((distances.argmin(1) == validation[1]).mean())
        checkpoint = {'state_dict': model.state_dict(), 'architecture': config['architecture'],
                      'class_count': model.encoder.classifier.out_features,
                      'prototype_std': config['prototype']['initialization_std'], 'epoch': epoch,
                      'validation_known_loss': val_loss}
        if epoch == 1:
            torch.save(checkpoint, output / 'epoch_001_model.pt')
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(checkpoint, output / 'best_model.pt')
        else:
            stale += 1
        history.append({'epoch': epoch, **{f'train_{key}_loss': value / len(labels) for key, value in totals.items()},
                        'validation_known_loss': val_loss, 'validation_known_classifier_accuracy': val_accuracy,
                        'validation_known_prototype_accuracy': prototype_accuracy,
                        'seconds': time.perf_counter() - started})
        save_csv(output / 'history.csv', history)
        print(f'Epoch {epoch:02d}: total={history[-1]["train_total_loss"]:.4f} '
              f'val_ce={val_loss:.4f} classifier_acc={val_accuracy:.4f} '
              f'prototype_acc={prototype_accuracy:.4f} seconds={history[-1]["seconds"]:.1f}', flush=True)
        if stale >= settings['early_stopping_patience']:
            break
    checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return {'epochs_executed': len(history), 'best_epoch': best_epoch,
            'best_validation_known_loss': best_loss,
            'stop_reason': 'early_stopping' if stale >= settings['early_stopping_patience'] else 'max_epochs'}


def export_view(model, prototypes, calibration, data, config, threshold, output, view, class_map):
    iq, labels, metadata = data
    features = extract_features(model, iq, config['inference_batch_size']).numpy()
    distances = squared_distances(features, prototypes)
    predictions, minimum = distances.argmin(1), distances.min(1)
    scores = calibration.scores(minimum)
    paper_predictions = np.where(scores >= config['evt']['paper_delta'], predictions, -1)
    metrics = export_scores(output, view, labels, predictions, scores, metadata, threshold, class_map,
                            arrays={'features': features, 'distances': distances, 'minimum_distances': minimum,
                                    'paper_open_set_predictions': paper_predictions},
                            per_signal={'minimum_squared_distance': minimum, 'paper_open_set_prediction': paper_predictions})
    metrics['body_score_count'] = int((scores == 1.).sum())
    raw_metrics = evaluate_open_set(labels, predictions, -minimum, -calibration.threshold)
    metrics['distance_ranking_auroc'] = raw_metrics['auroc_known_positive']
    metrics['distance_ranking_oscr'] = raw_metrics['oscr_auc']
    paper = evaluate_open_set(labels, predictions, scores, config['evt']['paper_delta'])
    return metrics, paper


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    reference_root = args.reference_run.resolve()
    reference = json.loads((reference_root / 'run.json').read_text(encoding='utf-8'))
    if reference['status'] != 'completed' or reference['method'] != 'CNN + MSP' or not reference['test_executed']:
        raise ValueError('Reference must be a completed primary MSP run.')
    validate_config(config, reference)
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError('max_epochs must be positive.')
        config['training']['max_epochs'] = args.max_epochs
    split_file = args.splits_root.resolve() / f'fold_{reference["fold"]:02d}.json'
    plan = json.loads(split_file.read_text(encoding='utf-8'))
    if (sha256_file(split_file) != reference['split_sha256'] or plan['class_map'] != reference['class_map']
            or plan['tx_roles'] != reference['tx_roles']):
        raise ValueError('Comparison split or class roles differ.')
    if sha256_file(reference_root / 'best_model.pt') != reference['checkpoint_sha256']:
        raise ValueError('Reference CNN checkpoint changed.')
    for relative, digest in reference['source_files'].items():
        if sha256_file(BASELINE_ROOT / relative) != digest:
            raise ValueError(f'Reference code changed: {relative}')
    readiness_path = PREPROCESSING / 'output/data_readiness.json'
    readiness = json.loads(readiness_path.read_text(encoding='utf-8'))
    if not readiness['verification_passed'] or readiness['source_sha256'] != plan['provenance']['source_sha256']:
        raise ValueError('Readiness verification differs from source data.')
    if args.threads < 1:
        raise ValueError('Threads must be positive.')
    seed = reference['seed']
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    mode = 'smoke' if args.validation_only else 'primary'
    output = args.output.resolve() if args.output else MODEL_ROOT / 'output' / f'{stamp}_{mode}_fold{reference["fold"]:02d}_seed{seed}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    info = {
        'status': 'running', 'started_at': now(), 'method': 'MTPL + EVT', 'variant': config['variant'],
        'scope': 'validation_only' if args.validation_only else 'single_fold_single_seed',
        'config': config, 'config_file_sha256': sha256_file(args.config), 'seed': seed, 'fold': reference['fold'],
        'class_map': plan['class_map'], 'tx_roles': plan['tx_roles'], 'data_protocol': plan['protocol'],
        'data_provenance': plan['provenance'], 'split_sha256': sha256_file(split_file),
        'readiness_sha256': sha256_file(readiness_path), 'reference_run': str(reference_root),
        'reference_checkpoint_sha256': reference['checkpoint_sha256'],
        'reference_metrics_sha256': sha256_file(reference_root / 'metrics.json'),
        'cnn_trained_from_scratch': True, 'msp_weights_loaded': False,
        'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--short'),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'runtime': {'python': sys.version, 'device': 'cpu', 'threads': args.threads,
                    'deterministic_algorithms': True,
                    'packages': {name: version(name) for name in reference['runtime']['packages']}},
        'source_files': {},
    }
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for path in [Path(__file__), MSP_SCRIPTS / 'run_experiment.py',
                 PROTOTYPE_SCRIPTS / 'run_prototype_experiment.py',
                 *sorted((MODELING / 'scripts').glob('*.py')),
                 *sorted((PREPROCESSING / 'scripts').glob('wisig_*.py'))]:
        info['source_files'][str(path.relative_to(BASELINE_ROOT))] = sha256_file(path)
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(split_file, output / split_file.name)
    save_json(output / 'config.json', config)
    save_json(output / 'run.json', info)
    print(f'Output: {output}; common-CNN MTPL; CPU {args.threads} threads', flush=True)
    try:
        training = load_view(args.prepared_root, split_file, 'train')
        val_known = load_view(args.prepared_root, split_file, 'validation_known')
        model = MultiTaskPrototype(len(plan['class_map']), config['architecture'], config['prototype']['initialization_std'])
        torch.save({'state_dict': model.state_dict(), 'architecture': config['architecture'],
                    'class_count': len(plan['class_map']), 'prototype_std': config['prototype']['initialization_std']},
                   output / 'initial_model.pt')
        info['initial_checkpoint_sha256'] = sha256_file(output / 'initial_model.pt')
        info['trainable_parameter_count'] = sum(p.numel() for p in model.parameters())
        info['training_result'] = train(model, training, val_known, config, seed, output)
        model.eval()
        model.requires_grad_(False)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        prototypes = model.prototypes.detach().numpy().astype(np.float64).copy()
        features = extract_features(model, training[0], config['inference_batch_size']).numpy()
        calibration = GlobalGPD(config['evt']['quantile'], config['evt']['minimum_tail_count']).fit(features, training[1], prototypes)
        np.savez_compressed(output / 'train_features.npz', features=features, labels=training[1],
                            true_class_distances=calibration.training_distances, tail_excesses=calibration.excesses)
        save_csv(output / 'calibration_training_signals.csv', [{**row, 'label': int(training[1][i])}
                                                             for i, row in enumerate(training[2])])
        np.savez_compressed(output / 'learned_prototypes.npz', prototypes=prototypes,
                            class_counts=np.bincount(training[1], minlength=len(plan['class_map'])))
        save_json(output / 'gpd.json', calibration.state())
        info['evt_training_count'] = len(training[1])
        info['evt_training_class_counts'] = np.bincount(training[1], minlength=len(plan['class_map'])).tolist()
        info['gpd'] = calibration.state()
        del training, features
        val_unknown = load_view(args.prepared_root, split_file, 'validation_unknown')
        validation = (torch.cat([val_known[0], val_unknown[0]]),
                      np.concatenate([val_known[1], val_unknown[1]]), val_known[2] + val_unknown[2])
        features = extract_features(model, validation[0], config['inference_batch_size']).numpy()
        distances = squared_distances(features, prototypes)
        selection = select_gpd_threshold(validation[1], distances.argmin(1), calibration.scores(distances.min(1)))
        save_json(output / 'threshold.json', selection)
        info['threshold_selection'] = selection
        val_result, paper_result = export_view(model, prototypes, calibration, validation, config, selection['threshold'],
                                               output, 'validation', plan['class_map'])
        results, paper_results = {'validation': val_result}, {'validation': paper_result}
        del features, distances, validation, val_known, val_unknown
        info.update(status='validation_complete', checkpoint_sha256=sha256_file(output / 'best_model.pt'),
                    prototypes_sha256=sha256_file(output / 'learned_prototypes.npz'),
                    gpd_sha256=sha256_file(output / 'gpd.json'), threshold_sha256=sha256_file(output / 'threshold.json'))
        save_json(output / 'run.json', info)
        if not args.validation_only:
            for date in plan['protocol']['test_dates']:
                view = 'test_' + date
                data = load_view(args.prepared_root, split_file, view)
                results[view], paper_results[view] = export_view(model, prototypes, calibration, data, config,
                                                                selection['threshold'], output, view, plan['class_map'])
                del data
                print(f'{view}: {json.dumps(results[view])}', flush=True)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state_before.items()):
            raise ValueError('Model state changed during frozen inference.')
        for path, expected in [(output / 'best_model.pt', info['checkpoint_sha256']),
                               (output / 'learned_prototypes.npz', info['prototypes_sha256']),
                               (output / 'gpd.json', info['gpd_sha256']),
                               (output / 'threshold.json', info['threshold_sha256']),
                               (reference_root / 'best_model.pt', reference['checkpoint_sha256']),
                               (reference_root / 'metrics.json', info['reference_metrics_sha256'])]:
            if sha256_file(path) != expected:
                raise ValueError(f'Frozen artifact changed: {path}')
        save_json(output / 'metrics.json', results)
        save_csv(output / 'metrics.csv', [{'view': view, **result} for view, result in results.items()])
        save_json(output / 'paper_delta_metrics.json', paper_results)
        save_csv(output / 'paper_delta_metrics.csv', [{'view': view, **result} for view, result in paper_results.items()])
        if not args.validation_only:
            references = {'cnn_msp': reference_root, 'cnn_cosine_prototype': args.prototype_run.resolve(),
                          'cnn_openmax': args.openmax_run.resolve(), 'supcon_prototype': args.supcon_run.resolve()}
            comparison = []
            for method, root in references.items():
                previous_run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
                if previous_run['status'] != 'completed' or previous_run['split_sha256'] != info['split_sha256']:
                    raise ValueError('Comparison split or run status differs.')
                previous = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
                for view, result in results.items():
                    if view == 'validation':
                        continue
                    if any(result[k] != previous[view][k] for k in ['sample_count', 'known_count', 'unknown_count']):
                        raise ValueError('Comparison sample counts differ.')
                    for metric in ['known_closed_set_accuracy', 'known_correct_accept_rate',
                                   'known_false_reject_rate', 'unknown_false_accept_rate',
                                   'balanced_open_set_accuracy', 'auroc_known_positive', 'oscr_auc']:
                        comparison.append({'view': view, 'reference_method': method, 'metric': metric,
                                           'reference_value': previous[view][metric], 'mtpl_evt': result[metric],
                                           'difference_mtpl_minus_reference': result[metric] - previous[view][metric]})
            save_csv(output / 'comparison.csv', comparison)
        info.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
                    test_executed=not args.validation_only, inference_state_unchanged=True)
        save_json(output / 'run.json', info)
        print(f'Completed: {output}', flush=True)
    except BaseException as exc:
        info.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=BASELINE_ROOT / '01_cnn_msp/output/20261006_primary_fold00_seed42')
    parser.add_argument('--prototype-run', type=Path, default=BASELINE_ROOT / '02_cnn_cosine_prototype/output/20261006_primary_fold00_seed42')
    parser.add_argument('--openmax-run', type=Path, default=BASELINE_ROOT / '03_cnn_openmax/output/20261006_primary_fold00_seed42')
    parser.add_argument('--supcon-run', type=Path, default=BASELINE_ROOT / '04_supcon_prototype/output/20261006_primary_fold00_seed42')
    parser.add_argument('--config', type=Path, default=MODEL_ROOT / 'experiments/mtpl_evt_baseline.json')
    parser.add_argument('--prepared-root', type=Path, default=PREPROCESSING / 'output/prepared-wisig')
    parser.add_argument('--splits-root', type=Path, default=PREPROCESSING / 'output/splits/day_shift_draft')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
