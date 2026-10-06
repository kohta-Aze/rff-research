"""Train a two-view SupCon encoder, then evaluate frozen cosine prototypes."""
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
from torch.nn import functional as F

MODEL_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MODEL_ROOT.parent
MODELING = BASELINE_ROOT / '00_common/modeling'
MSP_SCRIPTS = BASELINE_ROOT / '01_cnn_msp/scripts'
PROTOTYPE_SCRIPTS = BASELINE_ROOT / '02_cnn_cosine_prototype/scripts'
sys.path.insert(0, str(MSP_SCRIPTS))
sys.path.insert(0, str(PROTOTYPE_SCRIPTS))
sys.path.insert(0, str(MODELING / 'scripts'))
from run_experiment import PREPROCESSING, JST, load_view, save_csv, save_json, now, git_output
from run_prototype_experiment import extract_features
from wisig_common import sha256_file
from cosine_prototype import CosinePrototype
from open_set_metrics import select_threshold
from open_set_outputs import export_scores
from supcon import SupConEncoder, augment_iq, supervised_contrastive_loss


def validate_config(config, reference):
    if config['schema_version'] != 'supcon_prototype_experiment_v1':
        raise ValueError('Unsupported SupCon schema.')
    if config['architecture'] != reference['config']['architecture']:
        raise ValueError('Encoder architecture must match the MSP comparison.')
    if config['projection'] != {'head': 'linear_relu_linear_l2', 'dimension': 128,
                               'inference': 'discard_projection_use_raw_encoder_embedding'}:
        raise ValueError('Unsupported projection head.')
    training = config['training']
    fixed = {'optimizer': 'Adam', 'selection': 'minimum_validation_known_prototype_cross_entropy',
             'initialization': 'from_scratch_same_seed_as_msp',
             'loss': 'supervised_contrastive_equation_2_all_anchors',
             'batch_sampling': 'seeded_random_permutation_without_replacement', 'views_per_signal': 2}
    if any(training[key] != value for key, value in fixed.items()):
        raise ValueError('Unsupported training convention.')
    for key in ['batch_size', 'max_epochs', 'early_stopping_patience']:
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError('Training counts must be positive integers.')
    for key in ['learning_rate', 'temperature', 'base_temperature', 'selection_temperature']:
        if not np.isfinite(training[key]) or training[key] <= 0:
            raise ValueError('Learning rate and temperatures must be positive and finite.')
    if not np.isfinite(training['weight_decay']) or training['weight_decay'] < 0:
        raise ValueError('Weight decay must be nonnegative and finite.')
    augmentation = training['augmentation']
    if (augmentation['renormalization'] != 'sample_rms' or augmentation['time_shift'] is not False
            or augmentation['gain_change'] is not False
            or not np.isfinite(augmentation['phase_degrees']) or not 0 <= augmentation['phase_degrees'] <= 180
            or not np.isfinite(augmentation['noise_snr_db']) or not 0 <= augmentation['noise_snr_db'] <= 100):
        raise ValueError('Unsupported augmentation.')
    if config['prototype'] != {'construction': 'normalize_mean_raw_training_embeddings',
                                'normalization_epsilon': 1e-12, 'arithmetic_dtype': 'float64'}:
        raise ValueError('Unsupported prototype convention.')
    if config['threshold'] != {'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
                               'tie_break': 'highest_threshold',
                               'acceptance': 'similarity_greater_than_or_equal_to_threshold'}:
        raise ValueError('Unsupported threshold convention.')
    if type(config['inference_batch_size']) is not int or config['inference_batch_size'] < 1:
        raise ValueError('Inference batch size must be positive.')


def fit_prototype(model, iq, labels, config):
    features = extract_features(model, iq, config['inference_batch_size'])
    prototype = CosinePrototype(model.encoder.classifier.out_features,
                                config['prototype']['normalization_epsilon']).fit(features, labels)
    return features, prototype


def train(model, training, validation, config, seed, output):
    settings = config['training']
    iq, labels, _ = training
    if np.any(labels < 0) or np.any(validation[1] < 0):
        raise ValueError('Unknown samples cannot enter training or checkpoint selection.')
    targets = torch.from_numpy(labels)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    order_generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator().manual_seed(seed + 1)
    augmentation = settings['augmentation']
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    for epoch in range(1, settings['max_epochs'] + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(labels), generator=order_generator)
        loss_sum = 0.
        for start in range(0, len(labels), settings['batch_size']):
            indices = order[start:start + settings['batch_size']]
            batch, target = iq[indices], targets[indices]
            first = augment_iq(batch, augmentation_generator, augmentation['phase_degrees'], augmentation['noise_snr_db'])
            second = augment_iq(batch, augmentation_generator, augmentation['phase_degrees'], augmentation['noise_snr_db'])
            optimizer.zero_grad(set_to_none=True)
            projections = model(torch.cat([first, second])).reshape(2, len(indices), -1).transpose(0, 1)
            loss = supervised_contrastive_loss(projections, target, settings['temperature'], settings['base_temperature'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite contrastive training loss.')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        # Checkpoint selection uses known validation only and train-only class means.
        _, prototype = fit_prototype(model, iq, labels, config)
        val_features = extract_features(model, validation[0], config['inference_batch_size'])
        similarities = prototype.similarities(val_features)
        val_loss = float(F.cross_entropy(similarities / settings['selection_temperature'],
                                         torch.from_numpy(validation[1])))
        val_accuracy = float((similarities.argmax(1).numpy() == validation[1]).mean())
        checkpoint = {'state_dict': model.state_dict(), 'architecture': config['architecture'],
                      'projection_dim': config['projection']['dimension'],
                      'class_count': model.encoder.classifier.out_features, 'epoch': epoch,
                      'validation_known_loss': val_loss}
        if epoch == 1:
            torch.save(checkpoint, output / 'epoch_001_model.pt')
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(checkpoint, output / 'best_model.pt')
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_supcon_loss': loss_sum / len(labels),
                        'validation_known_loss': val_loss, 'validation_known_accuracy': val_accuracy,
                        'seconds': time.perf_counter() - started})
        save_csv(output / 'history.csv', history)
        print(f'Epoch {epoch:02d}: SupCon={history[-1]["train_supcon_loss"]:.4f} '
              f'val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} seconds={history[-1]["seconds"]:.1f}', flush=True)
        if stale >= settings['early_stopping_patience']:
            break
    checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return {'epochs_executed': len(history), 'best_epoch': best_epoch,
            'best_validation_known_loss': best_loss,
            'stop_reason': 'early_stopping' if stale >= settings['early_stopping_patience'] else 'max_epochs'}


def export_view(model, prototype, data, config, threshold, output, view, class_map):
    iq, labels, metadata = data
    features = extract_features(model, iq, config['inference_batch_size'])
    similarities = prototype.similarities(features).numpy()
    predictions = similarities.argmax(1)
    scores = similarities.max(1)
    return export_scores(output, view, labels, predictions, scores, metadata, threshold, class_map,
                         arrays={'features': features.numpy(), 'similarities': similarities})


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
        'status': 'running', 'started_at': now(), 'method': 'SupCon + Prototype',
        'scope': 'validation_only' if args.validation_only else 'single_fold_single_seed',
        'config': config, 'config_file_sha256': sha256_file(args.config),
        'seed': seed, 'fold': reference['fold'], 'class_map': plan['class_map'], 'tx_roles': plan['tx_roles'],
        'data_protocol': plan['protocol'], 'data_provenance': plan['provenance'],
        'split_sha256': sha256_file(split_file), 'readiness_sha256': sha256_file(readiness_path),
        'reference_run': str(reference_root), 'reference_checkpoint_sha256': reference['checkpoint_sha256'],
        'reference_metrics_sha256': sha256_file(reference_root / 'metrics.json'),
        'cnn_trained_from_scratch': True, 'msp_weights_loaded': False,
        'augmentation_seed': seed + 1, 'encoder_dropout_used': False, 'classification_head_used': False,
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
    print(f'Output: {output}; new SupCon encoder; CPU {args.threads} threads', flush=True)
    try:
        training = load_view(args.prepared_root, split_file, 'train')
        val_known = load_view(args.prepared_root, split_file, 'validation_known')
        model = SupConEncoder(len(plan['class_map']), config['architecture'], config['projection']['dimension'])
        torch.save({'state_dict': model.state_dict(), 'architecture': config['architecture'],
                    'class_count': len(plan['class_map']), 'projection_dim': config['projection']['dimension']},
                   output / 'initial_model.pt')
        info['initial_checkpoint_sha256'] = sha256_file(output / 'initial_model.pt')
        info['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info['training_result'] = train(model, training, val_known, config, seed, output)
        model.eval()
        model.requires_grad_(False)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        features, prototype = fit_prototype(model, training[0], training[1], config)
        np.savez_compressed(output / 'train_features.npz', features=features.numpy(), labels=training[1])
        save_csv(output / 'prototype_training_signals.csv', [{**row, 'label': int(training[1][i])}
                                                           for i, row in enumerate(training[2])])
        np.savez_compressed(output / 'prototypes.npz', prototypes=prototype.prototypes.numpy(),
                            raw_means=prototype.raw_means.numpy(), class_counts=prototype.counts.numpy())
        info['prototype_training_count'] = len(training[1])
        info['prototype_class_counts'] = prototype.counts.tolist()
        del training, features
        val_unknown = load_view(args.prepared_root, split_file, 'validation_unknown')
        validation = (torch.cat([val_known[0], val_unknown[0]]),
                      np.concatenate([val_known[1], val_unknown[1]]), val_known[2] + val_unknown[2])
        features = extract_features(model, validation[0], config['inference_batch_size'])
        similarities = prototype.similarities(features).numpy()
        selection = select_threshold(validation[1], similarities.argmax(1), similarities.max(1))
        selection['selected_from'] = ['validation_known', 'validation_unknown']
        save_json(output / 'threshold.json', selection)
        info['threshold_selection'] = selection
        results = {'validation': export_view(model, prototype, validation, config, selection['threshold'],
                                             output, 'validation', plan['class_map'])}
        del features, similarities, validation, val_known, val_unknown
        info.update(status='validation_complete', checkpoint_sha256=sha256_file(output / 'best_model.pt'),
                    prototypes_sha256=sha256_file(output / 'prototypes.npz'),
                    threshold_sha256=sha256_file(output / 'threshold.json'))
        save_json(output / 'run.json', info)
        if not args.validation_only:
            for date in plan['protocol']['test_dates']:
                view = 'test_' + date
                data = load_view(args.prepared_root, split_file, view)
                results[view] = export_view(model, prototype, data, config, selection['threshold'],
                                            output, view, plan['class_map'])
                del data
                print(f'{view}: {json.dumps(results[view])}', flush=True)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state_before.items()):
            raise ValueError('Model state changed during frozen inference.')
        for path, expected in [(output / 'best_model.pt', info['checkpoint_sha256']),
                               (output / 'prototypes.npz', info['prototypes_sha256']),
                               (output / 'threshold.json', info['threshold_sha256']),
                               (reference_root / 'best_model.pt', reference['checkpoint_sha256']),
                               (reference_root / 'metrics.json', info['reference_metrics_sha256'])]:
            if sha256_file(path) != expected:
                raise ValueError(f'Frozen artifact changed: {path}')
        save_json(output / 'metrics.json', results)
        save_csv(output / 'metrics.csv', [{'view': view, **result} for view, result in results.items()])
        if not args.validation_only:
            references = {'cnn_msp': reference_root,
                          'cnn_cosine_prototype': args.prototype_run.resolve(),
                          'cnn_openmax': args.openmax_run.resolve()}
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
                                           'reference_value': previous[view][metric], 'supcon_prototype': result[metric],
                                           'difference_supcon_minus_reference': result[metric] - previous[view][metric]})
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
    parser.add_argument('--config', type=Path, default=MODEL_ROOT / 'experiments/supcon_prototype_baseline.json')
    parser.add_argument('--prepared-root', type=Path, default=PREPROCESSING / 'output/prepared-wisig')
    parser.add_argument('--splits-root', type=Path, default=PREPROCESSING / 'output/splits/day_shift_draft')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
