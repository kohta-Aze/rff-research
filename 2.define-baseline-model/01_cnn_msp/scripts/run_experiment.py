"""Train CNN + MSP on an existing WiSig fold, then evaluate frozen day-shift tests."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

MODEL_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MODEL_ROOT.parent
PREPROCESSING = BASELINE_ROOT / '00_common' / 'data_preprocessing'
MODELING = BASELINE_ROOT / '00_common' / 'modeling'
sys.path.insert(0, str(PREPROCESSING / 'scripts'))
sys.path.insert(0, str(MODELING / 'scripts'))
from wisig_common import sha256_file
from wisig_dataset import WisigDataset
from cnn_backbone import IQClassifier
from open_set_metrics import evaluate_open_set, operating_curve, select_threshold

JST = timezone(timedelta(hours=9))


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                          encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args):
    return subprocess.check_output(['git', *args], cwd=BASELINE_ROOT,
                                   text=True, encoding='utf-8').strip()


def validate_config(config):
    if config['schema_version'] != 'cnn_msp_experiment_v1':
        raise ValueError('Unsupported experiment schema.')
    training = config['training']
    if training['optimizer'] != 'Adam' or training['augmentation'] != 'none':
        raise ValueError('This runner implements Adam without augmentation.')
    if training['selection'] != 'minimum_validation_known_cross_entropy':
        raise ValueError('Unsupported checkpoint selection rule.')
    if any(type(training[key]) is not int or training[key] < 1
           for key in ['batch_size', 'max_epochs', 'early_stopping_patience']):
        raise ValueError('Training counts must be positive integers.')
    if not np.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        raise ValueError('Learning rate must be positive and finite.')
    if not np.isfinite(training['weight_decay']) or training['weight_decay'] < 0:
        raise ValueError('Weight decay must be nonnegative and finite.')
    expected = {
        'rule': 'maximize_mean_known_correct_accept_and_unknown_reject',
        'tie_break': 'highest_threshold',
        'acceptance': 'msp_greater_than_or_equal_to_threshold', 'temperature': 1.0,
    }
    if config['threshold'] != expected:
        raise ValueError('Unsupported threshold rule or temperature.')


def load_view(prepared_root, split_file, view):
    dataset = WisigDataset(prepared_root, split_file, view, cache_groups=512)
    if not len(dataset):
        raise ValueError(f'No samples in {view}.')
    batches = list(dataset.iter_batches(512))
    iq = np.concatenate([batch[0] for batch in batches])
    labels = np.concatenate([batch[1] for batch in batches])
    metadata = [row for batch in batches for row in batch[2]]
    tensor = torch.from_numpy(np.ascontiguousarray(iq.transpose(0, 2, 1)))
    print(f'Loaded {view}: {len(labels)} signals', flush=True)
    return tensor, labels, metadata


def infer(model, iq, labels, batch_size, device):
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            logits.append(model(iq[start:start + batch_size].to(device)).cpu())
    logits = torch.cat(logits)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite model outputs.')
    probabilities = logits.softmax(dim=1)
    scores, predictions = probabilities.max(dim=1)
    return logits, predictions.numpy(), scores.numpy().astype(np.float64)


def known_validation_loss(model, iq, labels, batch_size, device):
    logits, predictions, scores = infer(model, iq, labels, batch_size, device)
    loss = float(nn.functional.cross_entropy(logits, torch.from_numpy(labels)))
    return loss, float(np.mean(predictions == labels))


def train(model, train_data, validation_data, config, seed, device, output):
    settings = config['training']
    iq, labels, _ = train_data
    if np.any(labels < 0) or np.any(validation_data[1] < 0):
        raise ValueError('Unknown signals cannot enter supervised training/model selection.')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    targets = torch.from_numpy(labels)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    for epoch in range(1, settings['max_epochs'] + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        loss_sum, correct_count = 0., 0
        for start in range(0, len(labels), settings['batch_size']):
            indices = order[start:start + settings['batch_size']]
            batch, target = iq[indices].to(device), targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct_count += int((logits.detach().argmax(dim=1) == target).sum())
        val_loss, val_accuracy = known_validation_loss(
            model, validation_data[0], validation_data[1], settings['batch_size'], device)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({
                'state_dict': model.state_dict(), 'architecture': config['architecture'],
                'class_count': model.classifier.out_features, 'epoch': epoch,
                'validation_known_loss': val_loss,
            }, output / 'best_model.pt')
        else:
            stale += 1
        history.append({
            'epoch': epoch, 'train_loss': loss_sum / len(labels),
            'train_accuracy': correct_count / len(labels),
            'validation_known_loss': val_loss, 'validation_known_accuracy': val_accuracy,
            'seconds': time.perf_counter() - epoch_started,
        })
        save_csv(output / 'history.csv', history)
        print(f'Epoch {epoch:02d}: train_loss={history[-1]["train_loss"]:.4f} '
              f'val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} '
              f'seconds={history[-1]["seconds"]:.1f}', flush=True)
        if stale >= settings['early_stopping_patience']:
            break
    checkpoint = torch.load(output / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return {'epochs_executed': len(history), 'best_epoch': best_epoch,
            'best_validation_known_loss': best_loss,
            'stop_reason': 'early_stopping' if stale >= settings['early_stopping_patience'] else 'max_epochs'}


def score_view(model, data, threshold, batch_size, device, output, view, class_map):
    iq, labels, metadata = data
    logits, predictions, scores = infer(model, iq, labels, batch_size, device)
    metrics = evaluate_open_set(labels, predictions, scores, threshold)
    inverse_map = {value: key for key, value in class_map.items()}
    accepted_labels = np.where(scores >= threshold, predictions, -1)
    rows = []
    for i, row in enumerate(metadata):
        rows.append({
            **row, 'true_label': int(labels[i]), 'closed_set_prediction': int(predictions[i]),
            'closed_set_predicted_tx': inverse_map[int(predictions[i])],
            'msp': float(scores[i]), 'open_set_prediction': int(accepted_labels[i]),
            'open_set_predicted_tx': inverse_map.get(int(accepted_labels[i]), 'unknown'),
        })
    save_csv(output / f'{view}_predictions.csv', rows)
    np.savez_compressed(output / f'{view}_logits.npz', logits=logits.numpy(),
                        labels=labels, predictions=predictions, msp=scores)
    if metrics['known_count'] and metrics['unknown_count']:
        curve = operating_curve(labels, predictions, scores)
        save_csv(output / f'{view}_oscr.csv', [
            {key: float(value[i]) for key, value in curve.items()}
            for i in range(len(curve['threshold']))
        ])
    subgroup_rows = []
    for field in ['tx_id', 'rx_id']:
        values = np.asarray([row[field] for row in metadata])
        for value in sorted(set(values)):
            mask = values == value
            subgroup_rows.append({'view': view, 'group_by': field, 'group': value,
                                  **evaluate_open_set(labels[mask], predictions[mask], scores[mask], threshold)})
    save_csv(output / f'{view}_subgroups.csv', subgroup_rows)
    confusion = []
    for true_label in [-1, *range(len(class_map))]:
        for predicted_label in [-1, *range(len(class_map))]:
            confusion.append({
                'true_label': true_label, 'predicted_label': predicted_label,
                'true_tx': inverse_map.get(true_label, 'unknown'),
                'predicted_tx': inverse_map.get(predicted_label, 'unknown'),
                'count': int(((labels == true_label) & (accepted_labels == predicted_label)).sum()),
            })
    save_csv(output / f'{view}_confusion.csv', confusion)
    return metrics


def run(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError('max_epochs must be positive.')
        config['training']['max_epochs'] = args.max_epochs
    if args.validation_only:
        config['name'] += '_validation_smoke'
    split_file = args.splits_root.resolve() / f'fold_{args.fold:02d}.json'
    plan = json.loads(split_file.read_text(encoding='utf-8'))
    readiness = json.loads(args.readiness.read_text(encoding='utf-8'))
    if readiness.get('verification_passed') is not True:
        raise ValueError('Data readiness verification has not passed.')
    if readiness['source_sha256'] != plan['provenance']['source_sha256']:
        raise ValueError('Readiness source hash does not match the fold.')
    if args.threads < 1 or args.seed < 0:
        raise ValueError('threads must be positive and seed nonnegative.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable.')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(args.seed)
    stamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    mode = 'smoke' if args.validation_only else 'primary'
    output = args.output.resolve() if args.output else MODEL_ROOT / 'output' / f'{stamp}_{mode}_fold{args.fold:02d}_seed{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    info = {
        'status': 'running', 'started_at': now(), 'method': 'CNN + MSP',
        'scope': 'validation_only' if args.validation_only else 'single_fold_single_seed',
        'config': config, 'config_file_sha256': sha256_file(config_path),
        'seed': args.seed, 'fold': args.fold, 'class_map': plan['class_map'],
        'tx_roles': plan['tx_roles'], 'data_protocol': plan['protocol'],
        'data_provenance': plan['provenance'], 'split_file': str(split_file),
        'split_sha256': sha256_file(split_file),
        'readiness_sha256': sha256_file(args.readiness),
        'prepared_root': str(args.prepared_root.resolve()),
        'git_commit': git_output('rev-parse', 'HEAD'),
        'git_status': git_output('status', '--short'),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'runtime': {'python': sys.version, 'platform': platform.platform(),
                    'device': str(device), 'threads': args.threads,
                    'logical_cpus': os.cpu_count(), 'deterministic_algorithms': True,
                    'packages': {name: version(name) for name in [
                        'numpy', 'torch', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']}},
        'source_files': {},
    }
    code_dir = output / 'source_snapshot'
    code_dir.mkdir()
    sources = [Path(__file__), *sorted((MODELING / 'scripts').glob('*.py')),
               *sorted((PREPROCESSING / 'scripts').glob('wisig_*.py'))]
    for source in sources:
        info['source_files'][str(source.relative_to(BASELINE_ROOT))] = sha256_file(source)
        shutil.copy2(source, code_dir / source.name)
    shutil.copy2(split_file, output / split_file.name)
    save_json(output / 'config.json', config)
    save_json(output / 'run.json', info)
    print(f'Output: {output}\nDevice: {device}; threads: {args.threads}', flush=True)
    try:
        train_data = load_view(args.prepared_root, split_file, 'train')
        val_known = load_view(args.prepared_root, split_file, 'validation_known')
        model = IQClassifier(len(plan['class_map']), **config['architecture']).to(device)
        info['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        info['training_result'] = train(model, train_data, val_known, config, args.seed, device, output)
        del train_data
        val_unknown = load_view(args.prepared_root, split_file, 'validation_unknown')
        validation = (torch.cat([val_known[0], val_unknown[0]]),
                      np.concatenate([val_known[1], val_unknown[1]]), val_known[2] + val_unknown[2])
        batch_size = config['training']['batch_size']
        _, predictions, scores = infer(model, validation[0], validation[1], batch_size, device)
        selection = select_threshold(validation[1], predictions, scores)
        selection['selected_from'] = ['validation_known', 'validation_unknown']
        save_json(output / 'threshold.json', selection)
        info['threshold_selection'] = selection
        results = {'validation': score_view(model, validation, selection['threshold'],
                                            batch_size, device, output, 'validation', plan['class_map'])}
        del validation, val_known, val_unknown
        # Commit checkpoint/threshold choices to disk before any test data are loaded.
        info['status'] = 'validation_complete'
        save_json(output / 'run.json', info)
        if not args.validation_only:
            prefixes = ['test_']
            if args.supplemental:
                prefixes += ['supplemental_test_', 'paired_supplemental_test_']
            for prefix in prefixes:
                for date in plan['protocol']['test_dates']:
                    view = prefix + date
                    data = load_view(args.prepared_root, split_file, view)
                    results[view] = score_view(model, data, selection['threshold'], batch_size,
                                               device, output, view, plan['class_map'])
                    del data
                    print(f'{view}: {json.dumps(results[view])}', flush=True)
        save_json(output / 'metrics.json', results)
        save_csv(output / 'metrics.csv', [{'view': view, **metrics} for view, metrics in results.items()])
        info.update(status='completed', ended_at=now(), elapsed_seconds=time.perf_counter() - started,
                    test_executed=not args.validation_only,
                    checkpoint_sha256=sha256_file(output / 'best_model.pt'),
                    threshold_sha256=sha256_file(output / 'threshold.json'))
        save_json(output / 'run.json', info)
        print(f'Completed: {output}', flush=True)
    except BaseException as exc:
        info.update(status='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=MODEL_ROOT / 'experiments' / 'cnn_msp_baseline.json')
    parser.add_argument('--prepared-root', type=Path, default=PREPROCESSING / 'output' / 'prepared-wisig')
    parser.add_argument('--splits-root', type=Path, default=PREPROCESSING / 'output' / 'splits' / 'day_shift_draft')
    parser.add_argument('--readiness', type=Path, default=PREPROCESSING / 'output' / 'data_readiness.json')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--supplemental', action='store_true')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
