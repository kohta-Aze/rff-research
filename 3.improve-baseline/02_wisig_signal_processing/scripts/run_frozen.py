"""B0/B1: select residual strength on source validation, then freeze for test."""
import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
import shutil

import libmr
import numpy as np
import torch

from paths import CONTRACT, STUDY, read_json, repo_path, sha256, verify_contract, write_json
from transforms import relative_l2, residual_emphasis, select_candidate
from cnn_backbone import IQClassifier
from openmax import OpenMax, openmax_scores
from open_set_metrics import evaluate_open_set
from run_experiment import load_view


def restore_calibration(state):
    calibration = OpenMax(state['class_count'], state['tail_size'], state['distance'], state['euclidean_scale'])
    calibration.means = np.asarray(state['mean_activations'], dtype=np.float64)
    calibration.tails = np.asarray(state['tail_distances'], dtype=np.float64)
    calibration.class_counts = np.asarray(state['correct_training_class_counts'])
    calibration.models = [libmr.load_from_string(value) for value in state['weibull_model_strings']]
    if len(calibration.models) != calibration.class_count or any(not m.is_valid for m in calibration.models):
        raise ValueError('Invalid saved OpenMax models.')
    for model, expected in zip(calibration.models, state['weibull_params']):
        np.testing.assert_allclose(model.get_params(), expected, rtol=1e-12, atol=1e-12)
    return calibration


def validate_config(config):
    if (config['schema_version'] != 'wisig_frozen_residual_v1' or config['condition'] != 'B1'
            or config['method'] != 'shared_residual_emphasis' or config['cnn_retrained'] is not False
            or config['openmax_recalibrated'] is not False
            or config['selection_view'] != 'source_validation_only'
            or config['selection_metric'] != 'oscr_auc' or config['tie_break'] != 'grid_order'):
        raise ValueError('Only the declared frozen B1 protocol is implemented.')
    grid = config['lambdas']
    if not grid or grid[0] != 0 or len(set(grid)) != len(grid) or any(not np.isfinite(v) or not -.5 <= v <= .5 for v in grid):
        raise ValueError('Invalid residual grid or missing identity control.')
    for key in ['max_relative_l2_per_signal', 'max_unknown_far_increase', 'max_known_accuracy_decrease']:
        if not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f'Invalid guard: {key}')
    if type(config['inference_batch_size']) is not int or config['inference_batch_size'] < 1:
        raise ValueError('Batch size must be a positive integer.')


def infer_processed(model, original, strength, batch_size):
    changed = residual_emphasis(original, strength)
    features, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(changed), batch_size):
            tensor = torch.from_numpy(np.ascontiguousarray(changed[start:start + batch_size].transpose(0, 2, 1)))
            embedding = model.features(tensor)
            features.append(embedding.numpy())
            logits.append(model.classifier(model.dropout(embedding)).numpy())
    logits, features = np.concatenate(logits).astype(np.float64), np.concatenate(features)
    if not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise ValueError('Nonfinite frozen model output.')
    return logits, features, relative_l2(original, changed)


def evaluate(model, calibration, alpha, threshold, original, labels, strength, batch_size):
    logits, features, distortion = infer_processed(model, original, strength, batch_size)
    probabilities = calibration.probabilities(logits, alpha)
    predictions, scores, _ = openmax_scores(probabilities)
    metrics = evaluate_open_set(labels, predictions, scores, threshold)
    arrays = dict(logits=logits, features=features, labels=labels, predictions=predictions,
                  scores=scores, probabilities=probabilities, relative_l2=distortion)
    return metrics, arrays


def check_identity(actual, expected):
    for key, value in actual.items():
        if key in expected and isinstance(value, (int, float)):
            if not np.isclose(value, expected[key], rtol=0, atol=1e-10):
                raise ValueError(f'Identity baseline mismatch: {key}: {value} vs {expected[key]}')


def run(args):
    if args.identity_only and args.evaluate_test:
        raise ValueError('Identity verification is validation-only.')
    if args.threads < 1:
        raise ValueError('Threads must be positive.')
    contract = verify_contract(args.contract)
    config = read_json(args.config)
    validate_config(config)
    reference = repo_path(contract['baseline_reference'])
    split = repo_path(contract['split'])
    prepared = repo_path(contract['prepared_root'])
    plan, previous = read_json(split), read_json(reference / 'metrics.json')
    state = read_json(reference / 'calibration.json')
    threshold = read_json(reference / 'threshold.json')['threshold']
    checkpoint = torch.load(reference / 'cnn_model.pt', map_location='cpu', weights_only=True)
    if checkpoint['class_count'] != len(contract['class_map']) or plan['tx_roles'] != contract['tx_roles']:
        raise ValueError('Class count or roles differ from the contract.')
    model = IQClassifier(checkpoint['class_count'], **checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    calibration = restore_calibration(state)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    output = args.output or STUDY / 'output' / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'baseline_contract.json', contract)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    scripts = sorted(Path(__file__).parent.glob('*.py'))
    for path in scripts:
        shutil.copy2(path, snapshots / path.name)
    info = dict(status='running', condition='B0_identity' if args.identity_only else 'B1',
                fold=contract['fold'], seed=contract['seed'], cnn_retrained=False,
                openmax_recalibrated=False, test_executed=False,
                exploratory=True, identity_reproduced=False,
                source_files={p.name: sha256(p) for p in scripts},
                runtime={name: version(name) for name in ['numpy', 'torch', 'scipy', 'libmr']},
                config_sha256=sha256(args.config), contract_sha256=sha256(args.contract))
    write_json(output / 'run.json', info)
    try:
        known = load_view(prepared, split, 'validation_known')
        unknown = load_view(prepared, split, 'validation_unknown')
        original = np.ascontiguousarray(torch.cat([known[0], unknown[0]]).numpy().transpose(0, 2, 1))
        labels, metadata = np.r_[known[1], unknown[1]], known[2] + unknown[2]
        del known, unknown
        write_json(output / 'validation_signals.json', metadata)
        candidates = []
        grid = [0.] if args.identity_only else config['lambdas']
        for strength in grid:
            metrics, arrays = evaluate(model, calibration, state['alpha_rank'], threshold,
                                      original, labels, strength, config['inference_batch_size'])
            if strength == 0:
                check_identity(metrics, previous['validation'])
                info['identity_reproduced'] = True
            candidates.append(dict(strength=strength, metrics=metrics,
                                   relative_l2_max=float(arrays['relative_l2'].max()),
                                   relative_l2_mean=float(arrays['relative_l2'].mean())))
            np.savez_compressed(output / f'validation_lambda_{strength:g}.npz', **arrays)
        selected = select_candidate(candidates, config)
        # Persist selection and candidate results BEFORE any test samples are loaded.
        selection = dict(selected=selected, candidates=candidates, selected_from='source_validation_only',
                         test_used_for_selection=False, fixed_threshold=threshold)
        write_json(output / 'selection.json', selection)
        info['status'] = 'identity_verified' if args.identity_only else 'validation_selected'
        write_json(output / 'run.json', info)
        results = {'validation_candidates': candidates}
        if args.evaluate_test:
            for date in plan['protocol']['test_dates']:
                view = f'test_{date}'
                iq, test_labels, metadata = load_view(prepared, split, view)
                original = np.ascontiguousarray(iq.numpy().transpose(0, 2, 1))
                write_json(output / f'{view}_signals.json', metadata)
                paired = {}
                for name, strength in [('baseline', 0.), ('processed', selected['strength'])]:
                    metrics, arrays = evaluate(model, calibration, state['alpha_rank'], threshold,
                                              original, test_labels, strength, config['inference_batch_size'])
                    if name == 'baseline':
                        check_identity(metrics, previous[view])
                    paired[name] = metrics
                    np.savez_compressed(output / f'{view}_{name}.npz', **arrays)
                paired['delta_oscr_auc'] = paired['processed']['oscr_auc'] - paired['baseline']['oscr_auc']
                results[view] = paired
            info['test_executed'] = True
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in checkpoint['state_dict'].items()):
            raise ValueError('CNN state changed.')
        verify_contract(args.contract)
        write_json(output / 'metrics.json', results)
        info.update(status='completed', selected_strength=selected['strength'], cnn_state_unchanged=True)
        write_json(output / 'run.json', info)
        print(f'Completed: {output}; identity reproduced; selected lambda={selected["strength"]}')
    except BaseException as error:
        info.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', info)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY / 'configs/residual_grid.json')
    parser.add_argument('--contract', type=Path, default=CONTRACT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--identity-only', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    run(parser.parse_args())
