"""Summarize saved residual results without selecting parameters or running tests again."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

from paths import STUDY, read_json, repo_path, sha256, verify_contract, write_json
from run_frozen import restore_calibration


def diagnostics(arrays, baseline, calibration):
    labels = arrays['labels']
    np.testing.assert_array_equal(labels, baseline['labels'])
    known = labels >= 0
    features = arrays['features'][known].astype(np.float64)
    y = labels[known]
    classes = np.unique(y)
    centers = np.stack([features[y == label].mean(axis=0) for label in classes])
    within = np.mean([np.mean(np.sum((features[y == label] - centers[i]) ** 2, axis=1))
                      for i, label in enumerate(classes)])
    between = np.mean([np.sum((a - b) ** 2) for i, a in enumerate(centers) for b in centers[i + 1:]])
    distances = calibration.distances(arrays['logits'])
    own = distances[np.flatnonzero(known), labels[known]]
    return {
        'known_cnn_argmax_accuracy': float(np.mean(arrays['logits'][known].argmax(axis=1) == y)),
        'known_openmax_prediction_changed_fraction': float(np.mean(arrays['predictions'][known] != baseline['predictions'][known])),
        'known_true_class_eucos_distance_median': float(np.median(own)),
        'known_mean_unknown_probability': float(arrays['probabilities'][known, -1].mean()),
        'unknown_mean_unknown_probability': float(arrays['probabilities'][~known, -1].mean()),
        'embedding_between_tx_squared_distance': float(between),
        'embedding_within_tx_squared_scatter': float(within),
        'embedding_between_within_ratio': float(between / within),
    }


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def summarize(run_root, output):
    run = read_json(run_root / 'run.json')
    if run['status'] != 'completed' or not run['test_executed'] or not run['identity_reproduced']:
        raise ValueError('A completed, verified paired test run is required.')
    contract = verify_contract(run_root / 'baseline_contract.json')
    calibration = restore_calibration(read_json(repo_path(contract['baseline_reference']) / 'calibration.json'))
    config, selection, metrics = [read_json(run_root / name) for name in ['config.json', 'selection.json', 'metrics.json']]
    baseline = load_npz(run_root / 'validation_lambda_0.npz')
    candidates = []
    for candidate in selection['candidates']:
        strength = candidate['strength']
        arrays = load_npz(run_root / f'validation_lambda_{strength:g}.npz')
        reasons = []
        if candidate['relative_l2_max'] > config['max_relative_l2_per_signal']:
            reasons.append('relative_l2_exceeds_0.10')
        reference = selection['candidates'][0]['metrics']
        if candidate['metrics']['unknown_false_accept_rate'] > reference['unknown_false_accept_rate'] + config['max_unknown_far_increase']:
            reasons.append('validation_far_increase_exceeds_limit')
        if candidate['metrics']['known_closed_set_accuracy'] < reference['known_closed_set_accuracy'] - config['max_known_accuracy_decrease']:
            reasons.append('validation_known_accuracy_decrease_exceeds_limit')
        candidates.append({**candidate, 'eligible': not reasons, 'exclusion_reasons': reasons,
                           'diagnostics': diagnostics(arrays, baseline, calibration)})
    dates = []
    for view, row in metrics.items():
        if not view.startswith('test_'):
            continue
        baseline_test = load_npz(run_root / f'{view}_baseline.npz')
        selected_test = load_npz(run_root / f'{view}_processed.npz')
        np.testing.assert_array_equal(baseline_test['labels'], selected_test['labels'])
        if selection['selected']['strength'] == 0:
            for key in baseline_test:
                np.testing.assert_array_equal(baseline_test[key], selected_test[key])
        dates.append({'date': view.removeprefix('test_'), **row})
    summary = {
        'scope': 'single_fold_single_seed_exploratory', 'method': 'M1_residual_emphasis_B1',
        'fold': run['fold'], 'seed': run['seed'], 'selected_strength': selection['selected']['strength'],
        'test_used_for_selection': False, 'cnn_retrained': False, 'openmax_recalibrated': False,
        'nonzero_strength_tested_on_test': selection['selected']['strength'] != 0,
        'validation_candidates': candidates, 'paired_test': dates,
        'source_run': run_root.relative_to(STUDY).as_posix(),
        'source_sha256': {name: sha256(run_root / name) for name in
                          ['run.json', 'metrics.json', 'selection.json', 'baseline_contract.json', 'config.json']},
        'all_result_npz_sha256': {path.name: sha256(path) for path in sorted(run_root.glob('*.npz'))},
        'analysis_script_sha256': sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / '残差強調の初回結果.json', summary)
    plot(candidates, output)
    print('Validation diagnostics:')
    for row in candidates:
        print(row['strength'], row['diagnostics'])
    print(f'Summary saved: {output}')


def plot(candidates, output):
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ['Meiryo', 'Yu Gothic', 'Noto Sans CJK JP'] if name in available), None)
    if font is None:
        raise RuntimeError('Japanese font is required.')
    plt.rcParams.update({'font.family': font, 'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    rows = sorted(candidates, key=lambda row: row['strength'])
    xs = [row['strength'] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=.07, right=.98, bottom=.24, top=.74, wspace=.27)
    fig.suptitle('残差強調では改善せず、無加工が選ばれた', fontsize=20, y=.97)
    fig.text(.5, .86, '03/01 validation・fold 0 / seed 42・CNN／OpenMax／しきい値を固定', ha='center', color='#555555')
    oscr = [row['metrics']['oscr_auc'] for row in rows]
    left.plot(xs, oscr, marker='o', color='#2364A0', linewidth=2)
    left.axhline(candidates[0]['metrics']['oscr_auc'], color='#777777', linestyle='--', linewidth=1)
    for x, value in zip(xs, oscr):
        left.annotate(f'{value:.4f}', (x, value), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    left.scatter([0], [candidates[0]['metrics']['oscr_auc']], color='#C56739', s=85, zorder=5, label='選択: 無加工')
    left.set_ylabel('OSCR面積（高いほどよい）')
    left.set_ylim(.605, .631)
    left.legend(loc='lower left', frameon=False)
    for key, label, color in [('known_correct_accept_rate', '既知の正解受入率 CCR（高いほどよい）', '#2364A0'),
                              ('unknown_false_accept_rate', '未知の誤受入率 FAR（低いほどよい）', '#C56739')]:
        right.plot(xs, [row['metrics'][key] for row in rows], color=color, marker='o', label=label, linewidth=2)
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.set_ylim(.4, .85)
    right.set_ylabel('固定しきい値での割合')
    right.legend(loc='lower left', fontsize=9, frameon=False)
    for ax in (left, right):
        ax.set_xlabel('残差の強調量 λ（負: 平滑化、0: 無加工、正: 強調）')
        ax.set_xticks(xs)
        ax.grid(axis='y', color='#dddddd')
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.07, .08, 'λ=0.5は波形変化の上限10%を超過。ほかの加工候補もvalidation OSCRで無加工を下回った。', color='#555555', fontsize=10)
    fig.text(.07, .025, '選択λ=0のため4評価日の比較も無加工と同一。非ゼロ強度の別日性能は未評価。', color='#555555', fontsize=10)
    fig.savefig(output / '残差強調の初回比較.png', dpi=180)
    with (output / '残差強調の初回比較.svg').open('w', encoding='utf-8', newline='\n') as stream:
        fig.savefig(stream, format='svg')
    svg_path = output / '残差強調の初回比較.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines()) + '\n',
                        encoding='utf-8', newline='\n')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=STUDY / 'output/20261007_residual_fold00_seed42')
    parser.add_argument('--output', type=Path, default=STUDY / '実験結果')
    args = parser.parse_args()
    summarize(args.run, args.output)
