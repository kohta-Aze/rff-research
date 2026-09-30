"""共通分割の生成・検証。波形・モデルの学習処理なし。"""
from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def object_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_signal_path(root: Path, relative: str) -> Path:
    """目次のパスを保存先配下の NPZ に限定。"""
    lexical = PurePosixPath(relative.replace('\\', '/'))
    if not relative or lexical.is_absolute() or PureWindowsPath(relative).drive or '..' in lexical.parts:
        raise ValueError(f'不正な相対パス: {relative}')
    path = (root.resolve() / Path(*lexical.parts)).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix != '.npz':
        raise ValueError(f'保存先外またはNPZ以外のパス: {relative}')
    return path


def ranked(values, seed: int, namespace: str):
    """SHA-256順位で抽出。OS・配列列挙順・NumPy乱数実装に依存しない。"""
    return sorted(values, key=lambda value: (
        hashlib.sha256(f'wisig-v1|{seed}|{namespace}|{value}'.encode()).digest(), str(value)))


def validate_config(config: dict, report: dict) -> None:
    if config['schema_version'] != 'wisig_protocol_v1':
        raise ValueError('未対応のプロトコル形式。')
    if config['status'] not in ('draft_for_review', 'reviewed'):
        raise ValueError('プロトコル状態が不正。')
    if config['representation'] != report['representation']:
        raise ValueError('指定表現と保存データが不一致。')
    for field in ('known_tx_count', 'validation_unknown_tx_count', 'test_unknown_tx_count',
                  'fold_count', 'minimum_group_signals', 'train_signals_per_group',
                  'validation_signals_per_group', 'test_signals_per_group'):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f'正の整数が必要: {field}')
    if type(config['split_seed']) is not int:
        raise ValueError('split_seed は整数が必要。')
    tx_count = len(report['labels']['tx'])
    if sum(config[k] for k in ('known_tx_count', 'validation_unknown_tx_count', 'test_unknown_tx_count')) != tx_count:
        raise ValueError('Tx役割の台数が元データのTx数と不一致。')
    if config['validation_unknown_tx_count'] != config['test_unknown_tx_count'] or config['fold_count'] * config['test_unknown_tx_count'] != tx_count:
        raise ValueError('2種類の未知Txを等数とし、全Txを巡回するfold数が必要。')
    if config['normalization'] not in ('none', 'sample_rms'):
        raise ValueError('未対応の正規化。')
    if type(config['supplemental_all_receivers']) is not bool:
        raise ValueError('supplemental_all_receivers はboolが必要。')
    dates, receivers = config['test_dates'], config['primary_rx_ids']
    if not dates or len(set(dates)) != len(dates) or not set(dates) <= set(report['labels']['day']) or config['reference_date'] not in dates:
        raise ValueError('評価日付の欠落・重複・不一致。')
    if not receivers or len(set(receivers)) != len(receivers) or not set(receivers) <= set(report['labels']['rx']):
        raise ValueError('Rxの欠落・重複・不一致。')
    required = sum(config[k] for k in ('train_signals_per_group', 'validation_signals_per_group', 'test_signals_per_group'))
    if config['minimum_group_signals'] < required:
        raise ValueError('最小件数が学習・検証・同日評価の合計未満。')
    minima = {r['rx_id']: r['minimum_signal_count'] for r in report['rx_profiles']}
    if any(minima[rx] < config['minimum_group_signals'] for rx in receivers):
        raise ValueError('主実験Rxに必要件数を満たさない条件がある。')


def tx_roles(config: dict, report: dict, fold: int) -> dict:
    if type(fold) is not int or not 0 <= fold < config['fold_count']:
        raise ValueError('fold番号が範囲外。')
    order = ranked(report['labels']['tx'], config['split_seed'], 'tx-roles')
    width = config['test_unknown_tx_count']
    test = order[fold * width:(fold + 1) * width]
    next_fold = (fold + 1) % config['fold_count']
    development = order[next_fold * width:(next_fold + 1) * width]
    known = [tx for tx in order if tx not in test + development]
    return {'known': sorted(known), 'validation_unknown': sorted(development), 'test_unknown': sorted(test)}


def view_specs(config: dict, report: dict, roles: dict) -> dict:
    primary = config['primary_rx_ids']
    specs = {
        'train': (roles['known'], primary, config['reference_date'], config['train_signals_per_group'], False),
        'validation_known': (roles['known'], primary, config['reference_date'], config['validation_signals_per_group'], False),
        'validation_unknown': (roles['validation_unknown'], primary, config['reference_date'], config['validation_signals_per_group'], False),
    }
    for date in config['test_dates']:
        specs[f'test_{date}'] = (roles['known'] + roles['test_unknown'], primary, date, config['test_signals_per_group'], False)
        if config['supplemental_all_receivers']:
            specs[f'supplemental_test_{date}'] = (roles['known'] + roles['test_unknown'], report['labels']['rx'], date, config['test_signals_per_group'], True)
            specs[f'paired_supplemental_test_{date}'] = (roles['known'] + roles['test_unknown'], report['labels']['rx'], date, config['test_signals_per_group'], True)
    return specs


def paired_limits(rows: list[dict], config: dict) -> dict:
    """全評価日の共通条件について、各日に同数を使用する上限。0件は除外。"""
    counts = {}
    for row in rows:
        if row['capture_date'] in config['test_dates']:
            counts.setdefault((row['tx_id'], row['rx_id']), []).append(row['signal_count'])
    return {key: min(config['test_signals_per_group'], min(values)) for key, values in counts.items()}


def selected_indices(row: dict, view: str, config: dict, roles: dict, limit: int | None = None) -> list[int]:
    count = row['signal_count']
    order = ranked(range(count), config['split_seed'], row['relative_path'])
    train, valid, test = (config[k] for k in ('train_signals_per_group', 'validation_signals_per_group', 'test_signals_per_group'))
    if view == 'train':
        chosen = order[:train]
    elif view == 'validation_known':
        chosen = order[train:train + valid]
    elif view == 'validation_unknown':
        chosen = order[:valid]
    elif row['tx_id'] in roles['known'] and row['rx_id'] in config['primary_rx_ids'] and row['capture_date'] == config['reference_date']:
        chosen = order[train + valid:train + valid + test]
    else:
        chosen = order[:test]
    return sorted(chosen if limit is None else chosen[:limit])


def build_fold(rows: list[dict], report: dict, config: dict, fold: int) -> dict:
    validate_config(config, report)
    roles = tx_roles(config, report, fold)
    class_map = {tx: index for index, tx in enumerate(roles['known'])}
    pairs = paired_limits(rows, config)
    views = {}
    for name, (transmitters, receivers, date, target, supplemental) in view_specs(config, report, roles).items():
        groups = []
        for row in rows:
            if row['tx_id'] not in transmitters or row['rx_id'] not in receivers or row['capture_date'] != date:
                continue
            limit = pairs[(row['tx_id'], row['rx_id'])] if name.startswith('paired_') else None
            if limit == 0:
                continue
            indices = selected_indices(row, name, config, roles, limit)
            if not supplemental and len(indices) != target:
                raise ValueError(f'固定件数を抽出できない: {name}/{row}')
            role = 'known' if row['tx_id'] in roles['known'] else ('validation_unknown' if name == 'validation_unknown' else 'test_unknown')
            groups.append({**row, 'tx_role': role, 'label': class_map.get(row['tx_id'], -1), 'signal_indices': indices})
        views[name] = groups
    return {
        'schema_version': 'wisig_split_v1', 'protocol': config, 'protocol_sha256': object_sha256(config),
        'fold': fold, 'tx_roles': roles, 'class_map': class_map,
        'provenance': {key: report[key] for key in ('source_sha256', 'manifest_sha256', 'skipped_groups_sha256', 'dataset_summary_sha256')},
        'views': views,
    }


def validate_fold(plan: dict, rows: list[dict], report: dict) -> dict:
    """条件網羅、役割、件数、抽出再現性、学習・検証・評価の非重複を検証。"""
    if plan['schema_version'] != 'wisig_split_v1':
        raise ValueError('未対応の分割形式。')
    config = plan['protocol']
    validate_config(config, report)
    if object_sha256(config) != plan['protocol_sha256']:
        raise ValueError('設定ハッシュの不一致。')
    for key, value in plan['provenance'].items():
        if key not in report or report[key] != value:
            raise ValueError(f'元データのハッシュ不一致: {key}')
    if set(plan['provenance']) != {'source_sha256', 'manifest_sha256', 'skipped_groups_sha256', 'dataset_summary_sha256'}:
        raise ValueError('来歴項目が不足。')
    roles = tx_roles(config, report, plan['fold'])
    class_map = {tx: index for index, tx in enumerate(roles['known'])}
    if plan['tx_roles'] != roles or plan['class_map'] != class_map:
        raise ValueError('Tx役割・クラス番号が不一致。')
    specs = view_specs(config, report, roles)
    if set(plan['views']) != set(specs):
        raise ValueError('分割の追加・欠落。')
    source = {(r['tx_id'], r['rx_id'], r['capture_date']): r for r in rows}
    pairs = paired_limits(rows, config)
    samples, counts, coverage = {}, {}, {}
    for name, groups in plan['views'].items():
        txs, rxs, date, target, supplemental = specs[name]
        expected = set(product(txs, rxs, [date]))
        excluded = 0
        if name.startswith('paired_'):
            available = {key for key in expected if pairs[key[:2]] > 0}
            excluded = len(expected) - len(available)
            expected = available
        found, identities = set(), set()
        for group in groups:
            key = (group['tx_id'], group['rx_id'], group['capture_date'])
            if key not in expected or key in found:
                raise ValueError(f'余分な組・重複組: {name}/{key}')
            found.add(key)
            row = source[key]
            if any(group.get(field) != value for field, value in row.items()):
                raise ValueError(f'元の目次と分割の不一致: {name}/{key}')
            indices = group['signal_indices']
            if any(type(i) is not int or not 0 <= i < row['signal_count'] for i in indices) or indices != sorted(set(indices)):
                raise ValueError(f'信号番号の重複・範囲外: {name}/{key}')
            role = 'known' if key[0] in roles['known'] else ('validation_unknown' if name == 'validation_unknown' else 'test_unknown')
            if group['label'] != class_map.get(key[0], -1) or group['tx_role'] != role:
                raise ValueError(f'ラベル・Tx役割の不一致: {name}/{key}')
            limit = pairs[key[:2]] if name.startswith('paired_') else None
            if indices != selected_indices(row, name, config, roles, limit):
                raise ValueError(f'抽出規則の不一致: {name}/{key}')
            if not supplemental and len(indices) != target:
                raise ValueError(f'主実験の必要件数不足: {name}/{key}')
            identities.update((row['relative_path'], i) for i in indices)
        if found != expected:
            raise ValueError(f'条件の欠落: {name}')
        samples[name] = identities
        counts[name] = len(identities)
        coverage[name] = {'group_count': len(groups), 'empty_group_count': sum(g['signal_count'] == 0 for g in groups),
                         'short_group_count': sum(0 < g['signal_count'] < target for g in groups),
                         'excluded_missing_pair_count': excluded}
    train = samples['train']
    validation = samples['validation_known'] | samples['validation_unknown']
    test = set().union(*(value for name, value in samples.items() if 'test_' in name))
    if train & validation or train & test or validation & test:
        raise ValueError('学習・検証・評価の信号が重複。')
    return {'fold': plan['fold'], 'signal_counts': counts, 'coverage': coverage, 'verification_passed': True}


def allocation_rows(plan: dict, rows: list[dict]) -> list[dict]:
    used = {}
    for groups in plan['views'].values():
        for group in groups:
            used.setdefault(group['relative_path'], set()).update(group['signal_indices'])
    roles = plan['tx_roles']
    result = []
    for row in rows:
        role = next(name for name, ids in roles.items() if row['tx_id'] in ids)
        count = len(used.get(row['relative_path'], set()))
        result.append({**row, 'tx_role': role, 'selected_unique_signal_count': count,
                       'unused_signal_count': row['signal_count'] - count,
                       'allocation_status': 'empty' if not row['signal_count'] else ('used' if count else 'unused')})
    return result
