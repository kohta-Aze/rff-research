#!/usr/bin/env python3
"""保存済み目次と空組記録から、全条件の件数と分割候補を集計。"""
from __future__ import annotations

from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path

from wisig_common import sha256_file, write_json


def build_profile(prepared_root: Path) -> tuple[list[dict], dict]:
    """各 Tx × Rx × 日付の件数を復元し、欠測・少数組・完全な Rx を集計。"""
    manifest = prepared_root / 'manifest.csv'
    skipped_path = prepared_root / 'skipped_groups.json'
    summary_path = prepared_root / 'dataset_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    skipped = json.loads(skipped_path.read_text(encoding='utf-8'))
    if summary['limited_output_for_code_check'] or skipped['limited_output_for_code_check']:
        raise ValueError('限定変換の結果では全体の分割候補を評価できない。')
    with manifest.open(encoding='utf-8', newline='') as handle:
        saved = list(csv.DictReader(handle))
    rows = []
    seen = set()
    file_paths = set()
    labels = {'tx': {}, 'rx': {}, 'day': {}}
    for row in saved + skipped['groups']:
        key = tuple(int(row[f'{axis}_index']) for axis in ('tx', 'rx', 'day'))
        if key in seen:
            raise ValueError(f'組み合わせの重複: {key}')
        seen.add(key)
        for axis, field, index in zip(('tx', 'rx', 'day'), ('tx_id', 'rx_id', 'capture_date'), key):
            label = row[field]
            if index in labels[axis] and labels[axis][index] != label:
                raise ValueError(f'添字とラベルの不一致: {axis}/{index}')
            labels[axis][index] = label
        count = int(row['signal_count'])
        saved_group = 'relative_path' in row
        if saved_group:
            relative = row['relative_path']
            if not relative or relative in file_paths:
                raise ValueError('保存パスの欠落・重複。')
            file_paths.add(relative)
            if int(row['sample_length']) != 256 or int(row['component_count']) != 2:
                raise ValueError('目次の信号形が不正。')
        elif row.get('shape') != [0, 256, 2] or row.get('reason') != 'empty_source_group':
            raise ValueError('空組記録の形・理由が不正。')
        if row['representation'] != summary['representation']:
            raise ValueError('表現の不一致。')
        if count < 0 or saved_group != (count > 0):
            raise ValueError(f'保存状態と件数の不一致: {key}')
        rows.append({
            'tx_index': key[0], 'tx_id': row['tx_id'],
            'rx_index': key[1], 'rx_id': row['rx_id'],
            'day_index': key[2], 'capture_date': row['capture_date'],
            'representation': row['representation'], 'signal_count': count,
            'status': 'available' if saved_group else 'empty',
            'relative_path': row.get('relative_path', ''),
            'file_sha256': row.get('file_sha256', ''),
            'dtype': row.get('dtype', ''),
        })
    expected = set(product(labels['tx'], labels['rx'], labels['day']))
    if any(set(mapping) != set(range(len(mapping))) or len(set(mapping.values())) != len(mapping)
           for mapping in labels.values()):
        raise ValueError('ラベルの重複または添字の欠落。')
    if seen != expected:
        raise ValueError('記録のない組み合わせがある。空組と未記録を区別して確認する必要がある。')
    if len(saved) != summary['group_count'] or len(skipped['groups']) != summary['skipped_empty_group_count']:
        raise ValueError('集計と目次の組数が不一致。')
    if skipped['representation'] != summary['representation'] or skipped['skipped_group_count'] != len(skipped['groups']):
        raise ValueError('空組の表現・集計が不一致。')
    if len(rows) != summary['source_group_count'] or sum(r['signal_count'] for r in rows) != summary['total_signal_count']:
        raise ValueError('集計と目次の対象組数・信号数が不一致。')
    rows.sort(key=lambda r: (r['tx_index'], r['rx_index'], r['day_index']))
    rx_profiles = []
    for index, rx_id in sorted(labels['rx'].items()):
        subset = [r for r in rows if r['rx_index'] == index]
        counts = [r['signal_count'] for r in subset]
        rx_profiles.append({
            'rx_index': index, 'rx_id': rx_id, 'group_count': len(subset),
            'empty_group_count': counts.count(0), 'minimum_signal_count': min(counts),
            'total_signal_count': sum(counts),
        })
    days = []
    for index, date in sorted(labels['day'].items()):
        subset = [r for r in rows if r['day_index'] == index]
        days.append({'capture_date': date, 'empty_group_count': sum(r['signal_count'] == 0 for r in subset),
                     'available_group_count': sum(r['signal_count'] > 0 for r in subset),
                     'total_signal_count': sum(r['signal_count'] for r in subset)})
    report = {
        'schema_version': 'wisig_manyrx_coverage_v1',
        'representation': summary['representation'], 'source_sha256': summary['source_sha256'],
        'manifest_sha256': sha256_file(manifest), 'skipped_groups_sha256': sha256_file(skipped_path),
        'dataset_summary_sha256': sha256_file(summary_path),
        'labels': {axis: [value for _, value in sorted(mapping.items())] for axis, mapping in labels.items()},
        'group_count': len(rows), 'signal_count': sum(r['signal_count'] for r in rows),
        'signal_count_histogram': dict(sorted(Counter(r['signal_count'] for r in rows).items())),
        'rx_profiles': rx_profiles, 'day_profiles': days,
        'all_conditions_positive_rx_ids': [r['rx_id'] for r in rx_profiles if r['minimum_signal_count'] > 0],
        'all_conditions_at_least_200_rx_ids': [r['rx_id'] for r in rx_profiles if r['minimum_signal_count'] >= 200],
        'nonzero_below_200_groups': [r for r in rows if 0 < r['signal_count'] < 200],
        'scope_note': '目次のカバレッジ確認。波形の再検査・分割の確定・精度評価は行わない。',
    }
    return rows, report
