#!/usr/bin/env python3
"""元ファイル・保存波形・分割・ローダーの整合を検証。モデル実験なし。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from wisig_common import sha256_file, write_json
from wisig_coverage import build_profile
from wisig_dataset import WisigDataset, normalize_iq
from wisig_protocol import allocation_rows, object_sha256, safe_signal_path, validate_fold


def split_file_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or Path(name).name != name:
        raise ValueError('分割索引のファイル名が不正。')
    return path


def verify_data(prepared_root: Path, splits_root: Path, source_path: Path | None = None) -> dict:
    rows, report = build_profile(prepared_root)
    index = json.loads((splits_root / 'split_index.json').read_text(encoding='utf-8'))
    if index['schema_version'] != 'wisig_split_index_v1' or index.get('complete') is not True:
        raise ValueError('分割の生成が未完了。')
    protocol_path = splits_root / 'protocol.json'
    config = json.loads(protocol_path.read_text(encoding='utf-8'))
    if index['protocol_sha256'] != object_sha256(config) or index['protocol_file_sha256'] != sha256_file(protocol_path):
        raise ValueError('設定ファイルのハッシュ不一致。')
    if index['manifest_sha256'] != report['manifest_sha256'] or index['protocol_status'] != config['status']:
        raise ValueError('分割索引と元データ・設定状態の不一致。')
    if [entry['fold'] for entry in index['folds']] != list(range(config['fold_count'])):
        raise ValueError('foldの重複・欠落。')
    if source_path is not None and sha256_file(source_path) != report['source_sha256']:
        raise ValueError('元pickleのハッシュ不一致。')
    total, verified_paths = 0, set()
    for row in rows:
        if row['signal_count'] == 0:
            continue
        path = safe_signal_path(prepared_root, row['relative_path'])
        if sha256_file(path) != row['file_sha256']:
            raise ValueError(f"波形のハッシュ不一致: {row['relative_path']}")
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ['iq']:
                raise ValueError(f'波形キーが不正: {path.name}')
            iq = payload['iq']
        if iq.shape != (row['signal_count'], 256, 2) or str(iq.dtype) != row['dtype'] or not np.isfinite(iq).all():
            raise ValueError(f'波形の形・型・値が不正: {path.name}')
        normalized = normalize_iq(iq, config['normalization'])
        if config['normalization'] == 'sample_rms':
            rms = np.sqrt(np.mean(np.sum(normalized.astype(np.float64) ** 2, axis=-1), axis=-1))
            if not np.allclose(rms, 1.0, atol=1e-6, rtol=0):
                raise ValueError(f'正規化後の電力が不正: {path.name}')
        total += iq.shape[0]
        verified_paths.add(path)
    signal_dir = prepared_root / 'signals' / config['representation']
    extra_files = sorted(str(path.relative_to(prepared_root)) for path in signal_dir.glob('*.npz')
                         if path.resolve() not in verified_paths)
    if extra_files:
        raise ValueError(f'目次にない残存NPZ: {extra_files[:5]}')
    validations, role_counts, loader_checks = [], {role: Counter() for role in ('known', 'validation_unknown', 'test_unknown')}, 0
    for entry in index['folds']:
        plan_path = split_file_path(splits_root, entry['file'])
        if sha256_file(plan_path) != entry['file_sha256']:
            raise ValueError('分割ファイルのハッシュ不一致。')
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['protocol'] != config or plan['fold'] != entry['fold']:
            raise ValueError('分割ファイルと索引の設定・foldが不一致。')
        validation = validate_fold(plan, rows, report)
        if any(entry.get(key) != value for key, value in validation.items()) or entry['tx_roles'] != plan['tx_roles']:
            raise ValueError('分割の件数・検証結果と索引が不一致。')
        for role, transmitters in plan['tx_roles'].items():
            role_counts[role].update(transmitters)
        allocation_path = split_file_path(splits_root, entry['allocation_file'])
        if sha256_file(allocation_path) != entry['allocation_sha256']:
            raise ValueError('未使用件数表のハッシュ不一致。')
        with allocation_path.open(encoding='utf-8', newline='') as handle:
            recorded = list(csv.DictReader(handle))
        expected = [{key: str(value) for key, value in row.items()} for row in allocation_rows(plan, rows)]
        if recorded != expected:
            raise ValueError('未使用件数表と分割が不一致。')
        views = ['train', 'validation_unknown', f"test_{config['test_dates'][-1]}"]
        if config['supplemental_all_receivers']:
            views.append(f"supplemental_test_{config['test_dates'][-1]}")
        for view in views:
            dataset = WisigDataset(prepared_root, plan_path, view, cache_groups=2)
            x, y, metadata = next(dataset.iter_batches(8))
            expected_size = min(8, len(dataset))
            if x.shape != (expected_size, 256, 2) or x.dtype != np.float32 or y.dtype != np.int64 or len(metadata) != expected_size:
                raise ValueError('共通ローダーの出力形式が不正。')
            if view == 'train' and np.any(y < 0):
                raise ValueError('学習バッチに未知Txが混入。')
            if view == 'validation_unknown' and np.any(y != -1):
                raise ValueError('未知validationのラベルが不正。')
            loader_checks += 1
        validations.append(validation)
    for tx in report['labels']['tx']:
        if role_counts['test_unknown'][tx] != 1 or role_counts['validation_unknown'][tx] != 1 or role_counts['known'][tx] != config['fold_count'] - 2:
            raise ValueError('Txの役割巡回回数が不一致。')
    return {
        'schema_version': 'wisig_data_readiness_v1', 'verification_passed': True,
        'protocol_status': config['status'], 'protocol_sha256': object_sha256(config),
        'source_sha256': report['source_sha256'], 'source_file_hash_checked': source_path is not None,
        'manifest_sha256': report['manifest_sha256'], 'verified_file_count': len(verified_paths),
        'verified_signal_count': total, 'empty_group_count': sum(r['signal_count'] == 0 for r in rows),
        'fold_count': len(validations), 'folds': validations, 'loader_smoke_check_count': loader_checks,
        'normalization': config['normalization'], 'experiment_executed': False,
        'limitation': '信号番号の非重複を検証。元パケットIDがないため、異なるRx間の同一パケット対応は保証しない。',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--splits-root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    write_json(output, {'verification_passed': False, 'status': 'checking', 'experiment_executed': False})
    report = verify_data(args.prepared_root.resolve(), args.splits_root.resolve(), args.source.resolve() if args.source else None)
    write_json(output, report)
    print(f"検証合格: {report['verified_file_count']}ファイル / {report['fold_count']} folds / ローダー{report['loader_smoke_check_count']}件")
    print('実験: 未実施')


if __name__ == '__main__':
    main()
