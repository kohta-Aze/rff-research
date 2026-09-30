#!/usr/bin/env python3
"""初期設定から共通分割・未使用件数を生成。学習・評価は実行しない。"""
import argparse
import csv
import json
from pathlib import Path

from wisig_common import sha256_file, write_json
from wisig_coverage import build_profile
from wisig_protocol import allocation_rows, build_fold, object_sha256, validate_config, validate_fold


def build_splits(prepared_root: Path, config_path: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding='utf-8'))
    rows, report = build_profile(prepared_root)
    validate_config(config, report)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / 'split_index.json'
    # 再生成中の失敗で旧完了記録を使用しない。データ本体は変更しない。
    write_json(index_path, {'schema_version': 'wisig_split_index_v1', 'complete': False})
    folds = []
    for fold in range(config['fold_count']):
        plan = build_fold(rows, report, config, fold)
        validation = validate_fold(plan, rows, report)
        filename = f'fold_{fold:02d}.json'
        plan_path = output_root / filename
        write_json(plan_path, plan)
        allocations = allocation_rows(plan, rows)
        allocation_name = f'fold_{fold:02d}_allocation.csv'
        with (output_root / allocation_name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(allocations[0]))
            writer.writeheader()
            writer.writerows(allocations)
        folds.append({'fold': fold, 'file': filename, 'file_sha256': sha256_file(plan_path),
                      'allocation_file': allocation_name, 'allocation_sha256': sha256_file(output_root / allocation_name),
                      'tx_roles': plan['tx_roles'], **validation})
    write_json(output_root / 'protocol.json', config)
    index = {'schema_version': 'wisig_split_index_v1', 'complete': True,
             'protocol_status': config['status'], 'protocol_sha256': object_sha256(config),
             'protocol_file_sha256': sha256_file(output_root / 'protocol.json'),
             'manifest_sha256': report['manifest_sha256'], 'folds': folds,
             'experiment_executed': False}
    write_json(index_path, index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    index = build_splits(args.prepared_root.resolve(), args.config.resolve(), args.output_root.resolve())
    print(f"分割生成完了: {len(index['folds'])} folds / 設定状態: {index['protocol_status']}")
    print('実験: 未実施')


if __name__ == '__main__':
    main()
