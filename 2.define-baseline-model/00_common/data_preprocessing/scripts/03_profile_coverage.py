#!/usr/bin/env python3
"""目次と空組記録から全条件の件数・Rx候補を集計。"""
import argparse
import csv
from pathlib import Path

from wisig_common import write_json
from wisig_coverage import build_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    rows, report = build_profile(args.prepared_root.resolve())
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'group_counts.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'coverage_summary.json', report)
    print(f"全組数: {report['group_count']} / 総信号数: {report['signal_count']}")
    print(f"全条件で200信号以上のRx: {len(report['all_conditions_at_least_200_rx_ids'])}台")
    print(f'保存先: {output}')


if __name__ == '__main__':
    main()
