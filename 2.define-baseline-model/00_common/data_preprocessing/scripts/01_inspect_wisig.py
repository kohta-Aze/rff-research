#!/usr/bin/env python3
"""WiSig ManyRx の構造と品質を、変換前に調べて JSON へ保存する。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from wisig_common import (
    REPRESENTATION_NAMES,
    audit_pickle_opcodes,
    dataset_labels,
    iter_source_arrays,
    load_manyrx,
    sha256_file,
    summarize_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """コマンドラインに書かれた入力場所と出力場所を読み取る。

    入力:
        端末から渡す ``--input`` と ``--output``。
    出力:
        ``input`` と ``output`` を属性として持つ Namespace。
    目的:
        ファイル場所をコードへ直接書かず、別の環境でも同じコードを使います。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ManyRx.pkl")
    parser.add_argument(
        "--output", type=Path, required=True, help="検査結果 JSON の保存先"
    )
    return parser.parse_args()


def inspect_representation(dataset: dict[str, Any], name: str) -> dict[str, Any]:
    """1つの表現について、全グループの形と信号数をまとめる。

    入力:
        dataset: 読み込んだ ManyRx 辞書。
        name: ``equalized`` または ``unprocessed``。
    出力:
        グループ数、総信号数、形ごとの件数、品質エラー件数の辞書。
    目的:
        特定の送信機・受信機・日に欠損や形の違いがないか把握します。
    """

    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    total_signals = 0
    non_finite_groups = 0
    invalid_shape_groups = 0
    zero_signal_groups = 0

    for _, array in iter_source_arrays(dataset, name):
        summary = summarize_array(array)
        shape_counts[str(tuple(summary["shape"]))] += 1
        dtype_counts[summary["dtype"]] += 1
        total_signals += summary["signal_count"]
        non_finite_groups += int(summary["finite_fraction"] != 1.0)
        invalid_shape_groups += int(not summary["shape_valid"])
        zero_signal_groups += int(summary["signal_count"] == 0)

    return {
        "group_count": sum(shape_counts.values()),
        "total_signal_count": total_signals,
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "invalid_shape_group_count": invalid_shape_groups,
        "non_finite_group_count": non_finite_groups,
        "empty_group_count": zero_signal_groups,
    }


def main() -> None:
    """元 pickle を検査し、データを変更せずに結果 JSON を作る。

    入力:
        ``parse_args`` で受け取る ManyRx.pkl と JSON 保存先。
    出力:
        指定した場所の検査 JSON と、完了を知らせる端末表示。
    目的:
        共通形式へ変換する前に、手元のデータが想定どおりか確認します。
    """

    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"入力ファイルがありません: {input_path}")

    audit = audit_pickle_opcodes(input_path)
    dataset = load_manyrx(input_path)
    labels = dataset_labels(dataset)
    available_names = [
        REPRESENTATION_NAMES[value] for value in labels["representation_value"]
    ]
    report = {
        "schema_version": "wisig_manyrx_inspection_v1",
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
        "pickle_audit": audit,
        "labels": labels,
        "tx_count": len(labels["tx"]),
        "receiver_count": len(labels["rx"]),
        "day_count": len(labels["day"]),
        "max_sig": int(dataset["max_sig"]),
        "representations": {
            name: inspect_representation(dataset, name) for name in available_names
        },
        "signal_order_note": "compact dataset の信号順は時系列と保証されていません",
    }
    write_json(args.output.expanduser().resolve(), report)
    print(f"検査結果を保存しました: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
