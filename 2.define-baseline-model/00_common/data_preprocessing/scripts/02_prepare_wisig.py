#!/usr/bin/env python3
"""WiSig ManyRx を、5手法で共有するグループ別 NPZ へ変換する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from wisig_common import (
    group_filename,
    iter_source_arrays,
    load_manyrx,
    sha256_file,
    summarize_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """変換条件をコマンドラインから読み取る。

    入力:
        入力 pickle、出力先、表現、数の型、圧縮方法、確認用の件数上限。
    出力:
        各設定を属性として持つ Namespace。
    目的:
        実行時の条件を明示し、同じ条件を後から再現しやすくします。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ManyRx.pkl")
    parser.add_argument("--output-root", type=Path, required=True, help="変換結果の親")
    parser.add_argument(
        "--representation",
        choices=["equalized", "unprocessed"],
        required=True,
        help="取り出す信号表現",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "keep"],
        default="float32",
        help="float32 へ変換するか、元の型を保つか",
    )
    parser.add_argument(
        "--compression",
        choices=["compressed", "none"],
        default="compressed",
        help="NPZ を圧縮するか",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="0 は全件。正の値は動作確認用に先頭だけ出力",
    )
    return parser.parse_args()


def convert_dtype(array: np.ndarray, dtype_mode: str) -> np.ndarray:
    """I/Q 配列を指定された数の型へそろえる。

    入力:
        array: 元の I/Q 配列。
        dtype_mode: ``float32`` または ``keep``。
    出力:
        C 連続で、指定した型になった NumPy 配列。
    目的:
        5手法が同じ型を使えるようにします。値の正規化は行いません。
    """

    if dtype_mode == "float32":
        return np.ascontiguousarray(array, dtype=np.float32)
    return np.ascontiguousarray(array)


def save_group(path: Path, iq: np.ndarray, compression: str) -> None:
    """1グループの I/Q 配列を NPZ へ保存する。

    入力:
        path: 出力ファイルの場所。
        iq: ``[信号数, サンプル長, 2]`` の配列。
        compression: ``compressed`` または ``none``。
    出力:
        戻り値はありません。中に ``iq`` 配列を持つ NPZ を作ります。
    目的:
        モデル側が常に同じキー ``iq`` で入力を読めるようにします。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "compressed":
        np.savez_compressed(path, iq=iq)
    else:
        np.savez(path, iq=iq)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """変換した全ファイルの目次を CSV へ保存する。

    入力:
        path: `manifest.csv` の場所。
        rows: 1ファイルにつき1行のラベル・品質情報。
    出力:
        戻り値はありません。UTF-8 の CSV を作ります。
    目的:
        ファイル名と送信機・受信機・日付の対応を1か所へ記録します。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("manifest に書くグループがありません。")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """全グループを共通形式へ変換し、目次と集計を保存する。

    入力:
        ``parse_args`` で受け取る ManyRx.pkl と変換条件。
    出力:
        `signals/` の NPZ 群、`manifest.csv`、`dataset_summary.json`。
    目的:
        5手法へ同一の入力ファイルを渡せる状態を作ります。
    """

    args = parse_args()
    if args.limit_groups < 0:
        raise ValueError("--limit-groups は0以上にしてください。")
    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"入力ファイルがありません: {input_path}")

    dataset = load_manyrx(input_path)
    signal_dir = output_root / "signals" / args.representation
    rows: list[dict[str, Any]] = []

    for group_number, (info, source_array) in enumerate(
        iter_source_arrays(dataset, args.representation), start=1
    ):
        if args.limit_groups and group_number > args.limit_groups:
            break
        iq = convert_dtype(source_array, args.dtype)
        quality = summarize_array(iq)
        if not quality["shape_valid"]:
            raise RuntimeError(f"I/Q 配列の形が不正です: {info} / {quality['shape']}")
        if quality["finite_fraction"] != 1.0:
            raise RuntimeError(f"NaN または無限大があります: {info}")

        filename = group_filename(info)
        output_path = signal_dir / filename
        save_group(output_path, iq, args.compression)
        relative_path = output_path.relative_to(output_root).as_posix()
        rows.append(
            {
                **info,
                "relative_path": relative_path,
                "file_sha256": sha256_file(output_path),
                "dtype": quality["dtype"],
                "signal_count": quality["signal_count"],
                "sample_length": quality["sample_length"],
                "component_count": quality["component_count"],
                "zero_signal_count": quality["zero_signal_count"],
                "mean_i": quality["mean_i"],
                "mean_q": quality["mean_q"],
                "rms_amplitude": quality["rms_amplitude"],
            }
        )

    write_manifest(output_root / "manifest.csv", rows)
    summary = {
        "schema_version": "wisig_manyrx_prepared_v1",
        "source_path": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "source_sha256": sha256_file(input_path),
        "representation": args.representation,
        "dtype_mode": args.dtype,
        "compression": args.compression,
        "limited_output_for_code_check": bool(args.limit_groups),
        "limit_groups": args.limit_groups,
        "group_count": len(rows),
        "total_signal_count": sum(int(row["signal_count"]) for row in rows),
        "normalization_applied": False,
        "augmentation_applied": False,
        "split_applied": False,
    }
    write_json(output_root / "dataset_summary.json", summary)
    print(f"変換が完了しました: {output_root}")
    print(f"グループ数: {summary['group_count']}")
    print(f"信号数: {summary['total_signal_count']}")


if __name__ == "__main__":
    main()
