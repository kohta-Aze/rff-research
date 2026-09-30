#!/usr/bin/env python3
"""共通形式へ変換した WiSig の全ファイルを manifest と照合する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from wisig_common import sha256_file, summarize_array, write_json


def parse_args() -> argparse.Namespace:
    """変換済みデータの場所と検査結果の保存先を読み取る。

    入力:
        端末から渡す ``--prepared-root`` と ``--output``。
    出力:
        2つの Path を属性として持つ Namespace。
    目的:
        どの変換結果を検査したかを明確にします。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    """manifest.csv を1行ずつ辞書として読み込む。

    入力:
        path: manifest.csv の場所。
    出力:
        各行を辞書にしたリスト。
    目的:
        期待するファイルとラベルを、実ファイルと照合できる形にします。
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_row(root: Path, row: dict[str, str]) -> list[str]:
    """manifest の1行と、その NPZ ファイルが一致するか調べる。

    入力:
        root: 変換済みデータの親ディレクトリ。
        row: manifest.csv の1行。
    出力:
        見つかった問題の短い説明リスト。問題がなければ空リスト。
    目的:
        欠損、書き換わり、配列の形や型のずれをファイル単位で発見します。
    """

    errors: list[str] = []
    file_path = root / row["relative_path"]
    if not file_path.is_file():
        return ["missing_file"]
    if sha256_file(file_path) != row["file_sha256"]:
        errors.append("sha256_mismatch")

    try:
        with np.load(file_path, allow_pickle=False) as payload:
            if set(payload.files) != {"iq"}:
                errors.append("unexpected_npz_keys")
                return errors
            iq = payload["iq"]
    except (OSError, ValueError) as exc:
        return errors + [f"npz_read_error:{type(exc).__name__}"]

    quality = summarize_array(iq)
    if not quality["shape_valid"]:
        errors.append("invalid_shape")
    if quality["finite_fraction"] != 1.0:
        errors.append("non_finite_value")
    if quality["dtype"] != row["dtype"]:
        errors.append("dtype_mismatch")
    if quality["signal_count"] != int(row["signal_count"]):
        errors.append("signal_count_mismatch")
    if quality["sample_length"] != int(row["sample_length"]):
        errors.append("sample_length_mismatch")
    if quality["component_count"] != int(row["component_count"]):
        errors.append("component_count_mismatch")
    return errors


def main() -> None:
    """manifest の全行を検査し、全体の合否を JSON へ保存する。

    入力:
        `parse_args` で受け取る変換済みデータと JSON 保存先。
    出力:
        問題件数、問題ファイル、`verification_passed` を含む JSON。
    目的:
        学習開始前に、変換結果が完全で変化していないことを確認します。
    """

    args = parse_args()
    root = args.prepared_root.expanduser().resolve()
    manifest_path = root / "manifest.csv"
    summary_path = root / "dataset_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "manifest.csv または dataset_summary.json がありません: " f"{root}"
        )

    rows = read_manifest(manifest_path)
    expected_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    problem_rows: list[dict[str, Any]] = []
    for row in rows:
        errors = verify_row(root, row)
        if errors:
            problem_rows.append(
                {"relative_path": row.get("relative_path", ""), "errors": errors}
            )

    manifest_count_matches_summary = len(rows) == int(
        expected_summary["group_count"]
    )
    report = {
        "schema_version": "wisig_manyrx_verification_v1",
        "prepared_root": str(root),
        "manifest_row_count": len(rows),
        "summary_group_count": int(expected_summary["group_count"]),
        "manifest_count_matches_summary": manifest_count_matches_summary,
        "problem_file_count": len(problem_rows),
        "problem_files": problem_rows,
        "verification_passed": manifest_count_matches_summary
        and not problem_rows
        and bool(rows),
    }
    write_json(args.output.expanduser().resolve(), report)
    print(f"検査結果を保存しました: {args.output.expanduser().resolve()}")
    print(f"合格: {report['verification_passed']}")


if __name__ == "__main__":
    main()
