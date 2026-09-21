"""WiSig ManyRx の3本の前処理スクリプトで共通に使う関数。

このファイルはモデルを学習しません。元データを安全寄りに読み、配列と
ラベルを同じルールで扱うための小さな道具だけをまとめています。
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import pickletools
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_KEYS = {
    "capture_date_list",
    "data",
    "equalized_list",
    "max_sig",
    "rx_list",
    "tx_list",
}

# 公式 ManyRx の NumPy 配列を復元するときに現れる名前です。
# NumPy 2 系で module 名に ``_core`` が入る場合も許可しています。
ALLOWED_PICKLE_GLOBALS = {
    "numpy.core.multiarray _reconstruct",
    "numpy._core.multiarray _reconstruct",
    "numpy ndarray",
    "numpy dtype",
    "numpy.core.multiarray scalar",
    "numpy._core.multiarray scalar",
}

REPRESENTATION_NAMES = {0: "unprocessed", 1: "equalized"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """ファイル全体から SHA-256 という内容の指紋を作る。

    目的:
        名前が同じでも中身が違うファイルを見分けるために使います。
    入力:
        path: 調べるファイルの場所。
        chunk_size: 一度に読む大きさ。大きなファイルを少しずつ読むための値。
    出力:
        16進数64文字の SHA-256 文字列。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Python の値を、人が読みやすい JSON として保存する。

    目的:
        検査条件と結果を、後から同じ形で確認できるようにします。
    入力:
        path: JSON を保存する場所。
        value: 辞書やリストなど、JSON に変換できる値。
    出力:
        戻り値はありません。指定場所に UTF-8 のファイルを作ります。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_pickle_opcodes(path: Path) -> dict[str, Any]:
    """pickle を実行する前に、入っている命令を最後まで調べる。

    目的:
        pickle は読み込み時に処理を実行できる形式です。公式 ManyRx の NumPy
        配列復元に不要な名前が入っていたら、読み込み前に止めます。
    入力:
        path: 調べる pickle ファイル。
    出力:
        命令数、参照された外部名、想定外の名前、合否を入れた辞書。
    注意:
        これは危険を減らす検査です。出所不明の pickle の安全を完全に証明する
        ものではありません。公式配布元と SHA-256 の確認も必要です。
    """

    opcode_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for opcode, argument, _ in pickletools.genops(handle):
            opcode_counts[opcode.name] += 1
            if opcode.name == "GLOBAL":
                global_counts[str(argument)] += 1
            elif opcode.name == "STACK_GLOBAL":
                global_counts["STACK_GLOBAL_UNRESOLVED"] += 1

    unexpected = sorted(set(global_counts) - ALLOWED_PICKLE_GLOBALS)
    return {
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "global_counts": dict(sorted(global_counts.items())),
        "unexpected_globals": unexpected,
        "safe_numpy_array_contract": not unexpected,
    }


def load_manyrx(path: Path) -> dict[str, Any]:
    """命令検査に合格した WiSig ManyRx pickle を読み込む。

    目的:
        3本のスクリプトが同じ安全確認とキー確認を通るようにします。
    入力:
        path: 公式 ManyRx.pkl の場所。
    出力:
        送信機一覧、受信機一覧、日付、配列などを持つ辞書。
    例外:
        想定外の pickle 命令やキーがあれば RuntimeError を出して停止します。
    """

    audit = audit_pickle_opcodes(path)
    if not audit["safe_numpy_array_contract"]:
        raise RuntimeError(
            "想定外の pickle 外部名があるため読み込みを中止します: "
            f"{audit['unexpected_globals']}"
        )

    with path.open("rb") as handle:
        dataset = pickle.load(handle)  # noqa: S301 - 上で全 opcode を検査済み

    if not isinstance(dataset, dict):
        raise RuntimeError("ManyRx の最上位が辞書ではありません。")
    if set(dataset) != EXPECTED_KEYS:
        raise RuntimeError(
            "ManyRx のキーが想定と違います: " f"{sorted(dataset)}"
        )
    return dataset


def dataset_labels(dataset: Mapping[str, Any]) -> dict[str, list[Any]]:
    """データ内の送信機・受信機・日・表現ラベルを取り出す。

    目的:
        数字の添字だけでなく、元データに書かれた本当の名前を保存します。
    入力:
        dataset: ``load_manyrx`` が返した辞書。
    出力:
        ``tx``, ``rx``, ``day``, ``representation_value`` の4リストを持つ辞書。
    """

    return {
        "tx": [str(value) for value in dataset["tx_list"]],
        "rx": [str(value) for value in dataset["rx_list"]],
        "day": [str(value) for value in dataset["capture_date_list"]],
        "representation_value": [int(value) for value in dataset["equalized_list"]],
    }


def representation_index(dataset: Mapping[str, Any], name: str) -> int:
    """``equalized`` などの名前を、入れ子配列の位置へ変換する。

    目的:
        表現の順番を思い込みで決めず、``equalized_list`` を見て選びます。
    入力:
        dataset: ``load_manyrx`` が返した辞書。
        name: ``equalized`` または ``unprocessed``。
    出力:
        ``dataset['data'][tx][rx][day]`` の中で使う0始まりの添字。
    """

    wanted_value = {value: key for key, value in REPRESENTATION_NAMES.items()}[name]
    values = [int(value) for value in dataset["equalized_list"]]
    if wanted_value not in values:
        raise RuntimeError(f"表現 {name!r} がデータ内にありません: {values}")
    return values.index(wanted_value)


def iter_source_arrays(
    dataset: Mapping[str, Any], representation: str
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """送信機・受信機・日ごとの I/Q 配列を、決まった順番で1つずつ渡す。

    目的:
        検査と変換が同じ順番・同じラベル解釈を使うようにします。
    入力:
        dataset: ``load_manyrx`` が返した辞書。
        representation: ``equalized`` または ``unprocessed``。
    出力:
        繰り返すたびに ``(ラベル情報の辞書, NumPy 配列)`` を返す iterator。
        ラベル辞書には送信機・受信機・日付と各添字が入ります。
    """

    labels = dataset_labels(dataset)
    rep_index = representation_index(dataset, representation)
    for tx_index, tx_id in enumerate(labels["tx"]):
        for rx_index, rx_id in enumerate(labels["rx"]):
            for day_index, capture_date in enumerate(labels["day"]):
                array = np.asarray(
                    dataset["data"][tx_index][rx_index][day_index][rep_index]
                )
                yield (
                    {
                        "tx_index": tx_index,
                        "tx_id": tx_id,
                        "rx_index": rx_index,
                        "rx_id": rx_id,
                        "day_index": day_index,
                        "capture_date": capture_date,
                        "representation": representation,
                    },
                    array,
                )


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    """1グループの I/Q 配列について、学習前の基本的な品質を調べる。

    目的:
        空配列、形の違い、NaN、無限大、全ゼロ信号などを早く見つけます。
        平均電力は異常の手がかりとして記録するだけで、値は変えません。
    入力:
        array: 通常は ``[信号数, 256, 2]`` の NumPy 配列。
    出力:
        形、型、有限値の割合、ゼロ信号数、RMS 振幅などを持つ辞書。
    """

    value = np.asarray(array)
    shape_valid = value.ndim == 3 and value.shape[-1] == 2
    finite = np.isfinite(value)
    finite_fraction = float(finite.mean()) if value.size else 0.0
    zero_signal_count = 0
    rms_amplitude = math.nan
    mean_i = math.nan
    mean_q = math.nan

    if shape_valid and value.shape[0] > 0:
        zero_signal_count = int(np.all(value == 0, axis=(1, 2)).sum())
        mean_i = float(np.mean(value[..., 0]))
        mean_q = float(np.mean(value[..., 1]))
        power = np.square(value[..., 0]) + np.square(value[..., 1])
        rms_amplitude = float(np.sqrt(np.mean(power)))

    return {
        "shape": [int(part) for part in value.shape],
        "dtype": str(value.dtype),
        "signal_count": int(value.shape[0]) if value.ndim >= 1 else 0,
        "sample_length": int(value.shape[1]) if value.ndim == 3 else 0,
        "component_count": int(value.shape[2]) if value.ndim == 3 else 0,
        "shape_valid": bool(shape_valid),
        "finite_fraction": finite_fraction,
        "nan_count": int(np.isnan(value).sum())
        if np.issubdtype(value.dtype, np.inexact)
        else 0,
        "inf_count": int(np.isinf(value).sum())
        if np.issubdtype(value.dtype, np.inexact)
        else 0,
        "zero_signal_count": zero_signal_count,
        "mean_i": mean_i,
        "mean_q": mean_q,
        "rms_amplitude": rms_amplitude,
    }


def group_filename(info: Mapping[str, Any]) -> str:
    """グループの添字から、衝突しない分かりやすいファイル名を作る。

    目的:
        元ラベルに空白や記号があっても安全な名前にし、実ラベルは manifest に
        そのまま残します。
    入力:
        info: ``iter_source_arrays`` が返すラベル情報。
    出力:
        例 ``tx_000__rx_004__day_002.npz`` の文字列。
    """

    return (
        f"tx_{int(info['tx_index']):03d}__"
        f"rx_{int(info['rx_index']):03d}__"
        f"day_{int(info['day_index']):03d}.npz"
    )
