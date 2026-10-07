"""Plot recorded baseline results; no training or metric recomputation."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


def main():
    story = Path(__file__).resolve().parents[1]
    repo = next(p for p in story.parents if (p / "2.define-baseline-model").is_dir())
    source = repo / "2.define-baseline-model" / "2026-10-06_初回比較結果.json"
    record = json.loads(source.read_text(encoding="utf-8-sig"))
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Meiryo", "Yu Gothic", "Noto Sans CJK JP")
                 if name in available), None)
    if font is None:
        raise RuntimeError("Install Meiryo, Yu Gothic or Noto Sans CJK JP for Japanese labels.")
    plt.rcParams.update({"font.family": font, "font.size": 11,
                         "axes.unicode_minus": False, "svg.fonttype": "path"})

    dates = record["data_protocol"]["test_dates"]
    times = [datetime.strptime(day, "%Y_%m_%d") for day in dates]
    methods = [
        ("cnn_msp", "MSP", "#9B6C15", "o"),
        ("cnn_cosine_prototype", "Cosine", "#C56739", "s"),
        ("cnn_openmax", "OpenMax", "#2364A0", "D"),
        ("supcon_prototype", "SupCon", "#747D42", "^"),
        ("mtpl_evt", "MTPL", "#AF6688", "v"),
    ]
    # Fail on a changed source schema instead of drawing incomplete results.
    for key, _, _, _ in methods:
        for day in dates:
            metric = record["runs"][key]["metrics"][f"test_{day}"]
            if not 0 <= metric["oscr_auc"] <= 1:
                raise ValueError(f"Invalid OSCR: {key}, {day}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 6.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.72, wspace=0.25)
    fig.suptitle("日をまたぐと、既知識別と未知判定はどう変わるか", fontsize=19, x=0.52, y=0.97)
    fig.text(0.52, 0.895, "WiSig ManyRx・equalized・学習／validation: 2021-03-01・fold 0／seed 42",
             ha="center", fontsize=11, color="#555555")

    for key, label, color, marker in methods:
        values = [record["runs"][key]["metrics"][f"test_{day}"]["oscr_auc"] for day in dates]
        left.plot(times, values, label=label, color=color, marker=marker,
                  linewidth=2 if label == "OpenMax" else 1.4, markersize=6)
    left.set_title("① 全5手法のOSCR面積", loc="left", pad=18, fontsize=13)
    left.set_ylabel("OSCR面積（高いほどよい）")
    left.set_ylim(0, 1)
    left.legend(loc="lower left", ncol=3, frameon=False, fontsize=10)

    openmax = [record["runs"]["cnn_openmax"]["metrics"][f"test_{day}"] for day in dates]
    for field, label, color, marker in (
        ("known_closed_set_accuracy", "既知Txの識別精度（拒否を無視）", "#2364A0", "o"),
        ("known_correct_accept_rate", "既知Txの正解受入率（CCR）", "#9B6C15", "s"),
        ("known_false_reject_rate", "既知Txの誤拒否率", "#555555", "^"),
    ):
        values = [row[field] for row in openmax]
        right.plot(times, values, label=label, color=color, marker=marker,
                   linewidth=1.7, markersize=6)
        for when, value, row in zip(times, values, openmax):
            other = ("known_false_reject_rate" if field == "known_correct_accept_rate"
                     else "known_correct_accept_rate")
            offset = 10 if field == "known_closed_set_accuracy" or value >= row[other] else -20
            right.annotate(f"{value:.1%}", (when, value), xytext=(0, offset),
                           textcoords="offset points", ha="center", fontsize=10, color=color)
    right.set_title("② OpenMax: 既知Txの識別と受入", loc="left", pad=18, fontsize=13)
    right.set_ylabel("既知Tx 3,840信号に対する割合")
    right.set_ylim(0, 1.06)
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
                 fontsize=10, ncol=1)

    for axis in (left, right):
        axis.set_xticks(times)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        axis.set_xlabel("評価日（2021年）")
        axis.set_xlim(times[0] - timedelta(days=1.6), times[-1] + timedelta(days=1.6))
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#BBBBBB")
        axis.axvline(times[0], color="#BBBBBB", linestyle=":", linewidth=1)

    fig.text(0.065, 0.075, "各日: 既知6 Tx×16 Rx×40件＝3,840件、未知2 Tx×16 Rx×40件＝1,280件。線は観測点間の接続。",
             color="#555555", fontsize=9)
    fig.text(0.065, 0.038, "初回1 fold・1 seedの探索的結果。反復に基づく誤差範囲なし。OSCR面積と運用しきい値での性能を区別する。",
             color="#555555", fontsize=9)
    for suffix in ("png", "svg"):
        path = story / f"日跨ぎベースライン比較.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
