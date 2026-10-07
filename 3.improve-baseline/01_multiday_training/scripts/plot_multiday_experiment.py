"""Plot one-, two-, and three-day experiments and the calibration ablation."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "実験結果"
    with (result_dir / "結果.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Run the experiment before plotting.")

    available = {item.name for item in matplotlib.font_manager.fontManager.ttflist}
    font = next((name for name in ("Meiryo", "Yu Gothic", "Noto Sans CJK JP")
                 if name in available), None)
    if font is None:
        raise RuntimeError("A Japanese font (Meiryo, Yu Gothic, or Noto Sans CJK JP) is required.")
    plt.rcParams.update({"font.family": font, "font.size": 10,
                         "axes.unicode_minus": False, "svg.fonttype": "path"})

    dates = [datetime.strptime(f"{row['test_date']}T00:00:00", "%Y_%m_%dT%H:%M:%S")
             for row in rows if row["condition"] == "A1_1day"]
    model_specs = [
        ("A1_1day", "1日学習（03/01）", "#2364A0", "o", {"2021_03_01"}),
        ("A2_2day", "2日学習（03/01・03/08）", "#C56739", "s", {"2021_03_01", "2021_03_08"}),
        ("A3_3day", "3日学習（03/01・03/08・03/15）", "#747D42", "^", {"2021_03_01", "2021_03_08", "2021_03_15"}),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.78, wspace=0.28)
    fig.suptitle("複数日の信号で学習すると、日跨ぎ性能は改善するか", fontsize=18, y=0.96)
    fig.text(0.52, 0.88, "同じ既知学習信号数11,520・既知6 Tx×16 Rx、OpenMaxも各条件で再校正",
             ha="center", color="#555555", fontsize=10)

    for name, label, color, marker, source_days in model_specs:
        subset = {row["test_date"]: row for row in rows if row["condition"] == name}
        values = [float(subset[f"2021_{day}"]["oscr_auc"]) for day in ("03_01", "03_08", "03_15", "03_23")]
        xs = dates
        left.plot(xs, values, color=color, marker=marker, markersize=7, linewidth=2, label=label)
        for day, x, value in zip(("2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23"), xs, values):
            left.plot(x, value, marker=marker, linestyle="none", markersize=7,
                      markerfacecolor=(color if day in source_days else "white"),
                      markeredgecolor=color, markeredgewidth=1.6)
    left.set_title("全評価日のOSCR", loc="left", fontsize=13, pad=16)
    left.set_ylabel("OSCR面積（高いほどよい）")
    left.set_ylim(0.35, 0.8)
    left.set_xticks(dates)
    left.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    left.set_xlabel("評価日（2021年）")
    left.legend(loc="lower left", frameon=False, fontsize=9)

    heldout_handle = Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                            markeredgecolor="#555555", markersize=7, label="白抜き＝その日の信号を学習に未使用")
    trained_handle = Line2D([], [], marker="o", linestyle="none", markerfacecolor="#555555",
                            markeredgecolor="#555555", markersize=7, label="塗りつぶし＝その日の信号を学習に使用")
    left.add_artist(left.legend(handles=[heldout_handle, trained_handle], loc="upper right",
                                frameon=False, fontsize=8))

    final_conditions = ["A1_1day", "A2_2day", "A3_3day", "H2_single_day", "H2_pooled_three_days"]
    labels = ["1日\n学習", "2日\n学習", "3日\n学習", "校正\n1日", "校正\n3日"]
    final_rows = {row["condition"]: row for row in rows if row["test_date"] == "2021_03_23"}
    values = [float(final_rows[name]["oscr_auc"]) for name in final_conditions]
    colors = ["#2364A0", "#C56739", "#747D42", "#8A79A8", "#AF6688"]
    bars = right.bar(range(len(labels)), values, color=colors, width=0.68)
    right.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10)
    right.set_xticks(range(len(labels)), labels)
    right.set_ylim(0.35, 0.67)
    right.set_ylabel("OSCR面積（03/23評価）")
    right.set_title("最終評価日のOSCR", loc="left", fontsize=13, pad=16)
    right.axvspan(-0.45, 2.45, color="#EAF0F5", zorder=0)
    right.text(0.02, 0.96, "左3条件: CNNと校正を更新\n右2条件: 1日学習CNNを固定",
               transform=right.transAxes, ha="left", va="top", fontsize=8.5, color="#444444")

    for axis in (left, right):
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#BBBBBB")
    fig.text(0.07, 0.085, "各条件の学習・validation・全評価日は重複なし。03/23は初回比較で既に確認した探索用データ。",
             color="#555555", fontsize=9)
    fig.text(0.07, 0.048, "fold 0・seed 42の単一実行。誤差範囲・統計的有意差は評価していない。全評価日の値は結果.csv参照。",
             color="#555555", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(result_dir / f"日跨ぎ実験.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    print(result_dir / "日跨ぎ実験.png")
    print(result_dir / "日跨ぎ実験.svg")


if __name__ == "__main__":
    main()
