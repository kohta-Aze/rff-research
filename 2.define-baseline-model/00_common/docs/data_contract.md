# 共通データの約束

## 1. 元データ

- データセット: WiSig ManyRx compact
- 想定ファイル名: `ManyRx.pkl`
- 信号: Wi-Fi identification signal
- 1信号の基本形: `256 × 2`
- 最後の2列: I 成分と Q 成分
- データのまとまり: `(送信機, 受信機, 収録日, 表現)`
- 表現: `unprocessed` または `equalized`

## 2. 共通変換後の形式

1つの `(送信機, 受信機, 収録日)` を1つの `.npz` ファイルに加工

```text
prepared_wisig/
├── dataset_summary.json
├── manifest.csv
└── signals/
    └── equalized/
        ├── tx_000__rx_000__day_000.npz
        └── ...
```

`.npz` の中には `iq` という名前の配列が1つ入り、形は `[信号数, 256, 2]` です。

`manifest.csv` はファイルの目次です。送信機名、受信機名、日付、信号数、形、ファイルの SHA-256 を記録します。

## 3. 変換で行うこと

- 入れ子になった pickle を、グループごとの `.npz` に分ける
- ラベルを `manifest.csv` へ明記する
- 通常は `float32` へ変換する
- NaN、無限大、ゼロ信号、平均電力などを検査する
- 元ファイルと出力ファイルの SHA-256 を記録する
