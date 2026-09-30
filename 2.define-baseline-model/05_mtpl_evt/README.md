# 05 MTPL + EVT

## 一言でいうと

分類、信号の復元、端末 prototype への集約を同時に学習し、既知端末 prototype からの距離の外れ側を EVT でモデル化して未知を判定します。

## この手法を比較に入れる理由

RFF の Open-Set を直接対象にした新しい一次研究で、原論文では比較手法より高い平均 AUROC が報告されています。ただし原論文と今回ではデータ条件が違うため、WiSig の共通条件で再測定してからベースライン候補か判断します。

## 入力と出力（予定）

- 入力: I/Q 配列 `[batch, 256, 2]`
- encoder 出力: 特徴ベクトル
- decoder 出力: 復元 I/Q
- classifier 出力: 既知端末 logits
- 学習する prototype: 既知端末ごとの特徴中心
- EVT fitting 出力: GPD の threshold、shape、scale
- 推論出力: 既知端末 ID または `unknown`

## 現在の状態

- MTPL 論文のコピー、該当ページ、説明: 作成済み
- データ前処理: 作成済み
- MTPL 学習・GPD fitting・評価コード: 未作成
- 実験: 未実行
