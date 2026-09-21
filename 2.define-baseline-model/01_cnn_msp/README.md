# 01 CNN + MSP

## 一言でいうと

CNN が出した「いちばん高い確率」が十分高ければ既知端末、低ければ未知端末とする方法です。

## この手法を比較に入れる理由

構造が単純で実装しやすく、他の手法が本当に改善しているかを見る基準になります。学習は普通の既知端末分類と同じで、未知判定は学習後に追加します。

## 入力と出力（予定）

- 入力: 共通形式の I/Q 配列 `[batch, 256, 2]`
- CNN の出力: 既知端末ごとの logits `[batch, known_class_count]`
- Softmax の出力: 既知端末ごとの確率
- Open-Set score: 最大 Softmax 確率
- 最終出力: 既知端末 ID または `unknown`

## 現在の状態

- 根拠論文と説明: 作成済み
- データ前処理: `00_common/data_preprocessing` に作成済み
- 学習・評価コード: 未作成
- 実験: 未実行

次は [`docs/method_explanation.md`](docs/method_explanation.md) を読んでください。
