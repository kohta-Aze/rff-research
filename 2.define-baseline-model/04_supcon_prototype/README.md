# 04 SupCon + Prototype

## 一言でいうと

同じ端末の信号特徴を近づけ、別の端末を遠ざける Supervised Contrastive Learning で CNN を学習し、最後は prototype との距離で未知を判定します。

## この手法を比較に入れる理由

単純な分類学習よりも、距離で比較しやすい特徴空間を直接作る方法です。CNN + Cosine Prototype との差から、特徴学習方法の効果を調べられます。

## 入力と出力（予定）

- 学習入力: 同じ信号から作る2つの view と端末ラベル
- encoder 出力: 特徴ベクトル
- projection head 出力: SupCon loss 用ベクトル
- 学習後の出力: 既知端末 prototype
- 推論出力: 既知端末 ID または `unknown`

## 現在の状態

- 根拠論文2本と説明: 作成済み
- データ前処理: 作成済み
- データ拡張・学習・評価コード: 未作成
- 実験: 未実行
