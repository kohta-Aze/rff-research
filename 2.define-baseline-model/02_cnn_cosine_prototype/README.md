# 02 CNN + Cosine Prototype

## 一言でいうと

CNN が作る特徴ベクトルと、各既知端末の代表ベクトルを比べ、向きが似ていれば既知、どれとも似ていなければ未知とする方法です。

## この手法を比較に入れる理由

Softmax の確率ではなく、特徴空間の距離で未知を判断する最小構成です。SupCon や MTPL の効果を測るとき、単純な prototype 判定との差を確認できます。

## 入力と出力（予定）

- 入力: I/Q 配列 `[batch, 256, 2]`
- CNN の出力: 特徴ベクトル `[batch, embedding_dim]`
- 学習後に作るもの: 既知端末ごとの prototype
- score: 最も近い prototype との cosine similarity
- 最終出力: 既知端末 ID または `unknown`

## 現在の状態

- 根拠論文と説明: 作成済み
- データ前処理: 作成済み
- 学習・評価コード: 未作成
- 実験: 未実行
