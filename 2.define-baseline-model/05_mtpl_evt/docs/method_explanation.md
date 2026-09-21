# MTPL + EVT のやさしい説明

## MTPL とは

Multi-Task Prototype Learning の略です。1つの encoder に、3つの宿題を同時にさせます。

1. **分類**: 既知端末を正しく見分ける。
2. **復元**: 小さくした特徴から元の信号を作り直す。
3. **prototype 学習**: 同じ端末の特徴を、その端末の代表点へ近づける。

分類だけでは正解名を当てるために都合のよい特徴だけを残す可能性があります。復元もさせることで信号の情報を残し、prototype loss で同じ端末を小さなかたまりにする考えです。

## 学習時の入力と出力

- 入力: 既知端末の I/Q 信号と端末ラベル
- encoder 出力: 特徴 `z`
- decoder 出力: 復元信号 `x_hat`
- classifier 出力: 既知端末 logits
- prototype: 学習可能な端末ごとの中心

全体 loss は次の3つの和です。

```text
total loss = classification loss
           + λr × reconstruction loss
           + λp × prototype loss
```

`λr` と `λp` は宿題ごとの重みです。test の結果を見て選ばず、validation で決めます。

## EVT と GPD は何か

EVT は Extreme Value Theory、極端な値を扱う統計の考え方です。GPD は Generalized Pareto Distribution で、端の大きな値の形を表す分布です。

MTPL では、既知の学習信号と正解 prototype の距離を集めます。その中で遠い上位部分を GPD へ当てはめます。新しい信号が既知としては極端に遠いかを確率で調べます。

## EVT fitting の入力と出力

- 入力: 学習済み encoder、prototype、既知 training 信号
- 中間値: 正解 prototype までの二乗 Euclidean 距離
- tail: 距離の上位部分。原論文は 0.9 quantile を例示
- 出力: threshold `ω`、GPD shape `ξ`、scale `σ`

## 推論の手順

1. 新しい信号を encoder へ入れる。
2. 全 prototype までの距離を計算する。
3. 最小距離の prototype を既知端末候補にする。
4. 最小距離が通常範囲なら候補端末を出す。
5. tail に入る場合は GPD の p-value を計算する。
6. p-value が significance level より小さければ `unknown` にする。

## 限界

- encoder、decoder、classifier、prototype を同時に学習するため、他手法より複雑です。
- loss 重み、tail threshold、significance level で結果が変わります。
- 原論文のデータは16台の Wi-Fi デバイスで、WiSig ManyRx と同じ条件ではありません。
- 原論文の高い AUROC が、受信機や日をまたぐ厳しい条件でも再現する保証はありません。
- GPD fitting に使う距離へ評価データを混ぜるとリークになります。
