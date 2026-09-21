# 03 CNN + OpenMax

## 一言でいうと

CNN の最終付近の特徴について、既知クラスの中心から遠い「外れ側」を Weibull 分布で学び、未知クラスの確率を作る方法です。

## この手法を比較に入れる理由

OpenMax は深層学習の代表的な Open-Set 手法です。単純な MSP と、prototype + EVT を使う MTPL の間に置く比較対象になります。

## 入力と出力（予定）

- 入力: I/Q 配列 `[batch, 256, 2]`
- CNN の出力: 既知端末ごとの activation vector
- calibration の入力: 正しく分類できた training または calibration 特徴
- calibration の出力: クラス中心と Weibull パラメータ
- 推論の出力: 既知端末確率と unknown 確率

## 現在の状態

- 根拠論文と説明: 作成済み
- データ前処理: 作成済み
- 学習・評価コード: 未作成
- 実験: 未実行
