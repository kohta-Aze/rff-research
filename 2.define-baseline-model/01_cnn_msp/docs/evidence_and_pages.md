# 根拠資料と読むページ

## 一次資料

- `papers/Hendrycks_Gimpel_2017_MSP.pdf`
- Dan Hendrycks and Kevin Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks,” ICLR 2017.
- arXiv: https://arxiv.org/abs/1610.02136

## 該当ページ

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 1 | 学習済み分類器の確率だけで異常例を見つける考え方 | MSP を基準手法にする根拠 |
| 2 | AUROC などの評価指標 | しきい値に依存しない比較の根拠 |
| 3 | maximum softmax probability の定義と使い方 | 実装の中心 |

## 再現上の注意

この論文は画像や自然言語で評価しており、WiSig の RFF 専用モデルではありません。今回再現するのは「最大 Softmax 確率を未知判定 score にする」という方法です。論文の公開スコアを WiSig の期待値として扱いません。
