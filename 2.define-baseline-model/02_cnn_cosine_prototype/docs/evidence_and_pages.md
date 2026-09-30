# 根拠資料と読むページ

## 一次資料

- `papers/Wang_et_al_2023_Improved_Prototype_Learning.pdf`
- Weidong Wang, Hongshu Liao, Lu Gan, “Open-Set RF Fingerprinting via Improved Prototype Learning,” 2023.
- arXiv: https://arxiv.org/abs/2306.13895

## 該当ページ

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 1 | RFF の Open-Set に prototype learning を使う問題設定 | RFF で prototype を比較する根拠 |
| 2 | prototype 特徴空間と、クラス内をまとめる考え方 | 端末の代表ベクトルを使う理由 |
| 2–3 | consistency regularization と online label smoothing | より高度な prototype 法との違い |
| 4 | OpenMax 等との実験比較 | 発展手法を比較候補にする根拠 |

## 再現上の注意

このディレクトリの手法は、比較のために単純化した **CNN + cosine prototype** です。論文の Improved Prototype Learning にある consistency regularization と online label smoothing は入れません。ここでそれらを入れると、単純な距離判定の基準ではなくなるためです。

したがって、この論文は RFF における prototype 学習の根拠資料であり、このディレクトリを論文完全再現と呼びません。
