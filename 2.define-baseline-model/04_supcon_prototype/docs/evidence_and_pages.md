# 根拠資料と読むページ

## 一次資料1: Supervised Contrastive Learning

- `papers/Khosla_et_al_2020_Supervised_Contrastive_Learning.pdf`
- Prannay Khosla et al., “Supervised Contrastive Learning,” NeurIPS 2020.
- 公式: https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 2 | 同じクラスを positive、他クラスを negative にする図 | 直感的な根拠 |
| 4 | encoder、projection network、2 view の学習手順 | モデル構成の根拠 |
| 5 | supervised contrastive loss の式 (2)、(3) | loss 実装の根拠 |
| 6 | triplet loss や N-pairs loss との関係 | 方法の位置づけ |

## 一次資料2: RFF の Prototype Learning

- `papers/Wang_et_al_2023_Improved_Prototype_Learning.pdf`
- Weidong Wang, Hongshu Liao, Lu Gan, “Open-Set RF Fingerprinting via Improved Prototype Learning,” 2023.
- arXiv: https://arxiv.org/abs/2306.13895

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 1–2 | Open-Set RFF と prototype 特徴空間 | RFF で prototype を使う根拠 |
| 2 | augmentation と consistency の考え方 | RFF の view 設計を考える材料 |
| 4 | 他の Open-Set 手法との比較 | 比較候補としての根拠 |

## 再現上の注意

SupCon 原論文は画像分類です。RFF 向けの view は原論文の画像拡張をそのまま使いません。また、この方法は2本の考え方を組み合わせる比較手法であり、どちらかの論文の完全再現ではありません。
