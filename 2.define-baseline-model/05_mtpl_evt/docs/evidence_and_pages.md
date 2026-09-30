# 根拠資料と読むページ

## 一次資料

- `papers/Ma_et_al_2025_MTPL.pdf`
- Zhao Ma, Shengliang Fang, Youchen Fan, “Open-Set Radio Frequency Fingerprint Identification Method Based on Multi-Task Prototype Learning,” Sensors 2025, 25(17), 5415.
- DOI: https://doi.org/10.3390/s25175415
- PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC12431036/

## 該当ページ

PDF のページ番号で示します。

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 1 | 目的、3タスク、mean AUROC 0.9918 の要約 | 候補に入れる根拠。ただし公開値は参考値 |
| 6 | 4.1 Method Overview、encoder / decoder / classifier、GPD の全体像 | 構造の根拠 |
| 7–8 | 4.2、式 (2)–(5)、3種類の loss、Algorithm 1 | 学習実装の根拠 |
| 9–10 | GPD の説明、Algorithm 2 の開始 | offline EVT fitting の根拠 |
| 11–12 | Algorithm 2 続き、Algorithm 3 | online unknown 判定の根拠 |
| 13 | 5.2、複素 CNN 構造、実験設定 | 再現条件の確認 |
| 14–16 | 比較方法、Table 3、AUROC | 公開結果と限界の確認 |

## 原論文の公開値をどう扱うか

原論文は平均 AUROC 0.9918 を報告しています。しかし、この値は原論文のデータ、既知10台・未知6台、ネットワーク、分割、調整条件で得られたものです。

今回の WiSig 共通ベンチマークで同じ値が出るとは考えません。5手法を同じデータと分割で実行し、その実測値で順位を決めます。

## 実装時に確認する点

本文の total loss の説明と記号の並びは慎重に照合します。原論文どおりの複素 CNN を使う版と、5手法で CNN 骨格をそろえる比較版を混同せず、別実験として記録します。
