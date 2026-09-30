# 00_common

5手法で同じものを使うための場所です。

## なぜ共通化するのか

最善の手法を決めるために入力データを一致させ、対照実験を行うため

## 今回対象とするデータ

主データ： **WiSig ManyRx**.  
選定理由ManyRx：複数の送信機、受信機、収録日を含み、OpenSet認証に求められる性能を検証するのに最適だから。

## 内容

- `docs/data_contract.md`: 全手法へ渡すデータの約束
- `docs/benchmark_plan_draft.md`: 後で固定する比較条件の下書き
- `docs/evidence_and_pages.md`: WiSig 論文の読む場所
- `docs/papers/`: WiSig 一次資料のコピー
- `data_preprocessing/`: データ確認・変換・再検査の手順とコード
