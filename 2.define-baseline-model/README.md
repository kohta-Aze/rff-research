
# Open-Set RFF ベースライン比較

- 目的: WiSigで5手法を比較し、日跨ぎ研究の基準モデルを定める。
- 確定: 2026-10-06、ユーザー判断でCNN＋OpenMaxを研究用ベースラインに採用。
## 構成

| フォルダ・ファイル | 用途 |
|---|---|
| `00_common/` | [前処理](00_common/data_preprocessing/README.md)、[共通モデル・評価・テスト](00_common/modeling/README.md) |
| [01_cnn_msp/](01_cnn_msp/README.md) | 単純なsoftmax基準、OpenMax用CNNの学習 |
| [02_cnn_cosine_prototype/](02_cnn_cosine_prototype/README.md) | 同一CNN特徴での距離型基準 |
| [03_cnn_openmax/](03_cnn_openmax/README.md) | 採用ベースラインの校正・判定 |
| [04_supcon_prototype/](04_supcon_prototype/README.md) | 対照学習の比較対象 |
| [05_mtpl_evt/](05_mtpl_evt/README.md) | 同時学習とGPD判定の比較対象 |
| `2026-10-06_*.md / *.json` | 初回比較の結果・判断と数値根拠 |

- 各手法の `scripts/` は実行コード、`experiments/` は固定設定、`docs/papers/` は根拠PDF、`output/` はローカル生成物。
- 必要な実行順: データ前処理 → MSP → Cosine → OpenMax → SupCon → MTPL。コマンドは再現手順に集約。