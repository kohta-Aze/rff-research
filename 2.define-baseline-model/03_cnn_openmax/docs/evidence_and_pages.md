# 根拠資料と読むページ

## 一次資料

- `papers/Bendale_Boult_2016_OpenMax.pdf`
- Abhijit Bendale and Terrance E. Boult, “Towards Open Set Deep Networks,” CVPR 2016.
- 公式公開ページ: https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html

## 該当ページ

| PDFページ | 内容 | 今回との関係 |
|---:|---|---|
| 1–3 | OpenMax の目的、activation vector で未知を扱う考え方 | 全体像 |
| 4 | Algorithm 1、mean activation vector とクラス別 Weibull fit | calibration の実装根拠 |
| 5 | OpenMax の activation 再配分と unknown class | 推論の実装根拠 |
| 7 | 距離、tail size、testing phase | ハイパーパラメータ記録の根拠 |

## 再現上の注意

論文は ImageNet を使っています。今回の CNN 骨格と I/Q 入力は RFF 用に置き換えますが、mean activation、tail fitting、unknown activation の考え方は原論文に合わせます。公開スコアは WiSig の比較値として使いません。
