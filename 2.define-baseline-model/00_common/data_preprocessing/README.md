# WiSig ManyRx 前処理

## ここで行うこと

大きな `ManyRx.pkl` をいきなりモデルへ渡さず、次の3段階で安全に準備します。

```text
ManyRx.pkl
   │
   ├─ 01_inspect_wisig.py       中身の構造を調べる
   │
   ├─ 02_prepare_wisig.py       共通の .npz 形式へ変換する
   │
   └─ 03_verify_prepared_data.py 変換後の欠損・破損を調べる
```

元の `ManyRx.pkl` は変更しません。

## 必要なもの

- Python 3.10 以上
- NumPy
- WiSig ManyRx の `ManyRx.pkl`
- 元 pickle を読み込むため、おおむね数 GB 以上の空きメモリ
- 変換結果を置くための十分な空き容量

環境の例:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 実行順

### 1. 元データを調べる

```bash
python scripts/01_inspect_wisig.py \
  --input /path/to/ManyRx.pkl \
  --output output/inspection.json
```

ここでは pickle 内で使われる命令を先に検査し、想定外の Python オブジェクトが含まれる場合は読み込みを止めます。その後、送信機数、受信機数、収録日、配列の形、信号数などを JSON へ保存します。

### 2. 共通形式へ変換する

```bash
python scripts/02_prepare_wisig.py \
  --input /path/to/ManyRx.pkl \
  --output-root /path/to/prepared_wisig \
  --representation equalized \
  --dtype float32
```

`equalized` は通信路の影響を一部取り除いた表現です。最初の比較でどちらを使うかは、ベンチマーク条件として実験前に固定します。

### 3. 変換結果を検査する

```bash
python scripts/03_verify_prepared_data.py \
  --prepared-root /path/to/prepared_wisig \
  --output output/verification.json
```

この処理は `manifest.csv` に書かれた全ファイルを開き、ファイルの SHA-256、配列の形、NaN・無限大、ラベルとファイル名の対応を調べます。

## 小さく動作確認する方法

本番変換の前に、先頭の数グループだけ出力できます。

```bash
python scripts/02_prepare_wisig.py \
  --input /path/to/ManyRx.pkl \
  --output-root /tmp/wisig_small_check \
  --representation equalized \
  --limit-groups 3
```

これはコードの動作確認用です。この限定出力で精度実験は行いません。

## 出力について

`output/` は検査レポートを置く例の場所です。大きな変換済みデータは Git 管理外のデータ保存場所へ出してください。

詳しい説明は次を読んでください。

- [`docs/preprocessing_guide.md`](docs/preprocessing_guide.md)
- [`docs/script_reference.md`](docs/script_reference.md)
