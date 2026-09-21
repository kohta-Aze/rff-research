# スクリプト説明書

## `wisig_common.py`

3本の実行スクリプトが共通で使う道具箱です。

| 関数 | 入力 | 出力 | 目的 |
|---|---|---|---|
| `sha256_file` | ファイルの場所 | 64文字の文字列 | ファイルが同じ内容か確認する指紋を作る |
| `write_json` | 保存先、Python の値 | なし | 人が読める JSON を保存する |
| `audit_pickle_opcodes` | pickle の場所 | 検査結果の辞書 | pickle を開く前に想定外の命令を探す |
| `load_manyrx` | pickle の場所 | WiSig の辞書 | 検査に合格した ManyRx を読み込む |
| `dataset_labels` | WiSig の辞書 | 送信機・受信機・日・表現 | 添字と本当のラベルを対応させる |
| `representation_index` | WiSig の辞書、表現名 | 表現の添字 | equalized などを正しい配列位置へ直す |
| `iter_source_arrays` | WiSig の辞書、表現 | グループ情報と配列 | 全グループを同じ順に1つずつ渡す |
| `summarize_array` | I/Q 配列 | 品質情報の辞書 | 形、有限値、ゼロ信号、電力を調べる |

## `01_inspect_wisig.py`

### 入力

- `--input`: 公式 `ManyRx.pkl`
- `--output`: 検査結果 JSON の保存先

### 出力

- pickle 命令の検査結果
- 元ファイルの大きさと SHA-256
- 送信機、受信機、収録日の一覧
- 各表現の配列形と信号数の集計

### 目的

データを加工する前に、自分が持っているファイルが想定どおりか確認します。

## `02_prepare_wisig.py`

### 入力

- `--input`: 公式 `ManyRx.pkl`
- `--output-root`: 変換結果の保存先
- `--representation`: `equalized` または `unprocessed`
- `--dtype`: `float32` または `keep`
- `--compression`: `compressed` または `none`
- `--limit-groups`: 動作確認だけに使うグループ数。0なら全件

### 出力

- グループごとの `.npz`
- `manifest.csv`
- `dataset_summary.json`

### 目的

5手法へ全く同じ I/Q 配列とラベルを渡せるようにします。

## `03_verify_prepared_data.py`

### 入力

- `--prepared-root`: `02_prepare_wisig.py` の出力先
- `--output`: 検査結果 JSON の保存先

### 出力

- ファイル件数
- 欠損ファイル
- SHA-256 不一致
- 形・型・有限値の不一致
- `verification_passed`

### 目的

変換が途中で失敗していないことを、学習前に確認します。

## まだ作らないスクリプト

- 既知・未知端末の分割
- CNN の学習
- Open-Set のしきい値決定
- 最終評価
- グラフ作成

これらは共通ベンチマーク条件を固定した後で追加します。
