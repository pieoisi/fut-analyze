# fut-analyze

発話データの試行単位openSMILE特徴量抽出と、小規模データ向けの探索的モデル評価を行います。

## 実データの置き場所

入力フォルダーは[`analysis/data/`](analysis/data/)です（別名の`input/`フォルダーはありません）。CSVとWAVをここに置いてください。各セッションの2ファイルは同じベース名にします（例：`P02_20261115_1020.csv`と`P02_20261115_1020.wav`）。フォルダー直下に複数セッションを一緒に置けます。

音声と参加者情報は機微なデータです。`analysis/data/`内のファイルはGitの追跡対象外です。参加者の同意と保管方針を確認し、このフォルダーを公開したり、データファイルをGitへ追加したりしないでください。

## 解析の実行

```bash
python -m pip install -r requirements.txt
python analysis/analyze.py analysis/data analysis/output
```

Python 3.10以降が必要です。

最初の引数が入力フォルダー、2つ目が出力フォルダーです。出力フォルダーは[`analysis/output/`](analysis/output/)として用意しています。2つ目を省略すると入力内に`analysis_output/`を作ります。出力先には試行ごとの特徴量、除外理由、話者単位交差検証の評価指標と予測値が作られます。出力には参加者IDや音響特徴量が含まれるため、公開リポジトリへ追加しないでください。

評価したいラベルを選ぶ場合は`--targets`を指定します。選択肢は`ease`、`valence`、`arousal`、`affect_9class`、`stuck`です。省略すると全タスクを処理します。

```bash
# 言葉の出やすさだけ
python analysis/analyze.py analysis/data analysis/output --targets ease

# valenceとarousal
python analysis/analyze.py analysis/data analysis/output --targets valence arousal
```

`trial_features.csv`が特徴量とラベル、`evaluation.json`が評価結果、`predictions.csv`が参加者単位の予測、`exclusions.csv`が特徴抽出できなかった試行と理由です。各ラベルクラスが2名以上の参加者にない場合は、そのタスクの性能評価を行いません。約30試行では特に9カテゴリ分類の結果は探索的なものとして扱ってください。

詳細な収集仕様、ラベルの意味、評価上の制約は[opensmile-data.md](opensmile-data.md)を参照してください。
