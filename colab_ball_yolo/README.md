# ball検出のYOLO追加学習：Colab用セット

> **旧Ultralytics版の記録・補助コードです。** 現在のYOLOv5s/mの追加学習は [中間発表用の手順](../basketball_tracking_midterm/README.md) に従ってください。本フォルダの `prepare_dataset.py` は現在も参照するデータセットの作成・検証コードです。以下の学習手順は旧版用で、YOLOv5版の重み・設定とは交換できません。画像・ラベル・ZIP・プレビューはローカル保管のため、cloneだけでは実行できません。

準備日：2026-10-09。学習そのものはまだ実行していません。

## 最初に行うこと

1. **colab_ball_yoloフォルダを丸ごとマイドライブ直下にコピー**します。中に **ball_yolo_colab.zip** があることを確認してください。ZIPだけを **マイドライブ / ball_yolo /** またはマイドライブ直下にアップロードする方法にも対応します。ZIPはDrive上で展開しません。
2. Google Colab（https://colab.research.google.com/）を開き、「ファイル → ノートブックをアップロード」で **train_ball_colab.ipynb** を開きます。
3. 「ランタイム → ランタイムのタイプを変更」でGPUを選択します。
4. 上からセルを実行し、Driveへの接続を許可します。最初の学習で事前学習済み重みがダウンロードされます。
5. Driveの `ball_yolo/runs/ball_日時/` に結果が保存されます。採用する重みは `weights/best.pt` です。

ZIPとノートブックの2ファイルだけでColab作業を始められます。元のCVAT ZIPや画像を追加でアップロードする必要はありません。
GPUメモリ不足の場合はBATCHを8から4または2に下げ、新しいrunとして学習してください。
インターネット接続はライブラリ・事前学習済み重みの取得に必要です。

## ZIPが見つからない場合

Driveに `colab_ball_yolo` フォルダをコピーした場合、最初のコードセルのZIP_PATHの行を次に変更して、そのセルを再実行してください。再アップロードは不要です。

```python
ZIP_PATH = Path("/content/drive/MyDrive/colab_ball_yolo/ball_yolo_colab.zip")
```

最新版は `colab_ball_yolo/`、`ball_yolo/`、マイドライブ直下の順にZIPを探し、使用する場所を表示します。複数の場所に別バージョンのZIPがある場合は、ZIP_PATHに使用するファイルを明示してください。
保存先のRUNSは読み込み元とは別の設定なので、変更する必要はありません。
`Drive already mounted` は接続済みという案内です。このZIPパスのエラーを直すために再マウントする必要はありません。

## データの内訳

| 用途 | 画像 | ボールあり | ボールなし | 矩形 |
|---|---:|---:|---:|---:|
| train（学習） | 300 | 187 | 113 | 187 |
| val（検証・調整、元のeval） | 50 | 30 | 20 | 30 |
| test（設定確定後の最終評価） | 58 | 30 | 28 | 30 |
| 合計 | 408 | 247 | 161 | 247 |

- クラスは **0: ball** の1種類。画像は1280×720。
- 元の3つのCVATアノテーションZIPはすべて内部が `labels/train` でした。ZIPファイル名と画像名・元manifestを照合して正しい用途へ割り当てました。
- アノテーションは既にUltralytics YOLOの矩形形式でした。座標形式の変更は不要です。
- ラベルファイルがない161枚は「すべて確認済みでボールなし」とユーザーに確認済みです。空のtxtラベルを明示的に作成しています。
- 元画像には `datasets/cvat_experiment11_12/cvat_upload/{train,eval,test}.zip` を使用し、元manifestのSHA-256と一致することを検証しています。
- 元の分割を保持しています。各動画の時間帯で70/15/15%に分け、境界の前後2秒を除いて抽出した分割です。画像の再ランダム分割はしません。
- 同一撮影動画内の評価なので、別試合・別カメラへの汎化を示す評価ではありません。類似プレーやリプレイの重複はハッシュ検査だけでは検出できません。

## ファイル

- `train_ball_colab.ipynb`：日本語の実行手順。学習・検証・任意の再開・最終test・重みダウンロード。
- `ball_yolo_colab.zip`：Colabへアップロードする全データ・コードのセット。
- `dataset/images/{train,val,test}/`：画像。
- `dataset/labels/{train,val,test}/`：同名のYOLOラベル。
- `dataset/data.yaml`：1クラスのデータ設定。学習スクリプトが環境に合わせて絶対パスの `data.colab.yaml` を生成します。
- `dataset/manifest.csv`：画像・ラベルのSHA-256、元動画・フレーム・時刻。
- `dataset/missing_labels.csv`：空ラベルを補った161枚の一覧。
- `dataset/validation_report.json`：分割ごとの検証結果・元ZIPのハッシュ。
- `dataset/provenance/`：元の画像抽出情報とmanifestのコピー。
- `preview_train.jpg`, `preview_val.jpg`：正解矩形を重ねた確認画像。各12枚（陽性9・背景3）。
- `image_quality_report.json`：全画像のデコード・寸法検査と矩形サイズの範囲。
- `prepare_dataset.py`：元CVAT ZIPからの再生成・検証。Python標準ライブラリのみ。
- `train_yolo.py`：追加学習・val/test評価用CLI。
- `requirements-colab.txt`：既存プロジェクトと同じUltralytics 8.4.121に固定。PyTorchはColabのGPU対応版を使用。
- `build_bundle.py`：画像検査・プレビュー・ZIP作成（Pillowが必要）。
- `test_prepare_dataset.py`：不正クラス・座標・重複矩形・パストラバーサル・改変ファイルなどの検査テスト。
- `bundle_sha256.txt`：完成ZIPのチェックサム。ZIP外に配置。

## 学習・調整の方針

初期値は既存プログラムに合わせた `yolo26n.pt` の事前学習重み、100 epochs、patience 20、imgsz 960、batch 8、AdamW、lr0 0.001です。
小さいボールの解像度を確保するため960を初期値にしています。最適値を検証済みという意味ではありません。
上下反転・mixupは無効、左右反転と控えめなmosaicを使用します。

1. 初期値で1回追加学習し、valのmAP50-95、Precision、Recall、予測画像を記録します。
2. 必要ならimgszを640/960/1280、lr0を0.0005/0.001で一つずつ変更して比較します。変更ごとに新しいrun名で実行します。
3. 採用するモデル・入力サイズ・推論時の信頼度閾値はvalで決めます。背景画像での誤検出とボールの見逃しを両方確認します。
4. 設定確定後にノートブックの `RUN_FINAL_TEST=True` にして最終testを実行します。
5. testを見て設定を再調整した場合、そのtestは独立した最終評価として扱えなくなります。

検証画像は50枚なので小さな指標差には注意し、必要なら同一条件でseedを変えて再現性を確認してください。
YOLOのmAP計算用評価はconf=0.001で行います。これは実際の追跡に使う検出閾値とは別です。
学習時のbest.ptはUltralyticsの検証fitnessによって保存されます。最終testで重みを選択しません。
大量の自動探索は初期設定に含めていません。

## 保存と再開

Driveのrunフォルダには `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, 図表が保存されます。
正常終了時には `run_metadata.json` と `environment.txt`（pip freeze）も保存します。
学習途中の切断では、まずDriveにlast.ptが存在することを確認し、ノートブックの再開セルを使ってください。
最低1 epoch完了前には再開可能な重みがまだない場合があります。完了済み学習の延長にはresumeではなくbest.ptから新しい学習を開始します。

## ローカルの既存追跡プログラムで使う

この節は旧構成の記録です。以下の `yolo_basketball_tracker.py` は削除済みのため、現在の実行手順としては使用しないでください。

ダウンロードしたbest.ptをプロジェクトの `weights/ball_best.pt` にコピーしてください。
`yolo_basketball_tracker.py` の起動引数に次を指定します（動画などの既存引数はそのまま）：

```text
--model weights/ball_best.pt --class-id 0
```

今回のモデルはballだけを検出します。player・ringの検出が必要な実験構成では、この重みだけではそれらを検出できません。
COCOのsports ball番号32を指定しないでください。

## 再生成と検証（ローカル）

`independent_reserch` を作業ディレクトリにして、Pythonが動く環境で実行します。

```powershell
python colab_ball_yolo/prepare_dataset.py --missing-labels background --output colab_ball_yolo/dataset_v2
python colab_ball_yolo/prepare_dataset.py --verify colab_ball_yolo/dataset
python -m unittest discover -s colab_ball_yolo -p "test_*.py" -v
```

`--missing-labels background` は全ての欠落ラベルが確認済み背景である場合のみ指定します。
今後未確認画像が含まれる場合は `--missing-labels exclude` にします。既定のerrorは欠落時に停止します。
再生成処理は出力先が存在すると停止し、既存データを上書きしません。
元の画像・アノテーションZIPは変更していません。

ZIPを作り直す場合は、完成ZIPを別名で保管してから `python colab_ball_yolo/build_bundle.py` を実行します。
この処理は `colab_ball_yolo/dataset` を対象とします。再生成した別フォルダを採用する場合は、旧datasetを保管したうえで配置を入れ替えてください。

## 公式資料

- 学習APIと再開：https://docs.ultralytics.com/modes/train/
- 検出用データ形式：https://docs.ultralytics.com/datasets/detect/
- YOLO26：https://docs.ultralytics.com/models/yolo26/

形式とAPIは公式資料で確認しています。ColabのGPUでの学習・精度評価は利用者がこれから実行する工程です。
