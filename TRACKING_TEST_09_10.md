# experiment09・10：YOLO＋ByteTrackの評価準備

> 旧Colab評価と、その前の準備手順の記録です。現在の実験A〜Gは [basketball_tracking_midterm](basketball_tracking_midterm/README.md) を使用してください。以下に登場する `evaluate_tracking_experiment.py` と `yolo_bytetrack_video.py` は整理時に削除済みで、リング付きCSVの説明も旧仕様です。現在はリング不要のボールGTを使います。experiment10はF/Gの設計にも使用したため、現在の結果は開発評価です。動画・注釈ZIP・重みはGitHubに含めません。

## experiment10のColab評価（2026-10-26追加）

experiment10のアノテーションを受領し、09と共通の評価器・ノートブックで実行する構成に対応しました。
使用ファイルと実行手順は [colab_ball_tracking/README.md](colab_ball_tracking/README.md) の冒頭を参照してください。
更新した `test_tracking_colab.ipynb` はexperiment10を初期選択します。配布ZIPは `colab_ball_tracking/ball_tracking_experiment10.zip` です。
元のダウンロードZIPも同名ですが、アノテーションのみのためColabには配布ZIPをアップロードしてください。
全2,512フレーム中1,883フレームに正解矩形があります。ラベルなし629フレームは2026-10-26にユーザーがすべて不可視と確認済みです。manifestでは `missing_policy = "not_visible"` としています。
09の実行結果フォルダを `REFERENCE_RUN` に指定すると、使用モデルと設定の一致を確認できます。

## experiment09のColab評価ファイル（2026-10-14追加）

experiment09のアノテーションを受領し、ボールだけで評価できる専用セットを作成しました。
**現在の実行手順は [colab_ball_tracking/README.md](colab_ball_tracking/README.md) を参照してください。**

- ノートブック：`colab_ball_tracking/test_tracking_colab.ipynb`
- DriveへアップロードするZIP：`colab_ball_tracking/ball_tracking_experiment09.zip`
- ZIPを置く場所：`マイドライブ/ball_tracking_test/`
- 学習済みモデル：Colabで既存のDrive上の `best.pt` を指定。

正解矩形はフレーム18〜464の447枚です。冒頭18枚はユーザー確認済みの不可視区間として扱います。
今回のセットはリングなしでIoU・TSR・CLE等を計算できます。リング必須の既存評価器は変更していません。
元のアノテーションZIPは元の場所に保持し、配布セットにはコピーを同梱しています。
以下は当初の準備手順と、リングを含む研究用評価へ拡張する場合の形式です。

## 保存先

基準フォルダは `D:\自主研究\independent_reserch` です。

```text
weights/
  ball_finetuned/
    best.pt                         ← 採用した追加学習モデル
datasets/
  tracking_test_experiment09_10/
    preparation.json                ← 動画情報・SHA-256・全フレーム読込検証
    experiment09/
      annotations/                  ← アノテーションツールの出力ZIP/XML等
      ground_truth.template.csv     ← 未記入テンプレート（そのまま評価しない）
      ground_truth.csv              ← 確認・変換後の評価用正解CSV
    experiment10/
      annotations/
      ground_truth.template.csv
      ground_truth.csv
```

元動画は `編集済み元データ/experiment09_編集.mp4` と
`編集済み元データ/experiment10_編集.mp4` にそのまま置きます。
今回の2本を学習用の `colab_ball_yolo/dataset/` に追加しません。

## best.ptを取り出す

`train_ball_colab.ipynb` は、通常Google Driveの
`マイドライブ/ball_yolo/runs/<学習実行名>/weights/best.pt` に保存します。
複数runがある場合は、検証データで採用したrunの重みを使います。

1. ノートブック末尾「モデルを取り出す」で `SELECTED_BEST` が採用したrunを指すことを確認。
2. `DOWNLOAD = True` にして、そのダウンロードセルを実行。
3. ダウンロードしたファイルを上記の `weights/ball_finetuned/best.pt` に保存。

Colabを再接続して変数が消えている場合は、Driveから直接ダウンロードできます。
取り出すために学習をやり直す必要はありません。
学習実行フォルダの `args.yaml`・`results.csv`・`run_metadata.json` も残してください。

## 正解データを作る

| 動画 | フレーム数 | 長さ | 解像度・FPS |
| --- | ---: | ---: | --- |
| experiment09 | 465 | 15.500秒 | 1280×720・30fps |
| experiment10 | 2,512 | 83.733秒 | 1280×720・30fps |

追跡の評価は連続フレームを使います。学習用画像と同じ2fpsの間引きでは、
連続失敗や復帰時間を同じ条件で測れません。元の動画・解像度・フレーム順を保持し、
全フレームを確認してください。補間を使ってラベルを作る場合も中間フレームを確認します。
編集による場面切替やシュートの区切りも記録し、境界をまたいだ復帰イベントを数えないようにします。

アノテーションツールの出力は、各動画の `annotations/` に元のZIP/XML等のまま保存してください。
これは保管場所であり、既存評価器にZIP/XMLを直接渡すことはできません。
出力形式とフレーム対応を確認し、最終的に各動画の `ground_truth.csv` へ変換して使用します。

CSVを直接記入する場合は `ground_truth.template.csv` を `ground_truth.csv` にコピーして使います。
テンプレートの `unannotated` は未確認を意味し、評価器が拒否します。空欄をボールなしとは解釈しません。

既存の `evaluate_tracking_experiment.py` に合わせるため、次を記録します。

- `frame_index`：0始まり。テンプレートの順番を保持。
- `timestamp_s`：テンプレートはフレーム番号÷30秒。動画時間であり処理時間ではありません。
- `shot_id`：各シュート・連続区間の識別子。場面切替で分けます。
- `visibility`：`visible`（可視）、`partial`（部分遮蔽）、`occluded`（完全遮蔽）、`out_of_frame`（画面外）。
- `bbox_x,y,w,h`：ボールの左上x,y・幅・高さ（元解像度のピクセル）。
  完全遮蔽・画面外では4値とも空欄にし、隠れた位置を推測で埋めません。
- `ring_bbox_x,y,w,h`：リングの矩形。同じく元解像度のピクセル。
  既存評価器では、可視・部分遮蔽フレームで必須です。固定画角でも場面切替やカメラ移動を確認します。
- `fast`、`small`：それぞれ0/1。条件別評価に使うため判定基準を事前に決めます。
- `score`：得点判定用。今回の追跡評価のみなら空欄で構いません。

ボールだけのラベルで進める場合は、リング幅で正規化するNCLEやリング付近の評価を
省略できるよう評価器の調整が必要です。リングを適当な値で埋めないでください。

## 実行・評価の流れ

モデルとアノテーションを配置した後、YOLO＋ByteTrackを動画ごとに新しく初期化して実行します。
検出クラスは追加学習モデルの `0: ball` です。入力サイズ・検出閾値・追跡設定は
検証データで決めた値を固定し、使用モデル・設定と結果を一緒に保存します。

既存の動画実行プログラムは `yolo_bytetrack_video.py` です。
そのCSVは1始まり・1フレーム複数候補で、既存評価器は0始まり・単一球の形式です。
評価前にフレーム番号と対象球の選択規則を揃える必要があります。
正解に最も近い候補を後から選ぶ方法では評価しません。

評価候補はIoU、追跡成功率TSR、中心位置誤差CLE、出力Coverage、Lost・復帰時間です。
リング正解があればNCLEとリング付近の成績も評価できます。
検出されたフレームの割合だけでは追跡精度を示せません。
この準備段階では実モデルの推論・精度評価はまだ行っていません。

## 準備の再現

```powershell
.\.venv\Scripts\python.exe .\prepare_tracking_test.py
```

既存の正解データを保護するため、出力先が存在すると停止します。
再確認用に作り直す場合は `--output` で別のフォルダを指定してください。
