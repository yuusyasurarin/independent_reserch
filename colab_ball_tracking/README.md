# experiment09／10：Google ColabでYOLO＋ByteTrackを評価

> **旧Ultralytics版の評価手順・データ準備コードです。** 現在の実験A〜Gは [basketball_tracking_midterm](../basketball_tracking_midterm/README.md) を使用してください。`build_bundle.py` と `tracking_eval.py` は注釈の変換・確認手順を残すため公開しています。以下の「未評価」等は準備当時の記録です。experiment10はその後F/Gの開発にも使用しています。動画・注釈ZIP・生成manifest・配布ZIPはローカル保管で、GitHubには含めません。

## experiment10の実行（2026-10-26追加）

1. **このフォルダの `ball_tracking_experiment10.zip`** をDriveの `マイドライブ/ball_tracking_test/` へアップロードします。
   ダウンロードフォルダの同名ZIPはアノテーションのみです。ここで作成した配布ZIPには動画・ラベル・実行コードが入っています。
2. 更新した `test_tracking_colab.ipynb` をColabで開き、GPUランタイムで上から実行します。
   最初のセルは `EXPERIMENT = 'experiment10'` に設定済みです。
3. `BEST_PATH` にexperiment09で使用した学習済みモデルを指定します。
   09の結果が残っている場合、`REFERENCE_RUN` に `.../results/experiment09_日時` を指定すると、
   `run_config.json` から入力サイズ・閾値・FP16を引き継ぎ、モデルのSHA-256・ByteTrack設定・Ultralyticsの一致を検査します。
   空欄の場合は下記の初期値を使うため、09の実行条件と手動で照合してください。
4. 結果は `マイドライブ/ball_tracking_test/results/experiment10_日時/` に保存されます。

experiment10は1280×720、30fps、全2,512フレーム（83.733秒）です。
1,883フレームにボールの枠が1個ずつあり、残り629フレームにはラベルがありません。
ラベルなし区間の扱いは配布ZIPの `manifest.json` の `missing_policy` と `confirmation` に記録します。
2026-10-26にユーザーが629フレームすべてを確認済みの「ボールが見えない区間」と確認しました。
配布ZIPでは `missing_policy = "not_visible"` としています。未確認フレームは0です。
位置精度の分母は枠のある1,883フレームです。全フレームを順番に追跡し、不可視区間の出力数は別集計します。

原本のコピーは `../datasets/tracking_test_experiment09_10/experiment10/annotations/ball_tracking_experiment10.zip`、
配布内容の展開状態は `bundles/experiment10/` にあります。09のZIP・元データは保持しています。
09を再実行する場合は `EXPERIMENT = 'experiment09'` に変更します。以下の09の手順・指標の定義も共通です。

準備時に動画全2,512フレームの読込、ラベル・SHA-256・配布ZIPの展開検査、評価器の6テスト、
09の設定引き継ぎと重み不一致時の停止を確認しました。学習済みモデルによるexperiment10の推論・精度測定は未実行です。

## 使うファイルは2つ

1. `ball_tracking_experiment09.zip`（約19MB）：動画・アノテーション・実行コードを同梱。
2. `test_tracking_colab.ipynb`：Colabで開くノートブック。

学習済みの `best.pt` はGoogle Driveから読み込みます。ZIPには含めていません。
このノートブックは追加学習を行いません。09と10で同じ学習済みモデルを使用してください。

## 実行手順

1. Google Driveのマイドライブ直下に **`ball_tracking_test`** フォルダを作り、
   **`ball_tracking_experiment09.zip` をその中へアップロード**します。ZIPをDrive上で展開する必要はありません。
2. [Google Colab](https://colab.research.google.com/) の「ファイル → ノートブックをアップロード」で
   `test_tracking_colab.ipynb` を開きます。
3. 「ランタイム → ランタイムのタイプを変更」でGPUを選び、上からセルを実行します。
   09の場合は最初のセルを `EXPERIMENT = 'experiment09'` に変更します。
4. Driveへの接続を許可します。ZIPはColabの一時領域へ展開されます。
5. モデル設定セルの **`BEST_PATH`** に採用した追加学習モデルのパスを書きます。例：

   ```python
   BEST_PATH = "/content/drive/MyDrive/ball_yolo/runs/ball_実際の学習実行名/weights/best.pt"
   ```

   `BEST_PATH = ""` の場合、既存の `ball_yolo/runs/` 等から候補を一覧表示します。
   候補が1つなら使い、複数なら指定するまで停止します。最新runを勝手に選びません。
   別途ダウンロード済みなら `MyDrive/ball_tracking_test/best.pt` にアップロードしても使えます。
6. `IMGSZ`、`CONF`、追跡設定が検証時に決めた条件と一致することを確認して実行します。
   初期値は既存追跡コードに合わせた `IMGSZ=960, CONF=0.10, NMS_IOU=0.5` で、最適値を保証するものではありません。
   ByteTrackの設定は同梱 `bytetrack.yaml` に固定しています。
7. 評価表・グラフ・動画を確認します。結果は自動的にDriveへ保存されます。

ランタイム再接続後も上から実行できます。再実行時には新しい結果フォルダを作ります。
推論にGPUが使えない場合、GPUランタイムへ切り替えてください。
モデル選択に使うのは検証データです。今回のtest成績を見て再調整した場合は、独立した最終testとは区別してください。

## 保存される結果

保存先：`マイドライブ/ball_tracking_test/results/experiment09_日時/`

| ファイル | 内容 |
| --- | --- |
| `metrics.json` / `summary.csv` | 評価指標・処理速度・評価上の制約 |
| `per_frame.csv` | 全465フレームの正解・選択した追跡結果・IoU・位置誤差 |
| `tracked.mp4` | 緑＝正解枠、オレンジ＝採用した追跡枠 |
| `tracked_browser.mp4` | Colab表示用H.264動画（ffmpeg利用時） |
| `candidates.jsonl` | 全候補、ID未確定候補、採用候補、処理時間 |
| `predictions.json` | 全フレームの追跡出力 |
| `run_config.json` | 重み・動画・注釈のSHA-256、実行設定、GPU・ライブラリ情報 |
| `bytetrack.yaml` / `environment.txt` | 追跡設定と実行環境 |

結果ZIPも作れます。動画がColab内で表示できない場合は、Drive上のMP4をダウンロードしてください。

## 正解データの取り扱い

元ZIP `ball_only_アノテーション_test09.zip` は移動・変更していません。
コピーをこのパッケージの `data/annotations09.zip` に保存しました。
元ZIP内の `labels/train` と `train.txt` はエクスポート時の名称です。
**今回の465フレームはすべてtest専用**として扱い、学習やモデル選択に使いません。

- 元動画：1280×720、30fps、465フレーム、15.5秒。0始まりの番号で対応。
- フレーム18〜464：447フレーム、各フレームにballの正解矩形1個。
- フレーム0〜17：ユーザー確認済みの「ボール不可視」。位置の正解は作りません。
- 不可視が完全遮蔽か画面外かはラベルから区別できません。
- 代表フレームの正解枠は `annotation_preview.jpg` で確認できます。

このZIPにはリング、シュート区切り、遮蔽属性、対象IDが含まれていません。
そのため既存のリング必須評価器を変更せず、今回専用の `tracking_eval.py` を用意しています。
`data/ground_truth_ball.csv` はボール専用形式で、既存の `ground_truth.template.csv` とは別形式です。

## 指標の読み方

- **TSR（IoU≥0.5）**：正解矩形がある447フレームのうち、追跡矩形のIoUが0.5以上の割合。
  IoU≥0.25の割合も補助的に出力します。
- **平均IoU**：同じ447フレームの平均。追跡出力がないフレームも0として含めます。
- **CLE**：ボール中心の距離誤差（ピクセル）。出力のあるフレームだけで計算し、件数・平均・中央値・95パーセンタイルを併記します。
- **Coverage**：正解矩形があるフレームで、追跡出力が得られた割合。位置が正しい割合とは異なります。
- **不可視区間の出力数**：冒頭18フレームで追跡枠が出た回数。隠れた球の正しい位置を確認できないため、TP/FPと断定しません。
- **連続失敗**：正解矩形のある連続区間でIoU閾値未達が3フレーム以上続いた回数と、最長連続失敗長。
  シュート単位のLost・復帰率ではありません。
- **ID数・隣接ID変更数**：診断用の値。IDF1・HOTA・標準的なIDスイッチ数ではありません。
- **処理速度**：ウォームアップ後の追跡処理時間と、描画・保存込みの速度。
  ColabのGPUとDrive入出力に依存し、カメラの撮影から出力までの遅延ではありません。

同一フレームに複数候補が出た場合は、直前まで採用していたIDがあれば維持し、
なければ信頼度が最も高いID確定候補を採用します。ID未確定の検出枠は追跡成功に数えません。
**候補の選択に正解データを使いません。**

リング幅で正規化するNCLE、リング付近の評価、部分遮蔽別評価、シュート単位の復帰率、
MOT指標は今回のラベルから算出しません。必要な追加ラベルを用意して別途評価します。

実装は [Ultralyticsの追跡API](https://docs.ultralytics.com/modes/track/) に従い、
`tracker` を明示してフレーム間で状態を保持します。Ultralyticsは学習環境と同じ8.4.121に固定しています。

## ローカル検証・再作成

プロジェクト直下から：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s colab_ball_tracking -p "test_*.py" -v
.\.venv\Scripts\python.exe .\colab_ball_tracking\make_notebook.py
.\.venv\Scripts\python.exe .\colab_ball_tracking\build_bundle.py
```

experiment10の再作成：

```powershell
.\.venv\Scripts\python.exe .\colab_ball_tracking\build_bundle.py --experiment experiment10 --annotations .\datasets\tracking_test_experiment09_10\experiment10\annotations\ball_tracking_experiment10.zip --confirm-not-visible
```

上記は今回確認されたアノテーションの再作成用です。別のアノテーションへ差し替える場合は、確認内容を引き継がず再確認してください。
experiment10の生成では09のZIP・manifest・正解CSVを上書きしません。

学習済みモデルがまだローカルにないため、追加学習モデルでの精度の実測値はここに記載していません。

準備時に評価ロジックの6テストが通過し、既存の汎用YOLOモデルで全465フレームの
推論・追跡・動画/CSV/JSON保存を確認しました。配布ZIPを別フォルダへ展開した実行も確認済みです。
これらはローカル環境での動作確認であり、Colab上での実行や追加学習モデルの精度検証はまだ行っていません。
