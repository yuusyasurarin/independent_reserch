# バスケットボールのボール追跡研究

現在の実験・追加学習・評価は **[basketball_tracking_midterm](basketball_tracking_midterm/README.md)** に集約しています。YOLOv5s/mの追加学習済みモデルを使い、毎フレーム検出、間欠検出、KCF、LK、固定遅延による未来情報の利用を比較します。

2026-10-29に構成を整理しました。旧KCF単独版、旧リアルタイム追跡版、旧YOLO/ByteTrack版、M1〜M6比較コードとその説明資料は削除しました。現在の実験で使うKCF・ByteTrack・LKの実装は `basketball_tracking_midterm/src/` に残っています。削除前のコードはGit履歴から参照できます。

## 研究の前提

- 一般的な体育館で、持ち運べるPCを使って実行できるシステムを目指します。
- 主な対象は、中学生・高校生・大学生などのアマチュアのバスケットボールです。

現在はボール追跡の精度と計算負荷を検証しています。実測の主な環境はRTX 3050 Ti Laptop GPU搭載PCですが、体育館での長時間運用や別の撮影条件への汎化は今後の検証対象です。

## 最初に読む資料

| 目的 | 資料 |
|---|---|
| 学習からWindowsでの評価まで | [実験全体のREADME](basketball_tracking_midterm/README.md) |
| 追加学習済みYOLOの条件Bを動かす | [条件Bの手順](basketball_tracking_midterm/EVALUATION_B.md) |
| 修正済みE・GとB・Dを比較する | [2026-10-28の比較結果・方法・動画一覧](basketball_tracking_midterm/reports/2026-10-28-fixed-lag-comparison/README.md) |
| A〜Eの初回測定 | [集計表](basketball_tracking_midterm/reports/2026-10-27/comparison.csv)（旧Eは初期化不具合の修正前） |
| 未来情報を使わないLKを含む条件F | [条件Fの開発実験](basketball_tracking_midterm/reports/2026-10-27-condition-f/README.md) |
| 固定遅延の方式・検出頻度・入力解像度の検討 | [条件Gの開発実験](basketball_tracking_midterm/reports/2026-10-27-condition-g/README.md) |
| KCF・LKの成功と失敗 | [実動画による説明](basketball_tracking_midterm/reports/2026-10-28-tracking-explanation/README.md) |
| カメラ入力を固定遅延3秒でリアルタイム追跡する | [リアルタイム固定遅延追跡の説明](basketball_tracking_midterm/reports/2026-11-16-realtime-fixed-delay/README.md) |
| ソフトウェアの出典とライセンス | [THIRD_PARTY.md](basketball_tracking_midterm/THIRD_PARTY.md) |
| 残したもの・削除したもの | [構成整理の記録](docs/repository_cleanup_20261029.md) |

## 現在の実験条件

| 条件 | 処理 | 未来情報 |
|---|---|---|
| A | 未調整のCOCO YOLOv5でsports ballを毎フレーム検出 | なし |
| B | 追加学習済みYOLOv5で毎フレーム検出 | なし |
| C | 追加学習済みYOLOv5＋ByteTrack | なし |
| D | 3フレームごとのYOLO検出＋間の位置保持 | なし |
| E | 3フレームごとのYOLO検出＋前向きKCF | なし。固定遅延比較では逆向きKCFを追加 |
| F | 適応的なYOLO再検出＋局所LK＋速度予測・候補選択 | なし |
| G | F相当の前向き処理＋固定遅延内の補間・平滑化 | あり。最新比較は2秒先読みのbridge方式 |

F/GはLKだけの処理ではありません。軽い追跡処理をSystem 1、必要時のYOLO再検出をSystem 2として組み合わせています。EとF/GではYOLO呼び出し回数も異なるため、結果の差をKCFとLKだけの差とは解釈しません。

## 最新の位置精度

experiment10全2,512フレームのうち、正解位置がある1,883フレームを分母とした **TSR@0.5**（正解枠とのIoUが0.5以上の割合）です。未出力も失敗に含めます。

| モデル | B | D | E・未来情報なし | E・未来補正あり | F相当・未来情報なし | G・未来補正あり |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 81.47% | 41.64% | 44.24% | 53.27% | 77.85% | 82.47% |
| YOLOv5m | 84.23% | 18.80% | 17.10% | 20.45% | 77.43% | 81.84% |

出典は [比較CSV](basketball_tracking_midterm/reports/2026-10-28-fixed-lag-comparison/comparison.csv) です。EとF相当の未来情報なし精度は、同じ実行で保存した未来補正前の `online_tracking` に基づきます。補正前だけの処理速度をこの実行から分離して測定したものではありません。

- 旧EにはKCF初期化の不具合がありました。最新比較は修正済みです。2026-10-27の旧Eの動画・指標は履歴として残し、修正後の結果と区別します。
- experiment10を見てF/Gを設計したため、現在の結果は開発評価です。別試合・別体育館への汎化は未確認です。
- 629不可視フレームの真の位置は不明です。「ボール不在」とは扱わず、その区間の位置の正しさは評価しません。
- 最新比較はバッテリー駆動で、m/Gだけ中断後の別セッションで再測定しています。設定上は2秒先読みでも、計算が30fpsに追いつかず実際の模擬出力遅延は増加しました。AC電源時の速度や、実カメラから画面までの遅延を示す測定ではありません。
- 得点判定、選手による保持の推定、2Dコート変換、3D復元は現在の実験には含めていません。

## フォルダ構成と必要なデータ

```text
independent_reserch/
├─ basketball_tracking_midterm/    # 現在の実験本体
│  ├─ colab/                      # YOLOv5s/mの追加学習Notebook
│  ├─ configs/                    # 学習・推論・データ由来の設定
│  ├─ src/                        # 検出・追跡・未来補正・評価
│  ├─ scripts/                    # 実行・監査・集計・検証
│  ├─ tests/                      # 回帰テスト
│  ├─ data/                       # GTと監査・検証記録
│  ├─ reports/                    # GitHubで読める方法・集計結果
│  ├─ weights/                    # 重み本体はローカル保存
│  ├─ results/                    # 動画・フレーム別ログはローカル保存
│  ├─ vendor/yolov5/              # 固定版のYOLOv5ソース（準備時に取得）
│  └─ .venv/                      # 実験専用のPython環境
├─ colab_ball_yolo/dataset/        # 参照中の学習データ（ローカル）
├─ colab_ball_tracking/bundles/experiment10/  # 元ラベル・manifest（ローカル）
├─ 編集済み元データ/                # 入力動画（ローカル）
├─ elp_camera_demo.py             # カメラ表示・手動静止画保存
├─ camera_video_recorder.py       # 撮影データ収集用の録画
├─ project_runtime.py            # カメラツール用venvの選択
├─ synchronized_video_writer.py  # 録画ツールの依存モジュール
├─ requirements.txt              # ルートのカメラツール用
└─ docs/                         # 構成整理の記録
```

**Git cloneだけではデータと学習済み重みは揃いません。** ローカルの次の資産を元の位置へ配置してください。既存のファイルを削除・再分割・再ラベル付けする必要はありません。

| 必要な資産 | 参照する場所・用途 |
|---|---|
| `colab_ball_yolo/dataset/` | 408枚のtrain/val/test画像・ラベル・provenance。学習とデータ監査で使用 |
| `colab_ball_yolo/ball_yolo_colab.zip` | Colabへ渡す既存のデータ入りZIP。Notebookの`DATASET_FOLDER`指定による代替も可能 |
| `colab_ball_tracking/bundles/experiment10/manifest.json`と`data/annotations10.zip` | 元ラベルの整合性確認とGT再生成に使用 |
| `編集済み元データ/experiment10_編集.mp4` | 連続追跡評価の入力動画 |
| `basketball_tracking_midterm/weights/finetuned/{s,m}/` | `best.pt`、`best.pt.metadata.json`、`final_selection.json`、学習設定・val/test評価などのエクスポート一式 |
| `basketball_tracking_midterm/results/` | 過去結果の再照合、比較動画・説明画像の再生成で使用する実行ログ |

外部データの位置は `basketball_tracking_midterm/configs/experiment*.json` と監査スクリプトで参照します。現在の学習・評価の入口は中間発表用フォルダです。元の撮影動画・注釈・重み・実験ログは今回の整理で削除していません。

データ作成の再現用に [CVAT画像抽出](CVAT_DATASET.md)、[評価動画の準備記録](TRACKING_TEST_09_10.md)、[データセット作成コードと旧Colab学習](colab_ball_yolo/README.md)、[旧Colab追跡評価](colab_ball_tracking/README.md) のソースも公開しています。旧Colab版はUltralyticsを使う別の構成であり、現在のYOLOv5版の手順とは区別してください。各フォルダの実画像・動画・注釈ZIP・生成パッケージはGit対象外です。

## 学習から評価までの入口

以下はリポジトリ直下からのPowerShell操作です。詳細な設定とデータ配置は [実験README](basketball_tracking_midterm/README.md) を参照してください。

```powershell
Set-Location basketball_tracking_midterm

# 初回のみ：Python 3.12の専用環境と固定版ソース・初期重みを準備
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\python.exe scripts/setup_assets.py --sizes s m

# 動作確認（学習・カメラ撮影は開始しない）
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_runtime.py

# Colab用のコード配布ZIPを作成
.\.venv\Scripts\python.exe scripts/build_colab.py
```

1. `colab/train_yolov5_colab.ipynb` と生成した `colab_upload.zip` を使い、Colabでs/mをそれぞれ学習します。学習・評価スイッチは初期状態でFalseです。
2. valでモデル・運用設定を確定し、エクスポートと最終test評価を行います。testをモデル選択には使いません。
3. エクスポート一式をWindowsの `weights/finetuned/s/` または `m/` に配置します。
4. 同じ映像を使い、以下の各条件を実行します。出力先は未使用の名前にしてください。

```powershell
# 以下もbasketball_tracking_midterm直下で実行
# B：毎フレームYOLO
.\.venv\Scripts\python.exe scripts/run_experiment.py --conditions B --sizes s m --output results/b_new --offline-guard

# E：修正済み前向きKCF。未来情報なし
.\.venv\Scripts\python.exe scripts/run_experiment.py --conditions E --sizes s m --output results/e_online_new --offline-guard

# F：適応再検出＋局所LKなど。未来情報なし
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/experiment_f.json --conditions F --sizes s m --output results/f_new --offline-guard

# B/Dと固定遅延ありE/Gを共通runnerで比較（s/mを順番に実行）
.\.venv\Scripts\python.exe scripts/compare_fixed_lag.py --output results/fixed_lag_new
.\.venv\Scripts\python.exe scripts/verify_g.py --fresh-root results/fixed_lag_new --output results/fixed_lag_new/verification.json
.\.venv\Scripts\python.exe scripts/summarize_fixed_lag.py --root results/fixed_lag_new --output reports/fixed_lag_new
```

標準設定は従来のYOLOv5 v7.0、imgsz=960、FP32です。sとmはそれぞれ対応する重みを使います。モデル・閾値・環境の照合に失敗した場合、別モデルへの置換で進めず、配置と設定を確認してください。過去の修正前Eとの比較も生成する集計スクリプトは、ローカルに残している `results/comparison_20261027/` も参照します。

## リアルタイム固定遅延追跡

`basketball_tracking_midterm/scripts/realtime_tracking.py` は、固定遅延ありで最良だったYOLOv5s＋G（bridge）をカメラ入力に対して3秒遅延で動かします。処理が追いつかないときは古い未処理フレームを間引いて補間で埋め、表示遅延を一定に保ちます。方式と2026-11-16の測定は [説明資料](basketball_tracking_midterm/reports/2026-11-16-realtime-fixed-delay/README.md) にまとめています。

```powershell
Set-Location basketball_tracking_midterm
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --camera-index 0 --width 1280 --height 720 --fps 30
```

## 次の目標：魚眼カメラ2台による3次元追跡と得点判定

2026-11-16に決めた次の目標です。魚眼カメラ2台で試合を撮影し、コート座標系の3次元でボールを追跡して得点を判定します。手順は次の順番で進めます。各段階の「判断」を満たしてから次へ進み、満たさない場合は前の段階に戻ります。

| 順 | やること | 主に使うプログラム | 状態 |
|---|---|---|---|
| 1 | 魚眼カメラ2台で試合を録画する（同時にキャリブレーション用の撮影も行う） | `dual_camera_recorder.py`、`basketball_tracking_midterm/scripts/fisheye_calibrate.py` | 新規。実カメラでは未検証 |
| 2 | CVATでボールをアノテーションする（追跡用の全フレームと、学習用の抽出画像） | `prepare_cvat_dataset.py --names`、`basketball_tracking_midterm/scripts/import_cvat_gt.py`、`colab_ball_yolo/prepare_dataset.py` | 改修・新規・既存 |
| 3 | 録画データで追跡精度を検証する（必要なら魚眼画像で追加学習） | `scripts/run_experiment.py`、`scripts/run_fixed_lag.py --config`、`colab/train_yolov5_colab.ipynb` | 既存（`--config` を追加） |
| 4 | リアルタイムで追跡精度を検証する | `scripts/realtime_tracking.py --config`、`scripts/evaluate_realtime_log.py --config` | 既存（`--config` を追加） |
| 5 | 2台の映像から3次元座標でボールを追跡する | `scripts/court_pose.py`、`scripts/triangulate_ball.py`、`configs/court_fiba.json` | 新規。合成データのテストのみ |
| 6 | 座標データから得点判定を行う | `scripts/score_detector.py` | 新規。合成データのテストのみ |

新規プログラムは `basketball_tracking_midterm/tests/test_stereo.py` の合成シーン（2台の魚眼カメラにボールの放物線を投影し、三角測量と得点判定を検査）で動作を確認しただけです。実際の魚眼カメラ・体育館での検証は、1の録画が得られた時点で行います。以下、`scripts/` と `configs/` は `basketball_tracking_midterm/` 直下です。

### 1. 魚眼カメラ2台で録画する

撮影時にしか取れないものが3つあります。試合だけ撮って帰ると5以降ができません。

1. **内部パラメータ用のチェスボード撮影**（カメラごと）：試合と同じレンズ・解像度・フォーカスのまま、チェスボードを画面の端まで含む20〜40姿勢で撮ります。`camera_video_recorder.py` で動画として撮れば十分です。
2. **コート基準点が写る静止フレーム**（カメラごと）：コートの角、フリースローライン端、センターなど `configs/court_fiba.json` の基準点が写る位置で1枚撮ります。カメラは試合中に動かさないでください。コートの寸法はメジャーで実測し、`court_fiba.json` を体育館の値に書き換えます。
3. **試合の2台同時録画**：`dual_camera_recorder.py` で2台を同じ実時間基準で保存します。開始直後に両カメラに見える位置で手を叩き（またはフラッシュ）、その瞬間に `M` キーでマーカーを記録します。後で両動画の該当フレームを目視で合わせ、時刻ずれを求めます。

```powershell
# リポジトリ直下。カメラ用の .venv を使う
.\.venv\Scripts\python.exe cameras.py                      # 番号の確認（d:\自主研究\cameras.py）
.\.venv\Scripts\python.exe dual_camera_recorder.py --cameras 0 1 --width 1280 --height 720 --fps 30 --output results/recordings/game01
.\.venv\Scripts\python.exe camera_video_recorder.py --camera-index 0 --output results/recordings/calib_cam1.mp4   # チェスボード

# 内部パラメータ（basketball_tracking_midterm 直下、実験用 .venv）
.\.venv\Scripts\python.exe scripts/fisheye_calibrate.py --video ../results/recordings/calib_cam1.mp4 --every 15 --pattern 9 6 --square 0.025 --output configs/cameras/cam1_fisheye.json --preview results/cam1_undistorted.jpg
```

判断：`rms_px` が1px程度以下で、プレビューの直線が直線に戻っていること。解像度・fpsはカメラの実値を `session.json` で確認します。魚眼カメラは2台とも同じ機種・同じ設定にします。

### 2. CVATでアノテーションする

用途が2つあり、CVATのタスクも分けます。

- **追跡評価用（全フレーム）**：各カメラの試合動画をCVATの動画タスクとして登録し、全フレームにボール1個の矩形を付けます。不可視のフレームは空のままにし、「全フレームを確認した」ことを記録します。「YOLO 1.1」でエクスポートし、次で正解CSVに変換します。

```powershell
.\.venv\Scripts\python.exe scripts/import_cvat_gt.py --annotations ../datasets/fisheye/game01_cam1_yolo.zip --video "../編集済み元データ/fisheye_game01_cam1.mp4" --output data/fisheye_game01_cam1_gt.csv --all-frames-reviewed
```

- **検出器の学習用（抽出画像）**：魚眼画像で追加学習が必要になった場合に使います。`prepare_cvat_dataset.py --names` で時間帯分割の画像を抽出し、[CVAT画像抽出の手順](CVAT_DATASET.md) どおりにアノテーションして、`colab_ball_yolo/prepare_dataset.py --source <その出力> --output colab_ball_yolo/dataset_fisheye` でデータセットを作ります。`prepare_dataset.py` は注釈ZIPの名前 `ball_only_アノテーション_{train,eval,test}.zip` を前提にしているため、同じ名前で保存します。

```powershell
python prepare_cvat_dataset.py --names fisheye_game01_cam1 fisheye_game01_cam2 --output datasets/cvat_fisheye_game01
```

判断：追跡評価用は両カメラとも全フレーム確認済みであること。学習用は3分割が別の時間帯であること。

### 3. 録画データで追跡精度を検証する

`configs/experiment_fisheye.template.json` を `experiment_fisheye.json` にコピーし、動画・正解・データセットのパスを書きます。まず現在のs重みのまま測り、魚眼で精度がどれだけ落ちるかを見ます。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/experiment_fisheye.json --conditions B F --sizes s --output results/fisheye_b_f --offline-guard
.\.venv\Scripts\python.exe scripts/run_fixed_lag.py --config configs/experiment_fisheye.json --size s --condition G --method bridge --lag 2 --realtime --output results/fisheye_g
```

判断：TSR@0.5 が experiment10 の82.47%（s/G）に近いか。大きく落ちる場合は、2の学習用データで魚眼画像のYOLOv5sを追加学習し（`colab/train_yolov5_colab.ipynb` の `DATASET_FOLDER` に `dataset_fisheye` を指定）、エクスポートを `weights/finetuned_fisheye/s/` に置いて `experiment_fisheye.json` の `models.s.finetuned` を差し替えてから測り直します。周辺部でボールが楕円に歪むこと、画角が広くボールが小さく写ることが主な劣化要因です。

### 4. リアルタイムで追跡精度を検証する

[リアルタイム固定遅延追跡](basketball_tracking_midterm/reports/2026-11-16-realtime-fixed-delay/README.md) のプログラムをそのまま使います。まず録画を30fps到着として流して採点し、次に実カメラで動かします。

```powershell
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --config configs/experiment_fisheye.json --video "../編集済み元データ/fisheye_game01_cam1.mp4" --no-display --record results/realtime_fisheye
.\.venv\Scripts\python.exe scripts/evaluate_realtime_log.py results/realtime_fisheye/realtime_<timestamp> --config configs/experiment_fisheye.json
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --config configs/experiment_fisheye.json --camera-index 0 --record results/realtime_fisheye
```

判断：AC電源で間引き率が数%以下、表示遅延が3.0秒に保たれ、TSRが3の録画評価と同程度であること。2台を同時にリアルタイム処理するプログラムはまだなく、5の三角測量は録画・ログに対する後処理です。GPU1枚で2台分の処理が30fpsに収まるかは、この段階で処理fpsを見て判断します。

### 5. 2台の映像から3次元座標で追跡する

追跡は魚眼画像のまま行い、幾何計算の直前にボール中心の画素だけを歪み補正します（画像全体の補正は不要）。

1. **外部パラメータ**：1で撮った静止フレーム上でコート基準点をクリックし、`solvePnP` でカメラの位置・向きをコート座標で求めます（カメラごと）。

```powershell
.\.venv\Scripts\python.exe scripts/court_pose.py click --video "../編集済み元データ/fisheye_game01_cam1.mp4" --frame 0 --court configs/court_fiba.json --output configs/cameras/cam1_points.json
.\.venv\Scripts\python.exe scripts/court_pose.py solve --calibration configs/cameras/cam1_fisheye.json --points configs/cameras/cam1_points.json --court configs/court_fiba.json --output configs/cameras/cam1_pose.json
```

2. **時刻合わせ**：`markers.csv` と両動画の拍手フレームから、カメラBの時刻 − カメラAの時刻を秒で求めます。
3. **三角測量**：両カメラの追跡ログ（3の `tracking.jsonl` または4の `frames.jsonl`）から3次元軌道CSVを作ります。

```powershell
.\.venv\Scripts\python.exe scripts/triangulate_ball.py --track-a results/fisheye_g/<run_cam1>/tracking.jsonl --camera-a configs/cameras/cam1_fisheye.json --pose-a configs/cameras/cam1_pose.json --track-b results/fisheye_g/<run_cam2>/tracking.jsonl --camera-b configs/cameras/cam2_fisheye.json --pose-b configs/cameras/cam2_pose.json --offset-s 0.033 --output results/fisheye_3d/game01.csv
```

判断：`court_pose.py solve` の再投影誤差が数px以下、`triangulate_ball.py` の残差の中央値が数px、2本の視線の角度が20°以上あること。精度の実測は、ボールをフリースローライン上やリング上など座標の分かる場所に置いた静止フレームで行い、推定座標との差を記録します。残差が大きいときは時刻ずれ、クリック点、コート寸法の順に疑います。

### 6. 座標データから得点判定を行う

最初の判定規則は「リング面（高さ3.05m）を上から下へ横切り、横切った点がリング中心から半径0.225m以内」です。半径0.225〜0.525mはリングに当たった可能性として `rim_miss` に分けます。正解は動画を見て得点時刻を記入したCSV（`time_s,event`）で、許容1秒で照合します。

```powershell
.\.venv\Scripts\python.exe scripts/score_detector.py --trajectory results/fisheye_3d/game01.csv --court configs/court_fiba.json --hoop A --truth data/fisheye_game01_goals.csv --output results/fisheye_3d/game01_events.csv
```

判断：得点の適合率・再現率を出し、取りこぼしの原因（リング付近での追跡失敗、三角測量の残差、時刻ずれ）を分類します。その結果を見て、リングやボードの検出、軌道の放物線当てはめ、片側カメラだけの区間の扱いなど、次の判定規則を決めます。判定の閾値は開発用の試合で決め、評価用の別試合では固定します。

## カメラ表示・録画

ルートに残した2つのツールは撮影データ収集用です。推論評価とは別のルート `.venv` を使い、`project_runtime.py` がその環境を選択します。

```powershell
# リポジトリ直下で実行。初回のみカメラ用環境を準備
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 表示。sで静止画保存、qで終了
.\.venv\Scripts\python.exe elp_camera_demo.py --camera-index 0

# 実行すると録画開始。終了はqまたはEsc
.\.venv\Scripts\python.exe camera_video_recorder.py --camera-index 0 --width 1280 --height 720 --fps 30
```

カメラ番号は接続状況に合わせます。解像度・FPSは要求値であり、実際の取得値はカメラに依存します。録画ツールは実時間に合わせてフレームを複製・間引きするため、厳密な実カメラ遅延・取得フレーム数の測定用ではありません。

VS Codeの既定Pythonは主実験の `basketball_tracking_midterm/.venv` に合わせています。ルートのカメラ環境と中間発表用環境のrequirementsを混ぜないでください。

当初の目的・計画は [研究計画書](様式2_申請書・自主研究計画書(基)_v3_tanaka.pdf) に残しています。現在の実装・測定済み範囲はこのREADMEと各実験レポートを基準にしてください。
