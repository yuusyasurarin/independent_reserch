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
