# リポジトリ構成の整理（2026-10-29）

現在の学習・推論・評価の入口を `basketball_tracking_midterm/` に揃えました。整理前のコミットは `b9ea8745da0d0d2944042ff74375af1f9764a3a3` です。旧コードが必要な場合はGit履歴から参照できます。

作業中にGitHub側へ追加されたREADME更新 `848535f` も統合しました。「一般的な体育館で持ち運べるPCを使う」「中学生・高校生・大学生などのアマチュア向け」という研究の前提を新READMEに引き継いでいます。

## 残すもの

- `basketball_tracking_midterm/` のソース、設定、Notebook、テスト、データ監査、実験レポート。旧Eの記録も不具合の注記とともに研究履歴として維持します。
- 学習元の `colab_ball_yolo/dataset/`、データ入りZIP、GT由来の `colab_ball_tracking/bundles/experiment10/`。これらは中間発表用の設定・監査処理が直接参照します。
- ローカルの元動画、注釈、重み、学習結果、推論動画、フレーム別ログ、既存venv。参照パスも変更しません。
- 撮影用の `elp_camera_demo.py`、`camera_video_recorder.py` と、その依存 `project_runtime.py`、`synchronized_video_writer.py`。
- 当初の研究計画書PDF。現在の実装仕様とは区別してルートREADMEから案内します。

作業開始時からGit未管理だった `colab_ball_*`、`prepare_cvat_dataset.py`、`prepare_tracking_test.py`、`CVAT_DATASET.md`、`TRACKING_TEST_09_10.md` は変更・削除・追加コミットしていません。データ生成の由来を保つためローカルに残します。GitHub上の現在の実験入口は中間発表用フォルダです。

## 削除した32ファイル

| 区分 | 削除対象 |
|---|---|
| 旧単独・リアルタイム追跡 | `kcf_basketball_tracker.py`、`realtime_basketball_tracker.py`、`tracking_baseline_experiment.py` |
| 旧YOLO/ByteTrack | `yolo_basketball_tracker.py`、`yolo_bytetrack_live.py`、`yolo_bytetrack_video.py`、`YOLO_ByteTrack_使い方.md` |
| 旧適応追跡 | `adaptive_basketball_tracker.py`、ルートの`adaptive_tracking.py`、`evaluate_adaptive_tracking.py`、`test_adaptive_tracking.py`、`ADAPTIVE_TRACKING.md` |
| 旧M1〜M6・得点評価 | `experiment_tracking.py`、`experiment_detectors.py`、`run_tracking_experiment.py`、`prepare_experiment_data.py`、`train_blob_cnn.py`、`evaluate_tracking_experiment.py`、`evaluate_score_events.py`、`test_experiment_pipeline.py`、`EXPERIMENT_PROTOCOL.md` |
| 旧実験の入力例 | `experiment_examples/detection.yaml`、`score_predictions.csv`、`sessions.csv`、`tracking_gt.csv` |
| 旧構想・環境・予算計算 | `tracking_experiment_setup.md`、`tracking_method_summary.md`、`既存手法まとめ.md`、`requirements-cuda.txt`、`budget.py` |
| Gitに入っていた生成キャッシュ | `__pycache__/elp_camera_demo.cpython-312.pyc`、`__pycache__/tracking_baseline_experiment.cpython-312.pyc` |

削除したのは旧ルート実装です。条件EのKCFは `basketball_tracking_midterm/src/tracking.py`、Fの処理は同フォルダ内の `src/adaptive_tracking.py`、Gや逆向きKCFは `src/lag_tracking.py` に残っています。ルートの同名モジュールを削除しても、パッケージ内の相対importと `src.*` のimportは維持されます。

## 更新した案内・設定

- ルートREADMEを、現在のA〜G条件、精度、制約、実行手順、ローカル資産の配置に書き換えました。
- ルート `requirements.txt` は、残すカメラツールに必要なNumPyとOpenCVに絞りました。既存のバージョン指定は維持し、venvのインストール状態は変更していません。
- VS Codeの既定Pythonを `basketball_tracking_midterm/.venv/Scripts/python.exe` に変更しました。既にVS Codeで選択済みの環境がある場合は、Pythonインタープリター選択で切り替えてください。
- 中間発表用のrequirements、設定値、ソース、Colab配布物、過去の測定結果は変更しません。

## 検証

整理前後で中間発表用のGit管理対象319ファイルのSHA-256がすべて一致しました。ソース・設定・既存レポートは変更していません。

| 確認 | 結果 |
|---|---|
| 中間発表用コードから削除したルートモジュールへのimport | なし |
| 学習データ | train 300枚・val 50枚・test 58枚の検証成功 |
| GT | 元注釈から一時フォルダへ再生成したCSVが既存GTと一致 |
| 元動画・重み・YOLOv5ソース | ハッシュ・metadata・固定ソースの検査成功 |
| 回帰テスト | 42件成功 |
| 実データによる短い推論 | s/B・s/E・s/F各12フレーム、s/G 65フレームが正常終了 |
| カメラツール・学習スクリプト | `--help`による起動確認成功 |
| Colab配布物 | ZIP内全ファイルと現行コードの一致、Notebookコードセルの構文確認 |
| READMEの結果表 | 保存済み比較CSVと照合 |

詳細は [検証記録](repository_cleanup_20261029.json) に保存しています。短い推論は動作確認であり、学習や全動画の精度・速度の再測定ではありません。今回の確認ではカメラの起動・撮影も行っていません。
