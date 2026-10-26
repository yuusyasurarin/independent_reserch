# バスケットボール追跡・中間発表用実験

目的は、全フレームYOLOに対して、低頻度YOLOの中間フレームをKCFで補う効果を実測することです。
2026-10-27にs/mのA〜Eを実測しました。方法・結果・制約は後半の「8. 実測結果」にまとめています。
得点判定、LSTM、選手追跡、コート変換、3D復元は含みません。

このフォルダ内に学習・推論・評価を新規作成しました。既存コード・元画像・ラベル・動画は参照のみです。
従来の **YOLOv5 v7.0（915bbf294bb74c859f0b41f1c23bc395014ea679）** を使用し、YOLOv5uやYOLO26へ置き換えません。
ColabとWindowsで同じソース、PyTorch 2.5.1／torchvision 0.20.1を使います。
ByteTrackはSupervision 0.25.0の実装です。出典・ライセンスは [THIRD_PARTY.md](THIRD_PARTY.md) に記載しています。

## 1. データセットの配置

Windowsの作業ディレクトリをこのフォルダにします。

```powershell
Set-Location 'D:\自主研究\independent_reserch\basketball_tracking_midterm'
```

確認済みのデータは次の場所です。複製せず、設定から参照します。

| 用途 | 場所・構成 | 確認結果 |
|---|---|---|
| 学習・検証・最終test | `../colab_ball_yolo/dataset/{images,labels}/{train,val,test}` | train 300、val 50、test 58枚 |
| 背景画像 | 上記の明示的な空txtラベル | train 113、val 20、test 28枚。既存資料で確認済み背景 |
| クラス | `data.yaml` | `0: ball`、1クラス |
| 元動画 | dataset/provenance | experiment11／12、時間帯で分割、境界に除外区間あり |
| 連続追跡用動画 | `../編集済み元データ/experiment10_編集.mp4` | 1280×720、30fps、2,512フレーム、83.733333秒 |
| experiment10の元ラベル | `../colab_ball_tracking/bundles/experiment10/data/annotations10.zip` | 1,883フレームに各1個のball矩形 |
| experiment10の位置不明区間 | 629フレーム | 2026-10-26にユーザー確認済みの不可視区間。ボール不在とは断定しない |

データセットtest（58枚）はモデル確定後の**静止画検出評価**です。experiment10は**連続動画の検出・追跡比較**で、役割を分けます。
データセット内の画像・ラベル対応、クラス・矩形範囲、破損、元manifestのハッシュ、split間の完全重複を検査します。
欠損ラベルはエラー、空ラベルは明示的な背景許可が必要です。自動修正・ラベル生成はしません。
今回の408枚は検査を通過し、split間の完全重複はありません。

experiment10と学習元動画のSHA-256一致、provenanceに記録された元フレームの画素ハッシュ一致はありませんでした。
2026-10-26にユーザーから、**experiment10は別の撮影映像で、同じプレーの重複もない**と確認を受けました。
`configs/data_provenance.json` にこの確認と対象動画のSHA-256を保存しています。機械的な完全一致検査と、人手の確認を区別して記録します。
今回のexperiment10は学習元動画と独立した評価映像として扱います。ただし、83.733秒の1動画であり、別試合・別体育館への一般化を十分に示すものではありません。
別動画へ設定を変えた場合、この確認は引き継ぎません。

検査を再実行するコマンド（環境準備は次節）：

```powershell
.\.venv\Scripts\python.exe scripts/audit_data.py --allow-background
```

`data/audit/` に一覧・ハッシュ・検査結果・動画情報が保存され、確認済みのexperiment10ラベルを `data/experiment10_gt.csv` へ変換します。
`--allow-background` は今回の確認済み背景161枚に対する指定です。別データでは確認を引き継がないでください。

## 2. ColabでYOLOv5sを追加学習

1. `colab_upload.zip` をDriveの `マイドライブ/basketball_tracking_midterm/` にアップロードします。このZIPはコードのみです。
2. 既存のデータ入りZIP `../colab_ball_yolo/ball_yolo_colab.zip` をDriveへ置きます。既に置いてあれば再アップロード不要です。
3. `colab/train_yolov5_colab.ipynb` をColabで開き、GPUランタイムを選択します。
4. 上からGPU確認・Driveマウント・専用仮想環境の準備を実行します。GPU種類とVRAMを毎回記録します。
5. `MODEL_SIZE='s'`、`DATASET_ZIP` を実際のDriveパスに指定します。
   ZIP以外では `DATASET_FOLDER` を指定できます。標準ではColabローカルへコピーします。
6. データ検証結果を確認し、学習セルの `RUN_TRAIN=True` で開始します。
7. 完了したrunを `SELECTED_RUN` に指定し、`RUN_VAL=True` でval評価します。
8. valでモデル・運用閾値を確認した後、`VAL_EVIDENCE` にval評価の `metrics.json`、`SELECTION_NOTE` に設定の確認根拠を記載し、`RUN_EXPORT=True` にします。
9. `FINAL_SELECTION` にエクスポートした `final_selection.json` を指定し、`RUN_FINAL_TEST=True` で最終testを実行します。

最初の例はimgsz=960、100 epochs、batch=4、seed=0、patience=20、SGDです。最適化済み設定ではありません。
学習スクリプトのhypは固定YOLOv5の `hyp.scratch-low.yaml` で、実行runにコピーされます。
bestはvalのYOLOv5 fitnessで選択し、testを学習・early stopping・閾値選択に使いません。
データ検査でtest画像の破損・重複は調べますが、test精度は最終評価セルを実行するまで計算しません。
PR/AP評価のconfidence floor=.001と、運用時のconfidenceを区別してください。
valログのPrecision/RecallはYOLOv5側の曲線上の選択閾値に依存し、指定した運用confidenceでの値とは限りません。

VRAM不足ならBATCHを4→2→1と変更して新しいrunで実行し、変更を記録します。
モデルサイズや画像サイズを黙って変更しません。自動バッチ選択は使用しません。
ColabのGPUは固定ではないので、Colabの速度をWindowsの3050 Tiの速度として報告しません。

保存先は `MyDrive/basketball_tracking_midterm/runs/s/日時/` です。
`best.pt`、`last.pt`、`resume_latest.pt`、`results.csv`、曲線、val画像、`opt.yaml`、`hyp.yaml`、
実行ログ、`training_metadata.json`、依存一覧、使用画像・ラベル一覧とハッシュが残ります。
Drive容量を抑えるため、`SAVE_PERIOD=-1`（training.jsonの`save_period=-1`）で、`epoch0.pt`、`epoch1.pt`…の個別保存を無効にしています。
`last.pt`と`resume_latest.pt`は毎epoch同じ名前で更新し、`best.pt`はvalの最良モデルとして更新します。
新規runのweights内の.ptは基本3個で、epoch数に比例して増えません。再開用ファイルの更新中は一時コピー用の空き容量も必要です。
複数runやexportsはそれぞれモデルを持つため、Drive全体の使用量が一定になるわけではありません。
最初のepoch終了前に切断された場合は再開用データがありません。

旧版を使用中の場合は、更新したNotebookを開き直し、再生成した`colab_upload.zip`をDriveの同名ファイルへ置き換えてください。
実行中の学習は自動では切り替わりません。中断から再開する場合、Colabのランタイムを削除して新規接続し、環境・設定・データ準備から実行します。
展開済みコードとZIPの不一致を検査するため、ZIPだけ差し替えた状態で同じランタイムの準備セルを再実行しないでください。
旧版で蓄積した`epoch*.pt`は、この変更では削除しません。空き容量は自動では回復しません。

中断からの再開は、環境・データ準備を再実行し、元runを `RESUME_RUN` に指定して `RUN_RESUME=True` にします。
`resume_latest.pt` はoptimizer、epoch、EMAを含みます。正常終了後のYOLOv5のlast/bestはoptimizerが除かれるため、再開には使いません。
bestだけを初期重みにして別runで続ける追加学習と、optimizerを復元するresumeは別です。完了済みrunをresumeで延長する処理は拒否します。
データやモデルサイズが変わったresumeも拒否します。
再開時も現在の`SAVE_PERIOD`を適用します。YOLOv5が再読込するrun内の`opt.yaml`を更新し、変更前の設定は`opt.before_storage_change.yaml`、保存方針は`resume_storage_policy.json`へ記録します。

## 3. ColabでYOLOv5mを追加学習

`MODEL_SIZE='m'` に変更し、設定セルから再実行して別runを作ります。
公式 `yolov5m.pt` を初期重みとして使用し、sで学習した重みは使用しません。
基本条件はsと共通にし、メモリ不足でbatch等を変更したら両runのmetadataを添えて比較します。
出力先は `runs/m/日時/`、エクスポート先は `exports/m/日時/` です。
Notebookの学習・再開・評価スイッチは初期状態でFalseであり、今回の実装作業では長時間学習を開始していません。

## 4. 重みをWindowsへ配置

最初のオンライン準備で専用環境と公式ソース・初期重みを取得します。
Python 3.12とインターネット接続が必要です。既存プロジェクトの仮想環境は変更しません。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\python.exe scripts/setup_assets.py --sizes s m
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_runtime.py
```

固定した主要依存関係は `requirements.txt` とsetupスクリプトを参照してください。
推移依存も `constraints-common.txt` で固定し、今回確認したWindows環境全体は `requirements-lock-windows.txt` に保存しています。
`opencv-contrib-python` のみを使用し、他のOpenCV wheelを重ねてインストールしません。
SupervisionはOpenCV wheelの競合を避けるため `--no-deps` で導入し、必要な依存はrequirements側で用意します。
このためpip checkのOpenCVディストリビューション名に関する警告と、実際のcv2/KCF利用可否は区別します。
再現用の実際の全依存バージョンは各runの `environment.txt` に保存します。

Colabのexportsフォルダを丸ごとダウンロードし、次の配置にします。

```text
weights/
  pretrained/yolov5s.pt  （setup_assetsで公式v7.0を取得）
  pretrained/yolov5m.pt
  finetuned/s/best.pt
  finetuned/s/best.pt.metadata.json
  finetuned/s/final_selection.json
  finetuned/m/best.pt
  finetuned/m/best.pt.metadata.json
  finetuned/m/final_selection.json
```

exports内の学習設定・val/test結果も一緒に残します。
`configs/experiment.json` のimgsz・confidence・nms_iou・halfを `final_selection.json` に合わせ、`selection_split` を `val` にします。
未調整COCOはsports ballの32、追加学習済みはballの0です。重みのSHA-256・クラス・サイズ・YOLOv5 commitを照合します。
追加学習済み重みと選択設定が未配置ならB〜Eは実行できません。未調整重みで代用しません。
s/mでval選択設定が違う場合は、設定JSONを別名で2つ用意して個別に実行します。

体育館へ移動する前にソース・必要な重み・依存・CUDA・KCFの確認を終えてください。
実験runnerは既存のローカルファイルだけを読み、重みの自動取得・依存の自動インストールをしません。
必要なら `--offline-guard` を付けると、実行中のPythonからのインターネット接続を拒否して確認できます。

## 5. experiment10の比較実験

| 条件 | 実装 | 最終対象選択 |
|---|---|---|
| A | 未調整YOLO、毎フレーム | ball候補の最高信頼度。過去位置・IDを使わない |
| B | 追加学習YOLO、毎フレーム | Aと同じ単純選択 |
| C | 追加学習YOLO＋ByteTrack、毎フレーム | 現在出力のある前IDを優先、なければ最高信頼度の追跡候補 |
| D | 追加学習YOLO、3フレームごと＋直前矩形保持 | 距離・経過時間・弱いIoU項で対応付け |
| E | 追加学習YOLO、3フレームごと＋KCF | Dと同じ再検出対応付け。YOLOでKCFを初期化・補正 |

B〜Eは同じサイズ内で同じ学習済み重み・imgsz・候補抽出設定です。
NMS後・最終対象選択前のYOLO候補をconfidence≥.001で保存します。A/B/D/Eの最終選択は共通confidence（初期値.25）以上です。
CはByteTrackの低信頼度対応付け用に **confidence>.1** の候補を残し、高信頼度閾値は `byte_activation_threshold`（初期値.25）です。
固定Supervision版の新規トラック開始閾値はactivation+.1です。これらは共通の検出候補抽出とは別の追跡設定として記録します。
最終追跡が出ていないlostトラックのKalman予測だけを位置出力にはしません。
KFのdtは1動画フレームで、Cだけを毎フレーム更新します。D/Eの間欠検出に同じKFを流用しません。

30fps／interval=3では0,3,6,…フレームでYOLOを実行し、名目10Hzです。実行回数と実現頻度も記録します。
低頻度の意味は「動画時間当たりの呼び出し頻度」であり、処理が実時間に間に合うという意味ではありません。

D/Eの初期化・ロスト後は最高信頼度から開始します。追跡中は現在位置から
`association_base_px + association_speed_px_s × 前回YOLO採用からの秒数` の距離内で対応付けます。
非重複の高速ボールも候補になり、距離・IoU・信頼度で順位を付けます。正解ラベルは参照しません。
予定したYOLOフレームで候補なし／対応付け失敗なら即座にlostとし、古い矩形を保持しません。
前回YOLOから `max_age_s`（初期値0.3秒）を超えた位置も出力しません。
KCF失敗・範囲外はlostです。KCFの成功フラグは対象が正しい保証ではなく、YOLOと同じ信頼度も作りません。
KCFが扱う座標も元画像です。YOLOはletterbox後に元画像のxyxyへ戻します。

まず、配置済みの未調整sだけを明示的に実行できます。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes s --conditions A
```

学習済みsを配置し、設定を合わせた後：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes s --conditions A B C D E
```

mも用意できた場合：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes s m --conditions A B C D E
```

描画・動画保存なしは `--no-video`、短い動作確認は `--max-frames 120` です。
部分処理結果は `complete_video=false` と記録し、全動画評価として扱いません。
最大速度で全フレームを順番に処理し、フレーム間引き・並列化・未来情報による補正は行いません。
`--allow-version-mismatch` は別環境のスモーク確認に限る明示オプションで、結果に非固定環境と記録します。

比較表は実際に完了したrunだけを指定します。存在しない条件の値は補いません。

```powershell
.\.venv\Scripts\python.exe scripts/compare_results.py --runs results/s_A_日時 results/s_B_日時 results/s_C_日時 results/s_D_日時 results/s_E_日時 --output results/comparison_s
```

入力、重み、設定、環境、保存有無が異なる場合や部分処理の場合は比較上の問題として出力します。
動画保存あり／なし、実時間ペース／最速、CPU／GPU、異なるGPUの結果は分けます。
検出・画像キャッシュは使いません。OSファイルキャッシュは制御していないため、その旨を記録します。

## 6. 正解がある場合の精度評価

experiment10には正解があるので、通常は実行後に自動評価します。
629フレームは確認済み不可視ですが、画面外／遮蔽／ボール不在の区別はないため `unknown_position` にします。
位置精度の分母は1,883フレームです。不可視区間に出た矩形は別集計し、FPと断定しません。

正解CSVは0始まりの `frame_id,timestamp_s,status,x1,y1,x2,y2,segment_id,note` です。座標は元画像のピクセル単位、時刻はframe_id/fpsです。

| status | 意味 | 矩形 | 検出／位置精度への使用 |
|---|---|---|---|
| visible | 人手でボール位置を評価できる | 必須 | 使用 |
| absent | 人手で対象ボール不在を確認 | 空欄 | 検出FPを判定。TSRの分母から除外 |
| unknown_position | 完全遮蔽・画面外などで位置不明 | 空欄 | 位置評価・検出FP/FNから除外 |
| unannotated | 未確認 | 空欄 | 除外 |

疎なCSVの欠落行はunannotatedです。欠落をabsentに変換しません。
人手で全フレームを確認して矩形と状態を付けます。YOLO自身の出力を正解にはしません。
別動画やラベルを編集した場合、対応するmetadataの動画・CSVハッシュも更新して整合性を保ってください。
metadataなしの独自CSVも受け付けますが、動画対応・状態・時刻・座標検査は行います。

正解がまだない別動画は、configのvideoを指定して `--no-gt` を付けると推論・動画・処理時間のみ計測できます。
後日人手正解ができた場合は、YOLOを再実行せず保存出力を評価できます。

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_saved.py --run results/s_A_日時 --gt data/experiment10_gt.csv --output results/reevaluated_A
```

検出は候補とGTを信頼度順に1対1で対応付け、IoU≥.5でTPとします。
本物を見逃してリングだけを誤検出した場合はFP=1、FN=1です。Precision・Recall・F1は運用confidenceで計算します。
AP@.5は候補floor以上のNMS後候補による101点補間PRです。KCF・保持出力にはPRを計算しません。
D/Eの検出PR/APは**YOLO実行フレームのみ**で、毎フレーム条件と評価フレーム集合が違います。
検出しなかった中間フレームを含む方式全体の比較には、共通のGT上の追跡TSRを使います。

TSRはIoU≥.3と≥.5を出力し、位置評価可能な全フレームを分母にし、未出力も失敗に数えます。
CLEは「位置評価可能かつ出力あり」の全フレームで計算し、誤追跡も含めます。件数・平均・中央値・95パーセンタイルと出力率を併記します。
連続失敗時間（IoU<.5）と連続未出力時間は、連続したvisible区間内だけを数えます。
不可視・未アノテーション・segment境界で区切るため、遮蔽をまたぐ真のロスト時間や復帰時間ではありません。
シュート境界未注釈や疎な正解の場合、時間指標には制約があります。ROCやTN不明なAccuracyは主指標にしません。

## 7. 任意の15分ループ安定性試験

83秒の動画の1回処理と、15分試験は別です。必要な場合だけ次を実行します。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes s --conditions E --loop-seconds 900 --realtime --no-video
```

同じexperiment10を元動画fpsのペースで反復し、ループ境界で追跡状態・検出スケジュール・軌跡をリセットします。
処理が遅い場合はフレームを落とさず遅れを記録します。意図した待機時間と処理時間を分けます。
これは**同じ映像の反復による処理安定性試験**であり、独立した15分映像の精度評価ではありません。
精度集計は最初の1周だけを使い、同じ正解を反復して標本数を増やしません。
83秒の1回結果から15分の安定性は保証しません。許容表示遅延だけで処理能力不足が解消するとは扱いません。

## 出力と時間計測

各runは別ディレクトリに保存します。

| ファイル | 内容 |
|---|---|
| run_config.json / environment.txt | 実行コマンド、設定、GPU、ライブラリ、重み・入力・自作ソースのハッシュ |
| candidates.jsonl | NMS後の全ball候補、YOLO実行有無、frame_id、cycle |
| tracking.jsonl / tracking.csv | 最終矩形、track_id、初期化・補正・追跡・保持・lost、情報源、時刻 |
| timing.csv | 前処理・推論・後処理・追跡・描画保存・ログ/計測の時間、待機・遅れ |
| metrics.json / summary.csv | 検出・追跡・性能の集計と分母・制約 |
| per_frame.csv / detector_pr.csv | GT対応、IoU、CLE、成功判定、PR点 |
| tracked.mp4 | 元fpsで保存した矩形・軌跡・状態表示動画 |
| tracking_metrics.png / detector_pr.png | 時系列と検出PR |
| failures/ | 出力あり・低IoUの代表例。誤対象か位置ずれかは目視で確認 |
| telemetry.json | nvidia-smiのGPUメモリ・温度などの定期値 |
| status.json | initializing / complete / failed、部分処理の区別 |

GPU段階の前後でCUDA同期します。前処理はletterbox・テンソル化・転送、推論はforward、後処理はNMS・座標復元・CPU取得です。
ウォームアップ3回は別記録で動画中のYOLO回数に含めません。
全体時間は読込・追跡・描画保存・ログ・ハードウェア計測を含み、モデル読込／ウォームアップと終了後の評価・グラフ作成は除外します。
処理時間の平均・中央値・95パーセンタイル、PyTorchの最大GPUメモリ、取得できた温度を保存します。
`processing_fps` は意図した実時間待機を除き、`wall_fps` は待機を含みます。
全フレーム検出は30fps達成を意味しません。これらはオフライン処理の時間であり、撮影から表示までの遅延ではありません。

## 実装確認・再生成

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_colab.py
```

重要テストはクラス・座標範囲、letterbox座標往復、0始まりのフレーム時刻、検出間隔、1対1のTP/FP/FN、
未出力を含むTSR、欠損／空ラベル、重複、KCF・ByteTrack・lost処理、再開用チェックポイント保存を対象にしています。
検証の実施結果と未実施項目は `VERIFICATION.md` に記載します。

最初に行う操作は **ColabへコードZIPと既存データZIPを用意し、YOLOv5sを追加学習すること**です。
Windows側の全条件比較は、s/mそれぞれの追加学習とvalでの設定確定後に実行してください。

## 8. 実測結果：2026-10-27（日本時間）

s/mの追加学習済みモデルを受領し、重み・学習記録・val結果のハッシュを検証しました。
両モデルは同じtrain/val/test（300/50/58枚）と主要学習条件（imgsz=960、最大100 epochs、batch=4、seed=0、patience=20、SGD）です。
mは省容量版のsave_period=-1、sは旧版の1で学習されています。これはepoch別の重み保存方針の差で、学習条件の変更ではありません。
両方のColab学習GPUはTesla T4です。以下の動画性能はWindowsのRTX 3050 Ti Laptop GPU 4GBで測定しました。

### 実行方法と固定条件

mのB〜Eを各120フレームで動作確認後、全10条件をm/A〜E、s/A〜Eの順にGPUで逐次実行しました。
前日の測定値は今回の表に混ぜていません。各条件は1回測定です。
入力960、confidence=0.25、候補floor=0.001、NMS IoU=0.45、FP32、動画描画・保存あり、検出・画像キャッシュなし。
`final_selection.json`の初期設定を固定しており、experiment10の結果を見て閾値を調整していません。追跡パラメータも初期設定です。
D/Eは3フレームごとの検出で、838回（毎フレーム条件2,512回の約1/3）。元動画時間上の名目頻度は10Hzです。
GPUは各段階の前後で同期し、ウォームアップ3回は計測から除外しました。

```powershell
Set-Location 'D:\自主研究\independent_reserch\basketball_tracking_midterm'
.\.venv\Scripts\python.exe scripts/check_runtime.py
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes m --conditions B C D E --max-frames 120 --offline-guard --output results/smoke_m_20261027
.\.venv\Scripts\python.exe scripts/run_experiment.py --sizes m s --conditions A B C D E --offline-guard --output results/comparison_20261027
```

実行フォルダ名の日時はUTCです。`20261026T15...Z`は日本時間2026-10-27に対応します。

### 位置精度と処理速度

共通の位置評価対象は1,883フレーム。未出力もTSRでは失敗として数え、不可視629フレームは位置不明として除外します。
CLEは出力がある位置評価可能フレームすべて（誤対象を含む）の平均で、未出力は含まず、対象件数を併記します。

| サイズ | 条件 | TSR≥0.3 | TSR≥0.5 | CLE平均px（件数） | 出力率 | 処理fps | YOLO回数 |
|---|---|---:|---:|---:|---:|---:|---:|
| s | A | 31.70% | 20.18% | 87.11（860） | 45.67% | 25.94 | 2512 |
| s | B | 86.51% | 81.47% | 8.08（1665） | 88.42% | 27.21 | 2512 |
| s | C | 52.31% | 48.86% | 10.09（1007） | 53.48% | 26.57 | 2512 |
| s | D | 61.29% | 41.64% | 15.28（1656） | 87.94% | 50.11 | 838 |
| s | E | 60.44% | 44.34% | 13.92（1362） | 72.33% | 46.61 | 838 |
| m | A | 49.81% | 36.06% | 58.08（1124） | 59.69% | 15.26 | 2512 |
| m | B | 89.11% | 84.23% | 64.81（1839） | 97.66% | 15.56 | 2512 |
| m | C | 16.68% | 14.60% | 625.66（1739） | 92.35% | 15.22 | 2512 |
| m | D | 27.46% | 18.80% | 445.20（1803） | 95.75% | 36.24 | 838 |
| m | E | 25.81% | 19.01% | 487.55（1692） | 89.86% | 33.50 | 838 |

![10条件の位置成功率と処理速度](reports/2026-10-27/comparison.png)

A=未調整YOLO、B=追加学習YOLO、C=追加学習YOLO＋ByteTrack、D=3フレームごとYOLO＋位置保持、E=3フレームごとYOLO＋KCF。

### 仮説に対する今回の結果

- **s**：EのTSR@0.5はDに対して+2.71ポイント、Bに対して-37.12ポイント。Eの処理fpsはBの1.71倍でした（この1回の測定）。
- **m**：EのTSR@0.5はDに対して+0.21ポイント、Bに対して-65.22ポイント。Eの処理fpsはBの2.15倍でした（この1回の測定）。

DとEは同じ検出間隔・重み・共通設定なので、この差がKCFで中間フレームを補った効果を示します。
一方、Bとの差は残るため、速度と精度の両方で判断します。単一動画・単一測定で一般的な優位性は断定しません。
今回のEはDよりTSR@0.5がわずかに改善しましたが、TSR@0.3と出力率は低下しています。KCF追加による一律の改善は確認できません。
両サイズともBとの精度差が大きく、「全フレーム検出に近い精度を保って低負荷化する」という目標には、現状の設定では達していません。
Cも対象選択・トラック成立条件によりBより成功率が下がることがあります。追跡を追加すれば必ず改善するわけではありません。

今回のm/Cではframe 651（21.7秒）で、正解ボールから離れた画面左下の枠をtrack_id=175として継続していることを目視確認しました。
ログではID175を701フレーム、ID344を1,143フレームで出力しています（不可視区間の出力も含む）。
現行Cは同じID、D/Eは過去位置との近さを優先するため、背景の誤検出を継続する問題があります。
これは現在の検出器・パラメータ・単一対象選択規則を組み合わせた条件の結果であり、ByteTrackやKCF一般の性能の結論ではありません。
Bについても、mは成功率だけでなくCLE・誤検出数・速度をsと合わせて判断してください。mの大型化がすべての指標を改善するとは限りません。

### 生の検出評価（最終1個の位置出力とは別）

| サイズ | 条件 | Precision | Recall | F1 | AP@0.5 | 検出評価フレーム |
|---|---|---:|---:|---:|---:|---:|
| s | A | 0.3833 | 0.2119 | 0.2729 | 0.1690 | 1883 |
| s | B | 0.8344 | 0.8216 | 0.8279 | 0.8426 | 1883 |
| s | C | 0.8344 | 0.8216 | 0.8279 | 0.8426 | 1883 |
| s | D | 0.8382 | 0.8195 | 0.8288 | 0.8405 | 626 |
| s | E | 0.8382 | 0.8195 | 0.8288 | 0.8405 | 626 |
| m | A | 0.5286 | 0.3675 | 0.4336 | 0.3088 | 1883 |
| m | B | 0.4175 | 0.8593 | 0.5620 | 0.8176 | 1883 |
| m | C | 0.4175 | 0.8593 | 0.5620 | 0.8176 | 1883 |
| m | D | 0.4195 | 0.8530 | 0.5624 | 0.8264 | 626 |
| m | E | 0.4195 | 0.8530 | 0.5624 | 0.8264 | 626 |

D/Eの検出PR/APはYOLOを実行したフレームだけです。A〜Cとは評価フレーム集合が異なるため、方式全体の精度比較は上表の共通TSRで行ってください。
KCFや位置保持には検出confidenceを付与せず、それらの出力から検出PR曲線を作っていません。

### Colabの静止画val/test（ユーザー実行済み結果）

| サイズ | split | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| s | val | 0.8431 | 0.8960 | 0.9053 | 0.3928 |
| s | test | 0.8581 | 0.8066 | 0.9221 | 0.5218 |
| m | val | 1.0000 | 0.9311 | 0.9861 | 0.4043 |
| m | test | 0.8894 | 0.8667 | 0.9051 | 0.5062 |

ここはColabの保存結果を確認したもので、Windowsで静止画testを再実行していません。
ColabのPrecision/RecallはYOLOv5が選ぶ曲線上の閾値に依存し、運用confidence 0.25での値とは限りません。
Colab評価はNMS IoU=0.6、動画運用は確定済みの0.45です。画像集合も違うため、testとexperiment10の値を同一条件として比べないでください。

### 結果の検証と保存先

全10条件で、動画を最後までデコードし2,512フレームを確認。候補・最終出力・CSVのフレーム対応、元動画時刻、
毎フレーム／3フレームごとの実際のYOLO回数、元画像座標、保持／KCF出力のconfidenceが空であることを検査しました。
全条件で保存済み予測から検出・追跡指標を再計算し、初回評価と完全一致しました。

```powershell
$comparisonRuns = @(Get-ChildItem -LiteralPath 'results/comparison_20261027' -Directory | Sort-Object Name | ForEach-Object { $_.FullName })
.\.venv\Scripts\python.exe scripts/verify_experiment_runs.py --runs $comparisonRuns --output results/comparison_20261027/verification.json
.\.venv\Scripts\python.exe scripts/compare_results.py --runs $comparisonRuns --output results/comparison_review
```

再実行時は、同じ親フォルダに複数回のrunを混ぜず、今回比較するrunだけを指定してください。比較表の出力先は未使用のフォルダ名にします。

- GitHubにも収録：[比較CSV](reports/2026-10-27/comparison.csv)、[比較JSON](reports/2026-10-27/comparison.json)、[全run検証記録](reports/2026-10-27/verification.json)、[モデル照合記録](reports/2026-10-27/model_audit.json)。
- 各条件の集計・実行設定・GPU計測は `reports/2026-10-27/s_A/`〜`m_E/`、Colab記録は `reports/2026-10-27/models/` に収録。
- 元の動画・全候補・フレーム別CSV・失敗画像はローカルの `results/comparison_20261027/<run名>/` に保存。実際のrun名は比較CSVを参照。
- 重み・元動画・仮想環境・大容量のフレームログはGitHubに含めません。GitHubのコードだけで再実行するには、README前半のデータ配置と重み配置が必要です。

### 解釈上の制約

- 同じ83.733秒の1動画を各条件1回測定しています。実行順は固定で、電力状態・温度・OSキャッシュを完全には制御していません。速度差は今回の実測値として報告し、統計的な優位性は主張しません。
- 処理fpsは動画の読込・描画保存・ログ・計測を含み、モデル読込・ウォームアップ・終了後の評価と作図を除きます。動画の再生fpsや撮影から表示までの遅延とは違います。
- 全体fpsが30を超えても各フレームの処理期限を満たす保証ではありません。特にD/Eは検出フレームと中間フレームで時間が異なります。
- GPUメモリ・温度・処理時間の平均／中央値／95パーセンタイルは各条件のmetrics.jsonとtelemetry.jsonに保存しています。
- 連続ロスト指標は位置評価可能な連続区間だけを数え、不可視区間をまたぐ真の再捕捉時間ではありません。
- 15分反復安定性試験、独立した別試合・別体育館での評価、Colab切断からの実optimizer再開は今回未実施です。
- `VERIFICATION.md`と`EVALUATION_B.md`は以前の時点の記録です。今回のs/m比較は本節と2026-10-27のreportsを参照してください。
