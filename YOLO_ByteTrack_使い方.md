# YOLO＋ByteTrack 実験の使い方

プロジェクトの `independent_reserch` フォルダーでPowerShellを開く。

## 準備（初回）

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. カメラで撮影しながら追跡

```powershell
.\.venv\Scripts\python.exe yolo_bytetrack_live.py --camera-index 1
```

カメラ番号は環境に合わせて0、1、2などに変更する。GPUは利用可能なら自動選択。
`Q`またはEscで終了、`S`で追跡画面の静止画を保存。

## 2. 録画済み動画を追跡

```powershell
.\.venv\Scripts\python.exe yolo_bytetrack_video.py --video "captures/sample.mp4"
```

動画は先頭から順番に処理し、末尾で終了する。実時間の再生速度には合わせない。
表示不要なら `--no-display` を追加する。

## 保存結果

実行ごとに `results/yolo/live_日時/` または `results/yolo/video_日時/` が作成される。

- `tracking.csv`：フレーム番号、検出有無、ID、信頼度、中心座標、矩形、処理時間、入力時刻。
- `tracked.mp4`：矩形・ID・軌跡付きの動画。
- `raw.mp4`：描画前の取得映像（カメラ用のみ。MP4へ再圧縮）。録画動画用では元ファイルを使う。
- `snapshots/`：Sキーで保存した静止画。

CSVの座標は元画像の左上が原点で、右向きがx、下向きがy（ピクセル）。
複数候補があれば1フレームに複数行、未検出でも1行を出力する。ID=-1は未確定。
`source_seconds`は動画内時刻（取得できない場合はフレーム番号とFPSから算出）、
カメラでは処理開始からフレーム取得までの経過秒であり露光時刻ではない。
`elapsed_seconds`は推論後の処理経過時間。両者を混同しない。

## よく使うオプション

```powershell
# 小さいボール：入力画像を大きくする（処理は重くなる）
.\.venv\Scripts\python.exe yolo_bytetrack_video.py --video "captures/sample.mp4" --image-size 1280

# 独自学習モデル
.\.venv\Scripts\python.exe yolo_bytetrack_live.py --camera-index 1 --model "runs/detect/train/weights/best.pt"
```

`--device cpu`でCPU、`--device 0`でGPU指定。`--max-frames 100`で100フレームだけ処理。
`--width 1920 --height 1080 --fps 60`はカメラへの要求値で、実際の動作は機器に依存する。
検出閾値の既定は0.10で、ByteTrackの低信頼度検出の対応付けに渡す候補を残す。

## 実験上の注意

- 標準モデルはCOCOのsports ballを対象とし、バスケットボール専用学習モデルではない。
- 検出フレーム率は正解率ではない。精度評価には正解座標との比較が必要。
- カメラ版は取得→推論→保存の順次処理。60fpsでの全フレーム録画は保証しない。
  保存動画はカメラ申告FPSの固定フレームレートのため、処理が遅い場合は実時間より短く再生される。
  実時間の解析にはCSVのsource_secondsを使う。一定FPSの比較実験には別途録画した動画を使う。
- 手動ROI指定は不要。完全遮蔽中の真の位置を復元するものではなく、失跡後にIDが変わる場合がある。
- 学習済みモデルを変更する場合は、比較条件間でモデル・入力サイズ・閾値を記録してそろえる。
