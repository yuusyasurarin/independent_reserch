# 当日の手順（2026-11-26 体育館）

PowerShellで実行します。追跡は `basketball_tracking_midterm` フォルダ、撮影はリポジトリ直下（`independent_reserch`）で実行します。

## 持ち物・出発前

- [ ] PCを100%まで充電（体育館では電源なしで動かす）
- [ ] カメラ（魚眼）、USBケーブル、三脚
- [ ] チェッカーボード（魚眼の補正用。内側の角が横9×縦6、マス25mm）
- [ ] 空き容量：生映像と結果映像で1時間あたり約3GB（30秒で約30MB×2本から計算）

## 手順1：カメラの確認（最初に1回）

```powershell
cd D:\自主研究\independent_reserch
.\.venv\Scripts\python.exe camera_probe.py
```

- [ ] 魚眼カメラの番号を確認する。**このPCの内蔵カメラが番号0なので、魚眼カメラは1以降のはず**
- [ ] 1280×720で実測30fps近く出るか確認する。出ない場合は、30fps近く出た解像度を使う（追跡は1280×720で学習・確認しているので、できるだけ1280×720）

以下、魚眼カメラの番号を `1` として書きます。違う場合は `--camera-index` / `--cameras` を変えてください。

## 手順2：魚眼の補正用の撮影（追跡の前に5分）

```powershell
cd D:\自主研究\independent_reserch
.\.venv\Scripts\python.exe camera_video_recorder.py --camera-index 1 --width 1280 --height 720 --fps 30 --output results\recordings\fisheye_checkerboard.mp4
```

- [ ] チェッカーボードを、画面の中央・四隅・端まで、傾けながら1〜2分映す（Qで終了）
- [ ] 試合と同じカメラ・同じ解像度で撮る（解像度を変えると補正値が使えない）

補正値の計算はあとでできます（`scripts/fisheye_calibrate.py --video ...`）。

## 手順3：バッテリーでの速度確認（試合前に3分、電源を抜いた状態で）

録画ファイルを30fpsのカメラ入力として流し、モデルA・Bが追いつくかを見ます。

```powershell
cd D:\自主研究\independent_reserch\basketball_tracking_midterm
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --video "../編集済み元データ/experiment10_編集.mp4" --duration 60 --no-display --record results/gym_20261126/check_A
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --video "../編集済み元データ/experiment10_編集.mp4" --duration 60 --no-display --record results/gym_20261126/check_B --roi-imgsz 320 --max-interval 1 --max-yolo-hz 30
```

最後に表示される `tracking:` の行の `drop_rate`（間引き率）を見ます。

- [ ] 表示に `power=battery` と出ていることを確認
- [ ] `drop_rate` が0.1（10%）を超えたら、そのモデルは入力サイズを下げて（`--imgsz 704` を付けて）もう一度確認する
- [ ] 704でも大きく間引く場合は、そのモデルは704で実施し、間引き率を記録しておく

## 手順4：リアルタイム追跡（ボールの追跡のみ）

カメラを三脚で固定し、コート全体とリングが映るようにします。表示は3秒遅れです。QまたはEscで終了します。

```powershell
cd D:\自主研究\independent_reserch\basketball_tracking_midterm
# モデルA：条件G
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --camera-index 1 --record results/gym_20261126/A --record-raw
# モデルB：G＋ROI、上限30回/秒
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --camera-index 1 --record results/gym_20261126/B --record-raw --roi-imgsz 320 --max-interval 1 --max-yolo-hz 30
```

手順3で704にすると決めたモデルには `--imgsz 704` を付けます。

- [ ] 画面左上の `dropped`（間引かれたフレーム）が増え続けていないか見る
- [ ] `--record-raw` で生映像（`raw.mp4`）を残す。あとで正解を付けて精度を測るのに必要
- [ ] A・Bは同じ場面を同時には測れないので、交互に同じくらいの時間ずつ行い、何時何分にどちらを動かしたかをメモする

## 手順5：リアルタイム追跡＋シュート判定

```powershell
cd D:\自主研究\independent_reserch\basketball_tracking_midterm
.\.venv\Scripts\python.exe scripts/realtime_tracking.py --camera-index 1 --record results/gym_20261126/shot_A --record-raw --select-rim
```

1. 最初の画面で、**リングの輪をぴったり囲む四角**をマウスでドラッグし、Enterを押す
2. 画面に `rim = x1,y1,x2,y2` と表示されるので、メモする。次からは `--select-rim` の代わりに `--rim x1,y1,x2,y2` で同じ位置を使える（カメラを動かしていなければ）
3. シュートのたびに、3秒遅れで `SHOT MADE` / `SHOT MISS` と累計が表示される

モデルBでシュート判定をする場合は、上のコマンドに `--roi-imgsz 320 --max-interval 1 --max-yolo-hz 30` を付けます。

- [ ] **リングを指定したあとはカメラを動かさない**（動いたらやり直す）
- [ ] リングを横から見る位置にカメラを置くと判定しやすい（手前・奥の区別ができないため）
- [ ] 実際の結果（成功・失敗）を紙やスマホで記録しておく。あとで判定と照らし合わせる

## 手順6：撮影のみ（追跡なし）

```powershell
cd D:\自主研究\independent_reserch
# 1台
.\.venv\Scripts\python.exe camera_video_recorder.py --camera-index 1 --width 1280 --height 720 --fps 30
# 2台（魚眼2台でステレオにする場合）。開始直後に拍手してMキー
.\.venv\Scripts\python.exe dual_camera_recorder.py --cameras 1 2 --width 1280 --height 720 --fps 30 --output results\recordings\game01
```

- [ ] 2台のときは、拍手やスマホのフラッシュの瞬間にMキーを押す（あとで2本の動画を合わせるため）

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `Cannot open camera index` | 番号を変える（手順1）。カメラアプリやZoomなど、ほかのアプリがカメラを使っていないか確認 |
| 映像が真っ黒・極端に暗い | USBを挿し直す。別のUSBポートを試す |
| 1280×720にならない | 手順1で出た解像度を `--width --height` に指定。追跡の精度は1280×720以外では未確認 |
| 実測fpsが30より低い | 照明が暗いとカメラが露出時間を延ばしてfpsが落ちることがある。明るい場所を映すか、640×480を試す |
| `dropped` がどんどん増える | `--imgsz 704` を付ける。それでも増えるならモデルAに切り替える |
| リング指定の画面が出ない | `--no-display` を付けていないか確認（シュート判定には画面が必要） |
