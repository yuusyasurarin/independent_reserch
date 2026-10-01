# 不確実性・リング位置に応じた適応的ボール追跡

固定カメラ・単一ボールの録画動画を対象に、YOLO、KCF、等速カルマンフィルタを
状況に応じて切り替える研究用実装。提案の検証対象はアルゴリズムの単純な併用ではなく、
**いつ再検出し、いつ運動予測への依存を下げると、精度と計算量がどう変わるか**である。
先行研究全体に対する新規性や実映像での優位性は、まだ確認していない。
得点判定・保持選手のID推定・オンライン学習は今回の対象に含まない。

## 実装した切り替え

| 状況 | 処理 | 検証したい効果 |
| --- | --- | --- |
| 通常・低不確実性 | KCF観測をカルマンで補正。既定6フレームごとにYOLOで検証 | 毎フレーム検出に対する計算量削減 |
| 不確実性増大 | 予定を待たずYOLO再検出 | KCFのドリフトの長期化を抑制 |
| KCF失敗／予測との大きな不一致 | YOLO再検出、失敗したKCFを破棄 | 誤追跡の連鎖を抑制 |
| 予測ボール矩形の30%以上が直近の人物矩形と重なる | 遮蔽疑いとしてKCF更新を止め、YOLO再検出 | 人物の模様を学習してしまう問題の抑制 |
| 観測できない | 最大12フレームだけ予測座標、以後は失跡 | 短い欠測の補完、長い誤予測の抑制 |
| リング付近 | 毎フレームYOLO。観測座標をそのまま出力し、通常の運動ゲートを緩和 | リバウンド時の検出拒否・平滑化による遅れの抑制 |
| リング付近で観測なし | 予測だけの座標を出力しない | 誤ったリング通過軌跡の生成を抑制 |
| 失跡後 | 全画面YOLOから再初期化 | 再捕捉 |

YOLO実行フレームで検出に失敗した場合、KCFの結果へ戻さず欠測とする。
YOLOで補正したフレームでKCFを再初期化する。遮蔽は2次元の重なりによる**疑い**であり、
実際の前後関係や保持者を断定しない。人物矩形は最大 `detect_interval` フレームだけ有効。
人物クラスのない独自モデルでは人物重なりの手掛かりが無効になり、起動時に表示する。

## 不確実性・位置関係の定義

状態は `[x, y, vx, vy]`、時間刻みは動画FPSの逆数。単位はピクセル、秒。
等速モデルに白色加速度ノイズを加える。重力をモデル化した放物線予測ではない。
位置共分散 `Pxy` の最大固有値から `sigma = sqrt(lambda_max(Pxy))` を求める。

```text
U = sigma / max(ball_width, ball_height, 1)
    + frames_since_YOLO / (2 * detect_interval)
```

`U >= 0.8` で再検出。Uは確率ではなく、共分散と検出からの経過時間を組み合わせた
制御用スコアである。KCFの成功フラグを信頼度とみなさない。OpenCV KCFにはこの実装で
使用できる連続的な追跡信頼度を求めず、観測残差のNISと再検出間隔を代用する。
NISは `residual.T @ inv(Pxy + R) @ residual`、通常ゲートは9.21。
YOLO観測の分散は25、KCF観測は100、加速度ノイズの標準偏差は1800。
これらは初期仮定であり、実測による校正が必要。

リングは元動画上の固定矩形 `(x, y, width, height)` を手動指定する。
ボール中心とリング中心の距離をリング幅で割り、1.5未満でリングモードへ入り、
2.0以上で解除する。予測位置と直近観測位置のうち近い方を使う。
リングモード中のYOLO候補は直近観測位置からリング幅の2倍以内に限定し、
距離と検出スコアで選ぶ。大きいNISの観測を受け入れた場合は速度をゼロへ再初期化する。
リング対応は接触検出ではなく、画像上の近接を利用した制御である。

## 実行

必要なパッケージは `numpy`, `opencv-contrib-python`, `ultralytics`, `torch`。
動作する研究用Python環境を有効化してから実行する。OpenCVの通常版とcontrib版が
同一環境に混在している場合は、KCF生成APIが利用できるか確認する。
`requirements.txt` の参照不整合は修正済み。環境再構築はREADMEの共通準備を参照。
2026年10月資料のM1〜M6比較には、新しい `run_tracking_experiment.py` を使用する。
本ファイルの適応的追跡は従来方式として残しており、M5の固定頻度版とは区別する。

```powershell
cd D:\自主研究\independent_reserch
python adaptive_basketball_tracker.py --help

# 数値は例。実映像のリング矩形へ必ず置き換える。
python adaptive_basketball_tracker.py --video sample.mp4 --ring 820 180 90 25 --mode adaptive --output results/adaptive/run01 --record

# 同一映像・検出器・閾値による比較。出力先は未作成の名前を指定する。
foreach ($method in @('yolo', 'fixed', 'adaptive', 'no-ring', 'no-uncertainty')) {
    python adaptive_basketball_tracker.py --video sample.mp4 --ring 820 180 90 25 --mode $method --no-preview --output "results/adaptive/$method"
}

python -m unittest discover -p test_adaptive_tracking.py -v
```

`--class-id` と `--person-class-id` で独自モデルのクラスを指定できる。
省略時はball系の名称とperson/playerを検索する。YOLO重みの既定は既存と同じ `yolo26n.pt`。
`--model`, `--device`, `--confidence`, `--image-size` も変更可能。
本実装は動画を全フレーム順次処理する同期構成であり、60fpsの実時間動作を保証しない。
表示時はQまたはEscで終了。動画FPSを取得できない場合は `--fps` が必要。
録画には全処理フレームを元動画のFPSで保存する。

各条件の意味:

- `yolo`: 毎フレームYOLOの最高信頼度ボールを出力。運動ゲート・予測補完を使用しない。
- `fixed`: YOLOを固定間隔で実行し、間をKCF＋カルマンで追跡。初期化・完全失跡時は即検出。
- `adaptive`: すべての適応的切り替えを使用。
- `no-ring`: 提案方式からリング近傍の特別処理だけを除く。
- `no-uncertainty`: 提案方式からUによる再検出だけを除く。KCF失敗・遮蔽・リングによる切り替えは残る。

比較対象が単一ボールであることを確認する。初期化／完全失跡後は最高信頼度のボールを
選ぶため、複数ボール映像では同一対象を保証しない。既存KCF・ByteTrackとの比較を行う場合は
開始フレーム・出力形式・対象ボールの選び方も揃える。

## 出力と評価

実験ディレクトリには `tracking.csv`, `config.json` と任意の `tracking.mp4` を保存。
CSVは0始まりのフレーム番号、映像時刻、状態、切り替え理由、YOLO実行有無、
U、NIS、人物重なりの疑い、観測／予測／欠測の区別、中心座標、リング相対座標を含む。
`switched` は出力状態の変化であり、初期フレームも1。`reason` はそのフレームで
優先された検出トリガー。複数の条件が同時成立する場合もある。
`config.json` に設定、入力動画と重みのSHA256、ライブラリ版を保存する。

正解CSVを元動画に対して作成する（この例の座標は書式説明用で、実験データではない）。
全フレームを注釈すると復帰時間・連続追跡を評価できる。座標を判断できない完全遮蔽は
空欄にし、見えていない位置を推測で正解にしない。

```csv
frame_index,center_x,center_y,visible,occluded,near_ring
0,100,200,1,0,0
1,,,0,1,0
2,130,180,1,0,1
```

`visible`, `occluded`, `near_ring` は0/1で指定する。
部分遮蔽で中心が特定できる場合は `visible=1, occluded=1` として座標を付ける。
near_ringは方式の出力ではなく、元映像から人手で一貫して付ける評価区間ラベル。

```powershell
python evaluate_adaptive_tracking.py --ground-truth annotations.csv --predictions results/adaptive/yolo/tracking.csv results/adaptive/fixed/tracking.csv results/adaptive/adaptive/tracking.csv results/adaptive/no-ring/tracking.csv results/adaptive/no-uncertainty/tracking.csv --threshold-px 20 --output results/adaptive/comparison.json
```

通常／遮蔽／リング付近／全体それぞれについて以下を集計する。

- 正解中心から閾値以内の成功率。予測出力も含む値と、観測を伴う値を別々に表示。
- 出力被覆率、平均中心誤差、RMSE。誤差は正解と出力の両方があるフレームだけなので、
  欠測も失敗に含める成功率と併記する。完全遮蔽の座標空欄は精度の分母に含めない。
- 失敗イベント数、最長連続成功、遮蔽解除から正しい観測へ復帰するフレーム数。
  復帰前の動画終了・再遮蔽・注釈欠落は打ち切りとして別集計する。
- 検出実行回数、追跡処理FPS。ウォームアップ・読込・描画・保存を除外した時間のため、
  カメラ入力から表示までの実効FPSではない。全方式で同じ計測範囲を使う。

正解があるのに予測CSVに行がないフレームも失敗に含め、途中終了を有利にしない。
normal、occluded、near_ringは同じ基準で比較し、遮蔽とリングの区間は重なり得る。
評価は単一物体の中心追跡であり、MOTA/IDF1ではない。

## 研究としての検証手順

1. 遮蔽、リングとの接触・跳ね返り、通常移動を含む複数の動画を撮影し、動画単位で調整用と評価用に分ける。
2. 調整用動画でU閾値、検出間隔、リング領域、予測期限を決め、評価用では固定する。
3. 同一モデル、解像度、デバイス、正解、成功閾値で5条件を実行する。
4. adaptive対fixedで切り替え全体、adaptive対no-ringでリング対応、adaptive対no-uncertaintyでUの効果を調べる。
5. 動画ごとの成功率差、復帰時間差、YOLO回数差を報告し、複数回の処理速度とそのばらつきも測る。
6. `reason` と失敗フレームを照合し、誤検出・人物矩形の古さ・リング領域の過大設定など、改善しなかった条件も記録する。

現段階では実映像での改善率は未測定。単体テストは切り替えロジックの検証であり、
実際の遮蔽検出精度、リバウンド追跡性能、研究上の新規性を証明するものではない。

API確認に使用した公式資料:
[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/)、
[OpenCV KCF](https://docs.opencv.org/doc/doxygen/html/d5/d6a/classcv_1_1legacy_1_1TrackerKCF.html)。
