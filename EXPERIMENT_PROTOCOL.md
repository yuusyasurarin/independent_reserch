# 2026年10月版：実験仕様と実装上の条件

設計の根拠は利用者指定の「自主研究での実験モデル.pdf」（2026-10-01、14ページ）と
「評価指標.pdf」（8ページ）。元PDFは作業フォルダーの親にあり、ここでは実装との対応を記録する。
資料の数値は設計上の初期値であり、達成済みの性能ではない。

## 資料とコードの対応

| 資料 | 実装 |
| --- | --- |
| モデル設計 pp.3–8、M1〜M6 | `run_tracking_experiment.py`、`experiment_tracking.py`、`experiment_detectors.py` |
| モデル設計 pp.2–3、2段階median・blob・分類CNN | `prepare_experiment_data.py background/extract`、`train_blob_cnn.py` |
| モデル設計 pp.9–10、選手・探索 | `--players P0/P1/P2`、`--search G0/G1/G2/G3` |
| モデル設計 p.11、非同期・元画像への補正 | `--execution realtime --async-detection`、有限履歴からの再追跡 |
| モデル設計 p.12、ring-zone | `--ring-policy R0/R1/R2/R3/R4` |
| 評価指標 pp.3–5、追跡・復帰・時間 | `evaluate_tracking_experiment.py`、`tracking.csv`、`runtime.json` |
| 評価指標 p.6、得点F1と判定不能 | `evaluate_score_events.py` |
| 評価指標 p.2、検出AP | Ultralyticsの`yolo detect val`。追跡評価器のTSRとは別評価 |

## 共通条件

単眼・固定カメラ・単一の対象ボールを前提とする。初期球は最高confidence候補を選択。
M1のみ誤初期化防止に連続2回の整合を要求し、`--confirm-frames 1`で他方式と同条件にできる。
画像座標は左上原点、右がx、下がy、bboxは`x,y,width,height`、frame_indexは0始まり。
KF予測bboxの幅・高さは直近の採用観測を保持する。評価GTの枠サイズを流用しない。
複数球の同一ID保証、3D軌道復元、LSTMモデルの学習はこの追加の対象外。

M2〜M6は同じYOLO重み・画像サイズ・閾値で比較する。M1はResNet-18の3分類を採用した研究版で、
サッカー論文のVGG-M忠実再現ではない。分類入力は107×107、RGB、ImageNet正規化。
モデル変更と追跡方式変更を同じ比較に混ぜない。

M1はblobの矩形を位置として使う。CNNはbboxを回帰しない。Grid fallbackは48px窓・半窓strideの
粗い位置推定であり、ボールの厳密な輪郭を復元しない。分類精度が高くてもIoUが低くなり得る。
`--pure-blob`とfallbackありを分けて報告する。

## 状態とルールの具体的な定義

- M2：処理フレームのYOLO観測のみ出力。間引いたフレームはmissing。ByteTrack・KF・KCFは使わない。
- M3：等速KF。dtは画像時刻差。Qの加速度ノイズ標準偏差1800、YOLO観測分散25、NIS gate 9.21。
- M4：YOLOでKCF初期化、間をKCF更新。運動KFなし。OpenCVのboolを確率として扱わない。
- M5/M6：YOLOがあるフレームはYOLOだけで補正。ないフレームのKCF候補はNISとサイズ比0.25〜4で検証。
  KCF観測分散100。同じ画像でYOLO・KCFの二重correctをしない。
- KFの観測なし出力は既定0.2秒まで。KCFは最後の検出から既定0.5秒まで許す。
  失敗した検出でKCFは破棄する。期限後はgateを解除して再初期化する。
- P0：選手不使用。P1/P2：直近球の拡張矩形と重なる選手の最近傍1人／全候補のunionを探索に利用。
  選手IDはIoUによる簡易対応付け、最大0.5秒保持。球が再観測されたら候補解除。
  保持者の確定ではなく、選手中心を球の観測として出力しない。探索への反映はG3で比較する。
- G0：全画面。G1：Lost時2×2・20%重なりのタイル。G2：Lost時に外周幅25%を優先。
  G2は明示的な実験条件で、画面外に出たと自動断定しない。G3：球の拡張ROIと選手ROI、なければ全画面。
  いずれも既定1秒ごとに全画面fallback。M6はblobなし・候補過多・Lostでもfallback。
- M6：余白付きblob ROIを統合し、既定8個・画像面積60%が上限。crop batchは4。
  Ultralyticsがletterboxを戻したcrop内座標にcrop原点だけを加算し、IoU 0.5で重複除去。
  blob純粋版ではfallback・G/RのROI追加を無効化する。リングは共通の別検出経路を使用。
- R0：切替なし。R1：zone内は観測位置を採用してKF速度を再初期化、観測なし予測は出さない。
  R2：検出時刻を増やさずzone cropを優先。R3：zone内の要求Hzを既定30に増やす。
  R4：R1とR2の併用。zoneは横±1.5リング幅、上2幅・下2幅、退出範囲1.2倍。
  R1/R4の観測優先以外でKF gateを緩めない。

G/Rは設計資料を具体化した初期の比較実装。適応的なROI拡張、外部KFによるKCF検索窓移動、
高度な選手MOTや保持者推定は追加していない。GPU時間の予算を自動的に同一化するschedulerではない。
ROI個数・推論時間・実遅延を実測し、比較条件を調整してから「同一予算」と報告する。
旧`adaptive_basketball_tracker.py`の不確実性による再検出は別の既存方式で、M5固定頻度版とは区別する。

## 非同期と時間計測

通常のofflineは全フレーム逐次処理で、処理能力・精度の比較用。動画再生時刻とホスト時計を混ぜない。
realtime動画は元の時刻に合わせて入力し、1フレーム分以上遅れた画像は捨てる。カメラ時刻は
ホストが画像を受け取った時刻で、センサー露光時刻やドライバー内の待ち時間は分からない。
カメラの`input_frames`は取得できた画像数であり、機器内部の欠落数ではない。

非同期はM3〜M6で利用可。workerは実行中1件＋待機最新1件。待機画像が更新された回数を保存する。
結果のframe_indexの直前状態へ戻り、その画像でYOLO補正・KCF再初期化し、履歴画像を順に再処理する。
KF状態もその時刻へ戻し、後続KCF観測を再計算する。履歴より古い結果、最大再処理数を超える結果は破棄。
既定で履歴30枚・再処理最大15枚。1920×1080カラー30枚だけでも約187MBなので装置に合わせて調整する。
既にCSVへ出力した過去行は修正せず、補正後の状態を次の出力へ使う。未来情報で過去の精度を良く見せない。

`processing_ms`は同期検出または結果取込・replayと追跡を含む。
`latency_ms`は動画の予定到着／カメラ画像受信から位置出力行を作るまで。
offlineではフレーム読込後からの値であり、撮影起点の遅延として報告しない。
録画・GUI・CSV書込みの時間は次フレームや全体wall時間に影響するが、現在行のlatencyには含めない。
`detector_ms`は背景差分、ROI作成、推論、CPUへの結果取得、統合を含む検出jobの実時間。
CUDA結果のCPU取得で同期しているが、GPUカーネル単独時間・GPU使用率ではない。

`runtime.json`のFPSは異なる入力フレームに対する行出力数／実時間（起動・ウォームアップを除く）。
`source=missing`の行もフレーム処理数に含め、位置の有効性はCoverageで別集計する。
System 2の要求頻度と実job数は異なる。`detections.jsonl`で採用・棄却、source frame、age、
YOLO batch数、共通リング検出batch数、ROI数／面積、CNN patch数を確認する。
最後の位置出力後に完了したjobは採用せずログに残す。実Hzはその測定区間を揃えて集計する。
`detector_job_fps`は位置出力の測定区間内に完了したjobだけを分母時間と対応させて集計する。
この1jobにはM6の複数ROIやM1の複数CNNパッチを含み得るため、YOLOの呼出し回数と区別する。

## 評価規則

評価可能集合はGTの`visible`と`partial`。`occluded`と`out_of_frame`の球bboxは空欄とする。
eligibleなフレームの出力なし／CSV行なしはIoU=0、TSR・Coverageでは失敗。
CLE/NCLEは両座標があるフレームだけで計算し件数を併記。NCLEの分母はGTリング幅。
リング相対誤差は予測球・予測リングとGT球・GTリングから別々に計算する。
リング手動指定条件の結果をリング検出精度として報告しない。

AUCは連続積分に等しい平均IoUと、0.01刻みの台形積分を両方出す。
TSRは`IoU >= threshold`。0閾値ではIoU=0も成功となるため、数値積分には微小な差がある。

Lostは連続k=3失敗で1イベント、起点はその失敗列の最初。
復帰は連続m=3成功、時刻はその成功列の最初。完全遮蔽、注釈の欠番、shotの境界では区間を切る。
再可視化後は別イベントとして集計。0.5秒の追跡可能な観測期間が残らないイベントは、
早期復帰していても復帰率の分母から除く厳格な窓規則を採用し、censoredとして報告する。
復帰時間は実際に復帰を観測した例を集計し、未復帰を0秒にしない。

通常・高速・部分遮蔽・ring-zone・小球別、shot別とshotマクロ平均を出力。
zone評価はGTリングと固定係数を使用。高速／小球は注釈前に閾値を決めてGT列に記録する。
95%区間はshot bootstrap（seed42、1000回）。同一session内の相関が強い最終研究報告では、
session単位での再集計・bootstrapも行う。独立性がないフレームbootstrapは使わない。
APは既存の検出評価器を使用し、ball/ringを個別報告する。

Score評価は全シュートを入力し、追跡失敗も`unknown`として残す。成功のunknownは成功F1のFN、
失敗のunknownは別件数として記録し、Accuracyではどちらも不正解。
Macro-F1では失敗側unknownも失敗クラスのFNとする。存在しないクラスのF1はN/A。

## 検証方法と残る実験

`python -m unittest discover -p "test*.py" -v`で方式の違い、予測期限、ROI座標、
過去画像replay、欠測・遮蔽・復帰打切り、候補Recall、ラベル確認を検証する。
合成動画と既存YOLO重みでCLI入出力の動作確認も行う。
これらは実映像精度・CNN学習結果・60fps達成・Score F1を保証しない。
CNNとball/player/ring検出器には撮影条件に合う注釈・学習・未使用テスト評価が必要。

API確認先：
[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/)、
[Ultralytics Detection Dataset](https://docs.ultralytics.com/datasets/detect/)、
[Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。
