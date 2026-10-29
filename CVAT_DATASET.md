# experiment11 / experiment12 のCVAT用画像

> 画像抽出・アノテーション準備時の記録です。現在はballのみの408枚のデータセットが完成しており、学習・評価は [basketball_tracking_midterm](basketball_tracking_midterm/README.md) を使います。以下の「空」「未完成」は準備当時の状態です。画像・注釈・配布ZIPはGitHubに含めません。

## 作成したデータ

保存先は `datasets/cvat_experiment11_12/`。元動画は変更しません。

- `images/train/`：学習用
- `images/eval/`：学習中の検証・チューニング用（YOLOの `val` に対応）
- `images/test/`：モデル・閾値を確定した後の最終評価用
- `cvat_upload/train.zip`, `eval.zip`, `test.zip`：CVATに投入する画像だけのZIP
- `manifest.csv`：各画像の元動画、0始まりフレーム番号、秒数、分割、画像ハッシュ
- `preparation.json`：元動画のハッシュ、分割区間、抽出条件、件数
- `preview_*.jpg`：分割別の確認用一覧。アノテーション対象には含めません。
- `labels/{train,eval,test}/`：アノテーション結果の配置先。現時点では空です。
- `data.yaml.template`：クラス名の確定後に使うYOLO設定のひな形

各動画の前半70%をtrain、続く15%をeval、最後の15%をtestとし、
各境界の前後2秒を除外して毎秒2枚抽出します。比率は除外前の時間比率なので、
実際の画像枚数の比率とは異なります。画像は1280×720のままJPEG品質95で保存します。
同一画素のフレームは全分割を通して重複排除し、除外履歴を残します。
ぼけ、遮蔽、小さいボール、ボールが写らない画像も自動除外しません。

これは時間帯を分離した同一撮影条件の評価です。分割境界をまたぐプレーや
編集動画中のリプレイ・類似場面までは自動識別しません。CVAT作業前に一覧と動画を確認し、
同じプレーの再掲が別分割に存在したら、該当プレーを同じ分割へまとめてください。
2本しかないため3分割を動画単位で独立させることはできません。
別試合への性能を報告する場合は、追加の未使用試合をtestとして確保してください。

## CVATでの作業

1. 対象クラスを確定し、共通のプロジェクトに同じラベル定義で3タスクを作ります。
   既存研究に合わせるなら `ball`, `player`, `ring` が候補ですが、ボールのみなら `ball` のみで構いません。
2. それぞれ `train.zip` / `eval.zip` / `test.zip` を画像データとしてアップロードします。
   これはアノテーション済みデータのインポート用ZIPではありません。
3. 矩形でアノテーションします。小さいボールは拡大して確認し、見えている対象を漏れなく囲みます。
   遮蔽・画面端・リングの囲み方は作業開始時に統一します。
4. 全画像を確認してYOLO検出形式でエクスポートします。ラベルのクラス番号と順番を全タスクで揃えます。
5. 同じ画像名の `.txt` を、元の分割どおり `labels/train/`, `labels/eval/`, `labels/test/` に配置します。
   画像名を変更しないでください。対象なしと確認した画像だけ空ラベルにし、未確認画像を背景扱いしないでください。
6. `data.yaml.template` を `data.yaml` にコピーし、`names` にエクスポートと同じ番号・クラス名を記入します。
   `val: images/eval` は意図した設定です。全ラベルを確認後に学習に利用してください。

testの結果を見て設定を調整すると最終評価用ではなくなるので、チューニングにはevalを使います。
画像だけではまだYOLOの学習データとして完成していません。

公式の形式説明：
- https://docs.cvat.ai/docs/dataset_management/export-datasets/
- https://docs.ultralytics.com/datasets/detect/

## 再実行

OpenCVとNumPyが使えるPython環境で、`independent_reserch` を作業フォルダーにして実行します。

```powershell
python prepare_cvat_dataset.py
```

既存の出力先には上書きしません。別条件では別の出力先を指定してください。

```powershell
python prepare_cvat_dataset.py --sample-fps 4 --guard-seconds 2 --output datasets/cvat_experiment11_12_4fps
```

今回のPCでは既存 `.venv` のPython本体参照が無効だったため、インストール済みPython 3.12から
既存 `.venv` 内のOpenCV・NumPyを読み込んで実行しました。同じPCでの再実行例：

```powershell
py -3.12 -c "import sys,runpy; sys.path.insert(0,r'.venv\Lib\site-packages'); runpy.run_path('prepare_cvat_dataset.py',run_name='__main__')" --output datasets/cvat_experiment11_12_v2
```

途中で失敗した場合は `preparation.json` の `status: complete_unlabeled` が存在するかを確認してください。
完了していない出力先には追記せず、新しい出力先を指定してやり直します。
フレーム時刻はこの素材の30fpsを前提にフレーム番号/FPSで記録します。
