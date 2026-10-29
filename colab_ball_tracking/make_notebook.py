"""Generate a self-contained Japanese Colab workflow (no training cells)."""
import json
from pathlib import Path
import textwrap

cells=[]


def add(kind,source):
    cell=dict(cell_type=kind,metadata={},source=textwrap.dedent(source).strip().splitlines(keepends=True),id=f'track-{len(cells):03d}')
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown', '''
# experiment09／10：YOLO＋ByteTrackの追跡テスト

> 旧Ultralytics版の記録です。現在のYOLOv5実験A〜Gはリポジトリの basketball_tracking_midterm/README.md を参照してください。データ・重みは別途必要です。experiment10は現在、開発評価用です。

追加学習済みの **best.pt** を使い、学習に使っていない動画を評価します。再学習はしません。

1. 作成済みの `ball_tracking_experiment10.zip` を **マイドライブ / ball_tracking_test /** にアップロード。
2. 「ランタイム → ランタイムのタイプを変更」でGPUを選択。
3. 上から実行し、モデル選択セルの `BEST_PATH` を設定。

動画・正解ラベルはZIPに同梱しています。experiment10は全2,512フレーム中1,883フレームにball矩形があります。
ラベルのない629フレームは、ユーザー確認済みの不可視区間です（2026-10-26確認）。ZIP内のmanifestに記録しています。リングのラベルは不要です。
experiment09を再実行する場合は、次の `EXPERIMENT` を `"experiment09"` に変更し、09用ZIPを使用します。
ダウンロード元のアノテーションZIPと配布ZIPは同名です。アップロードするのは、このフォルダで作成した動画入りの配布ZIPです。
''')
add('code', '''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import json, zipfile, shutil, subprocess, sys

DRIVE = Path('/content/drive/MyDrive')
PROJECT = DRIVE / 'ball_tracking_test'
EXPERIMENT = 'experiment10'  # 'experiment09' に切り替え可能
if EXPERIMENT not in ('experiment09', 'experiment10'):
    raise ValueError('EXPERIMENTはexperiment09またはexperiment10を指定してください')
ZIP_PATH = PROJECT / f'ball_tracking_{EXPERIMENT}.zip'
if not ZIP_PATH.is_file():
    raise FileNotFoundError(f'ZIPをアップロードしてください: {ZIP_PATH}')

# 再実行しても古い展開内容を使わないよう、新しい作業フォルダへ展開します。
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
TEMP = Path('/content') / (EXPERIMENT + '_' + STAMP)
TEMP.mkdir(exist_ok=False)
with zipfile.ZipFile(ZIP_PATH) as archive:
    if 'ball_tracking_colab/manifest.json' not in archive.namelist():
        raise ValueError('これはアノテーションのみのZIPです。動画・実行コード入りの配布ZIPをアップロードしてください。')
    for item in archive.infolist():
        destination = (TEMP / item.filename).resolve()
        if not destination.is_relative_to(TEMP.resolve()) or not item.filename.startswith('ball_tracking_colab/'):
            raise ValueError('不正なZIPパスです')
        if (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('ZIP内のシンボリックリンクは扱いません')
    archive.extractall(TEMP)
WORK = TEMP / 'ball_tracking_colab'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(WORK/'requirements-colab.txt')], check=True)
import torch, ultralytics
print('torch:', torch.__version__, 'ultralytics:', ultralytics.__version__)
assert torch.cuda.is_available(), 'ColabのGPUランタイムを選択してください'
print('GPU:', torch.cuda.get_device_name(0))
print('作業先:', WORK)
''')
add('markdown', '''
## 正解データとコードの検査

元動画とアノテーションのSHA-256、フレーム対応を検査します。緑枠が正解です。
ラベルなし区間は、確認済みならnot_visible、それ以外はunknownとして位置精度の評価から除外します。
''')
add('code', '''
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(WORK), '-p', 'test_*.py', '-v'], check=True)
sys.path.insert(0, str(WORK))
from tracking_eval import sha256, load_annotations
manifest = json.loads((WORK/'manifest.json').read_text(encoding='utf-8'))
assert Path(manifest['video']['file']).stem == EXPERIMENT, '選択した実験とZIP内の動画が異なります'
for key in ('video', 'annotations'):
    entry = manifest[key]
    assert sha256(WORK/entry['file']) == entry['sha256'], f'{key}のハッシュが一致しません'
v = manifest['video']
gt, annotation_report = load_annotations(WORK/manifest['annotations']['file'], v['frame_count'], v['width'], v['height'], manifest['annotations']['missing_policy'])
print(json.dumps(annotation_report, ensure_ascii=False, indent=2))
print('ラベルなし区間の扱い:', manifest['annotations']['confirmation'])
from IPython.display import display, Image, Video, FileLink
display(Image(filename=str(WORK/'annotation_preview.jpg')))
''')
add('markdown', '''
## 使用する学習済みモデル・設定

`BEST_PATH` に、検証データで採用したモデルのパスを指定します。
例：`/content/drive/MyDrive/ball_yolo/runs/ball_実際の学習実行名/weights/best.pt`

空欄なら候補を表示します。候補が複数ある場合は停止するので、表示されたパスを指定してこのセルを再実行してください。
学習したときのDrive保存先にモデルがあれば、PCへのダウンロード・再アップロードは不要です。

入力サイズ・信頼度・ByteTrack設定は検証時に決めた値に合わせてください。
初期値は既存追跡コードの値です。test結果を見て最適化した場合は、独立した最終評価と区別します。
experiment09の結果フォルダが残っている場合、`REFERENCE_RUN` にそのパスを指定すると、
入力サイズ・閾値・FP16を引き継ぎ、モデル・ByteTrack設定・Ultralyticsの一致を確認できます。
''')
add('code', '''
BEST_PATH = ""  # 採用したbest.ptのフルパス。候補が1つだけの場合は空欄でも使用できます。
IMGSZ = 960
CONF = 0.10
NMS_IOU = 0.5
EVAL_IOU = 0.5
FP16 = False
REFERENCE_RUN = ""  # 例: /content/drive/MyDrive/ball_tracking_test/results/experiment09_日時

reference = None
if REFERENCE_RUN.strip():
    reference_dir = Path(REFERENCE_RUN.strip())
    reference = json.loads((reference_dir/'run_config.json').read_text(encoding='utf-8'))
    assert reference['complete_video'], '全フレームを評価したexperiment09の結果を指定してください'
    assert Path(reference['video']['file']).stem == 'experiment09', 'experiment09の結果を指定してください'
    settings = reference['arguments']
    assert settings['class_id'] == 0, '単一クラスballモデルの結果を指定してください'
    IMGSZ, CONF = settings['imgsz'], settings['conf']
    NMS_IOU, EVAL_IOU, FP16 = settings['nms_iou'], settings['eval_iou'], settings['fp16']
    if not BEST_PATH.strip() and (reference_dir/'selected_model_source.txt').is_file():
        BEST_PATH = (reference_dir/'selected_model_source.txt').read_text(encoding='utf-8').strip()
    assert sha256(WORK/'bytetrack.yaml') == reference['tracker_sha256'], 'ByteTrack設定が09と異なります'
    assert ultralytics.__version__ == reference['versions']['ultralytics'], 'Ultralyticsのバージョンが09と異なります'

if BEST_PATH.strip():
    BEST = Path(BEST_PATH.strip())
else:
    candidates = set()
    for direct in (PROJECT/'best.pt', DRIVE/'colab_ball_yolo/best.pt', DRIVE/'ball_yolo/best.pt'):
        if direct.is_file(): candidates.add(direct)
    for runs in (DRIVE/'ball_yolo/runs', DRIVE/'colab_ball_yolo/runs'):
        if runs.is_dir(): candidates.update(runs.glob('*/weights/best.pt'))
    candidates = sorted(candidates)
    print('モデル候補:')
    for candidate in candidates: print(candidate)
    if len(candidates) != 1:
        raise ValueError('BEST_PATHを指定してください。候補がない場合はbest.ptをMyDrive/ball_tracking_test/へアップロードしても使えます。')
    BEST = candidates[0]
if not BEST.is_file(): raise FileNotFoundError(BEST)
LOCAL_MODEL = WORK/'best.pt'
shutil.copy2(BEST, LOCAL_MODEL)
if reference is not None:
    assert sha256(LOCAL_MODEL) == reference['model_sha256'], 'モデルの重みがexperiment09と異なります'
print('使用モデル:', BEST)
print('モデルSHA-256:', sha256(LOCAL_MODEL))
print((WORK/'bytetrack.yaml').read_text())
print(dict(imgsz=IMGSZ, conf=CONF, nms_iou=NMS_IOU, evaluation_iou=EVAL_IOU, fp16=FP16))
''')
add('markdown', '''
## 選択した動画の全フレームを追跡・評価

ByteTrackを新しく初期化して実行します。追跡IDが未確定の検出は位置出力に数えません。
複数候補は直前のIDを優先し、なければ最高信頼度のID確定候補を選びます。正解を見て候補を選ぶ処理はありません。
''')
add('code', '''
RUN_NAME = EXPERIMENT + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR = PROJECT/'results'/RUN_NAME
command = [sys.executable, str(WORK/'run_tracking_test.py'), '--bundle', str(WORK),
           '--model', str(LOCAL_MODEL), '--output', str(RUN_DIR), '--device', '0',
           '--imgsz', str(IMGSZ), '--conf', str(CONF), '--nms-iou', str(NMS_IOU), '--eval-iou', str(EVAL_IOU)]
if FP16: command.append('--fp16')
print('結果保存先:', RUN_DIR)
subprocess.run(command, check=True)
(RUN_DIR/'selected_model_source.txt').write_text(str(BEST)+'\\n', encoding='utf-8')
if reference is not None:
    (RUN_DIR/'reference_run_config.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding='utf-8')
    (RUN_DIR/'reference_run_source.txt').write_text(REFERENCE_RUN, encoding='utf-8')
''')
add('markdown', '''
## 数値と時系列を確認

TSR・平均IoUの分母は正解矩形のあるフレームです（09：447、10：1,883）。追跡出力なしは失敗・IoU=0です。
CLEは出力があるフレームだけの位置誤差なので、件数とCoverageも合わせて確認します。
不可視区間は位置精度から除外し、出力数を別表示します。単純に誤検出とは断定しません。
リング正規化誤差・IDF1・HOTA・シュート単位の復帰率は、このラベルからは算出しません。
''')
add('code', '''
import pandas as pd
import matplotlib.pyplot as plt
metrics = json.loads((RUN_DIR/'metrics.json').read_text(encoding='utf-8'))
table = [
    ('評価フレーム数', metrics['evaluated_bbox_frames']),
    ('TSR（設定IoU閾値以上）', metrics['tsr']),
    ('TSR（IoU≥0.25）', metrics['tsr_iou_025']),
    ('平均IoU', metrics['mean_iou']),
    ('出力Coverage', metrics['output_coverage']),
    ('CLE平均（px）', metrics['cle_px']['mean']),
    ('CLE計算フレーム数', metrics['cle_px']['n']),
    ('追跡出力なし（正解あり区間）', metrics['missing_output_on_bbox_frames']),
    ('確認済み不可視フレーム数', metrics['not_visible_frames']),
    ('未確認フレーム数（位置評価から除外）', metrics['unknown_frames']),
    ('不可視区間の出力数', metrics['output_on_not_visible_frames']),
    ('3フレーム以上の連続失敗回数', metrics['failure_runs_ge3']),
    ('最長連続失敗（秒）', metrics['longest_failure_duration_s']),
    ('追跡処理のみFPS', metrics['track_only_fps']),
    ('描画・保存込みFPS', metrics['wall_fps']),
]
display(pd.DataFrame(table, columns=['指標', '値']))
per_frame = pd.read_csv(RUN_DIR/'per_frame.csv')
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(per_frame.timestamp_s, per_frame.iou)
axes[0].axhline(EVAL_IOU, color='red', linestyle='--', label='success threshold')
axes[0].set(ylabel='IoU', ylim=(-.03,1.03)); axes[0].legend()
axes[1].plot(per_frame.timestamp_s, per_frame.cle_px)
axes[1].set(xlabel='Video time (s)', ylabel='Center error (px)')
fig.tight_layout(); fig.savefig(RUN_DIR/'tracking_metrics.png', dpi=160); plt.show()
display(per_frame.head(25))
''')
add('markdown', '''
## 正解枠と追跡枠の動画

緑が正解、オレンジが採用した追跡結果です。Colab表示用にH.264へ変換します。
動画表示や変換に問題があっても、元の `tracked.mp4` と評価結果はDriveに保存されています。
''')
add('code', '''
browser_video = RUN_DIR/'tracked_browser.mp4'
ffmpeg = shutil.which('ffmpeg')
if ffmpeg:
    conversion = subprocess.run([ffmpeg, '-y', '-i', str(RUN_DIR/'tracked.mp4'),
                                 '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                 str(browser_video)], capture_output=True, text=True)
    if conversion.returncode == 0:
        display(Video(str(browser_video), embed=True, width=900))
    else:
        print('表示用変換に失敗しました。Driveのtracked.mp4をダウンロードしてください。')
        print(conversion.stderr[-1500:])
else:
    print('ffmpegがないため、Driveのtracked.mp4をダウンロードして確認してください。')
print('保存先:', RUN_DIR)
''')
add('markdown', '''
## 結果ZIPを作る（任意）

すべての結果は既にDriveに保存済みです。PCへまとめて取り出す場合は下を実行してください。
''')
add('code', '''
MAKE_RESULTS_ZIP = True
DOWNLOAD_RESULTS = False
if MAKE_RESULTS_ZIP:
    RESULT_ZIP = Path(shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name))
    print('結果ZIP:', RESULT_ZIP)
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(RESULT_ZIP))
''')

notebook=dict(cells=cells,metadata=dict(colab=dict(name='test_tracking_colab.ipynb',provenance=[]),
              kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
path=Path(__file__).resolve().parent/'test_tracking_colab.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path)
