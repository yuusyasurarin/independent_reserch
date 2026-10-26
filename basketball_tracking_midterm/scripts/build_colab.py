"""Generate Japanese Colab notebook and a code-only upload archive (no dataset copies)."""
import json
from pathlib import Path
import textwrap
import zipfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,sha256

cells=[]
def add(kind,text):
    cell=dict(cell_type=kind,metadata={},id=f'midterm-{len(cells):03}',source=textwrap.dedent(text).strip().splitlines(keepends=True))
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)

add('markdown','''
# バスケットボール：従来のYOLOv5s／mを追加学習
YOLOv5 v7.0、commit `915bbf294bb74c859f0b41f1c23bc395014ea679` を学習・Windows推論で共用します。
YOLOv5uや既存のYOLO26モデルへの置換は行いません。

準備：`colab_upload.zip` をマイドライブの `basketball_tracking_midterm/` に置きます。
既存の `ball_yolo_colab.zip`（データセット入り）はそのまま再利用できます。
GPUを選択し、上から環境・データを確認します。学習・再開・最終testの実行スイッチは初期状態でFalseです。
最初にsを実行し、その後 `MODEL_SIZE='m'` にして別runで実行してください。
''')
add('code','''
import subprocess, sys, os, json, zipfile, shutil, hashlib
from pathlib import Path
from datetime import datetime, timezone
# GPUの種類は決めつけず、毎回確認します。
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], capture_output=True, text=True)
print(gpu.stdout)
if gpu.returncode != 0: raise RuntimeError('ColabのGPUランタイムを選択してください')
from google.colab import drive
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive')
PROJECT = DRIVE/'basketball_tracking_midterm'
PROJECT.mkdir(exist_ok=True)
''')
add('markdown','''
## コードと固定環境を準備
Colab本体のPythonパッケージを置き換えず、専用仮想環境を作ります。
Python 3.11/3.12以外の場合だけ、固定版uvを使ってPython 3.12を準備します。
依存関係・初期重みの取得にはインターネット接続が必要です。学習はこのセルでは始まりません。
''')
add('code','''
CODE_ZIP = PROJECT/'colab_upload.zip'
WORK_PARENT = Path('/content/midterm_code')
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def safe_extract_once(archive, destination):
    marker = destination/'.archive_sha256'
    if destination.exists():
        if not marker.is_file() or marker.read_text() != digest(archive):
            raise ValueError('異なるZIPの展開先が残っています。ランタイムを再起動してから再実行してください。')
        return
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination/item.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('不正なZIPパス')
        z.extractall(destination)
    marker.write_text(digest(archive))
safe_extract_once(CODE_ZIP, WORK_PARENT)
WORK = WORK_PARENT/'basketball_tracking_midterm'
VENV = Path('/content/midterm_venv')
if not (VENV/'bin/python').is_file():
    if sys.version_info[:2] in ((3,11),(3,12)):
        subprocess.run([sys.executable,'-m','venv',str(VENV)],check=True)
    else:
        BOOT = Path('/content/midterm_bootstrap')
        subprocess.run([sys.executable,'-m','pip','install','--target',str(BOOT),'uv==0.5.11'],check=True)
        env = dict(os.environ, PYTHONPATH=str(BOOT))
        subprocess.run([sys.executable,'-m','uv','venv','--python','3.12','--seed',str(VENV)],env=env,check=True)
PYTHON = str(VENV/'bin/python')
subprocess.run([PYTHON,'-m','pip','install','pip==24.3.1','setuptools==75.6.0'],check=True)
subprocess.run([PYTHON,'-m','pip','install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121'],check=True)
subprocess.run([PYTHON,'-m','pip','install','-r',str(WORK/'requirements.txt')],check=True)
subprocess.run([PYTHON,'-m','pip','install','--no-deps','supervision==0.25.0'],check=True)
subprocess.run([PYTHON,'-c','import torch; print(torch.__version__,torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'],check=True)
''')
add('markdown','''
## データパス・モデル・共通学習条件
`MODEL_SIZE`はsまたはm。両方を学習する場合はモデルごとに別runを作ります。
同じimgsz・epochs・seedを基本にします。VRAM不足ならBATCHを4→2→1と明示的に変更し、新しいrunでやり直します。
モデルサイズや画像サイズは自動変更しません。変更内容はtraining_metadata.jsonに保存します。
空ラベル161枚は既存資料で確認済み背景です。別データを使う場合はこの確認を引き継がないでください。
''')
add('code','''
MODEL_SIZE = 's'  # 次に 'm' を選んで別runで学習
DATASET_ZIP = DRIVE/'colab_ball_yolo/ball_yolo_colab.zip'  # 実際のDrive保存場所に変更
DATASET_FOLDER = ''  # ZIPを使わず既存のdatasetフォルダを参照する場合の絶対パス
COPY_FOLDER_TO_LOCAL = True
IMGSZ = 960
EPOCHS = 100
BATCH = 4
SEED = 0
PATIENCE = 20
WORKERS = 2
SAVE_PERIOD = -1  # epoch0.pt等を蓄積しない。best/last/再開用の最新版のみ更新
ALLOW_CONFIRMED_BACKGROUND = True
assert MODEL_SIZE in ('s','m')
TRAIN_CONFIG = json.loads((WORK/'configs/training.json').read_text())
TRAIN_CONFIG.update(model_size=MODEL_SIZE,imgsz=IMGSZ,epochs=EPOCHS,batch=BATCH,seed=SEED,
                    patience=PATIENCE,workers=WORKERS,save_period=SAVE_PERIOD,allow_background=ALLOW_CONFIRMED_BACKGROUND)
CONFIG_PATH = WORK/'configs/colab_current.json'
CONFIG_PATH.write_text(json.dumps(TRAIN_CONFIG,indent=2))
RUN = PROJECT/'runs'/MODEL_SIZE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
print('新規学習の保存先:',RUN)
''')
add('code','''
if DATASET_FOLDER:
    source = Path(DATASET_FOLDER)
    if COPY_FOLDER_TO_LOCAL:
        DATASET = Path('/content/midterm_dataset_folder')
        if not DATASET.exists(): shutil.copytree(source,DATASET)
        # 再接続・再実行時も元Driveフォルダと画像・ラベルのhashを照合します。
        for folder in ('images','labels'):
            files = {p.relative_to(source):digest(p) for p in (source/folder).rglob('*') if p.is_file()}
            copied = {p.relative_to(DATASET):digest(p) for p in (DATASET/folder).rglob('*') if p.is_file()}
            if files != copied: raise ValueError('ローカルコピーとDriveのデータが異なります')
    else: DATASET = source
else:
    extraction = Path('/content/midterm_dataset_zip')
    safe_extract_once(DATASET_ZIP,extraction)
    DATASET = extraction/'ball_yolo_colab/dataset'
if not (DATASET/'data.yaml').is_file(): raise FileNotFoundError(DATASET/'data.yaml')
subprocess.run([PYTHON,str(WORK/'scripts/setup_assets.py'),'--sizes',MODEL_SIZE],check=True)
audit_script = "from pathlib import Path; from src.data import validate_dataset; import sys; validate_dataset(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]=='True')"
AUDIT = PROJECT/'dataset_audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
subprocess.run([PYTHON,'-c',audit_script,str(DATASET),str(AUDIT),str(ALLOW_CONFIRMED_BACKGROUND)],cwd=WORK,check=True)
print((AUDIT/'dataset_validation.json').read_text())
''')
add('markdown','''
## 新しい追加学習を開始
trainで更新し、valだけでbest重み・early stoppingを決めます。testはこの処理に使いません。
保存容量を抑えるため、epoch0.pt、epoch1.pt…の個別保存は無効（SAVE_PERIOD=-1）です。
last.ptとresume_latest.ptは毎epoch同じ名前で更新し、best.ptはvalの最良モデルとして更新します。
1つのrunのweights内に残る.ptは基本3個で、epoch数に比例して増えません。更新時には一時コピーの空き容量も必要です。
最初のepochが完了する前の切断では再開できません。複数runやexportsのモデルは別途容量を使います。
''')
add('code','''
RUN_TRAIN = False  # 検査結果を確認してからTrueに変更
if RUN_TRAIN:
    subprocess.run([PYTHON,str(WORK/'scripts/train_yolov5.py'),'train','--size',MODEL_SIZE,
                    '--config',str(CONFIG_PATH),'--dataset',str(DATASET),'--run',str(RUN)],check=True)
else: print('学習は未実行です。RUN_TRAIN=Trueで開始します。')
''')
add('markdown','''
## 切断した学習を再開（必要な場合のみ）
環境・データ準備セルを再実行後、元runのパスを指定します。モデルサイズも元runと一致させます。
resume_latest.ptはoptimizer・epoch・EMAを含むチェックポイントです。
best.ptだけを初期重みとして別runを学習することは「追加学習」であり、この再開とは異なります。
完了済みrunのepochs延長には使いません。正常終了後のlast.ptはYOLOv5がoptimizerを除くため、再開用にはresume_latest.ptを使用します。
旧版で開始したrunも、再開時に現在のSAVE_PERIODを適用します。元の保存設定は記録し、既存のepoch*.ptは削除しません。
''')
add('code','''
RUN_RESUME = False
RESUME_RUN = ''  # 例: /content/drive/MyDrive/basketball_tracking_midterm/runs/s/日時
if RUN_RESUME:
    if not RESUME_RUN: raise ValueError('RESUME_RUNを指定してください')
    subprocess.run([PYTHON,str(WORK/'scripts/train_yolov5.py'),'resume','--size',MODEL_SIZE,
                    '--config',str(CONFIG_PATH),'--dataset',str(DATASET),'--run',RESUME_RUN],check=True)
    RUN = Path(RESUME_RUN)
''')
add('markdown','''
## val評価と学習曲線
ここでモデル・運用閾値を確認します。testやexperiment10の結果で設定を選択しません。
Precision/RecallのログはPR曲線上の閾値に依存します。運用時のconfidenceと同じ値とは限りません。
''')
add('code','''
RUN_VAL = False
SELECTED_RUN = ''  # 学習が完了したsまたはmのrunを明示
if RUN_VAL:
    if not SELECTED_RUN: raise ValueError('SELECTED_RUNを指定してください')
    subprocess.run([PYTHON,str(WORK/'scripts/train_yolov5.py'),'evaluate','--size',MODEL_SIZE,
                    '--dataset',str(DATASET),'--run',SELECTED_RUN,'--split','val'],check=True)
    from IPython.display import display, Image
    for name in ('results.png','confusion_matrix.png','PR_curve.png'):
        image = Path(SELECTED_RUN)/name
        if image.is_file(): display(Image(filename=str(image)))
    print('val_evaluation_日時フォルダの評価画像とmetrics.jsonも確認してください。')
''')
add('markdown','''
## valで設定を確定し、Windows用にエクスポート
val評価のmetrics.jsonと、運用設定を選んだ根拠を指定します。
ここで保存するconfidence/NMS値は、ユーザーがvalで確認した設定です。初期値の最適性は主張しません。
sとmはそれぞれの学習runから個別にエクスポートしてください。
''')
add('code','''
RUN_EXPORT = False
VAL_EVIDENCE = ''  # SELECTED_RUN/val_evaluation_日時/metrics.json
DEPLOY_CONF = 0.25
DEPLOY_NMS_IOU = 0.45
SELECTION_NOTE = ''  # valのどの評価で確認した設定かを記載
EXPORT_DIR = PROJECT/'exports'/MODEL_SIZE/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
if RUN_EXPORT:
    subprocess.run([PYTHON,str(WORK/'scripts/train_yolov5.py'),'export','--size',MODEL_SIZE,
                    '--dataset',str(DATASET),'--run',SELECTED_RUN,'--export-dir',str(EXPORT_DIR),
                    '--val-evidence',VAL_EVIDENCE,'--confidence',str(DEPLOY_CONF),'--nms-iou',str(DEPLOY_NMS_IOU),
                    '--selection-note',SELECTION_NOTE],check=True)
    print('Windowsへ渡すフォルダ:',EXPORT_DIR)
''')
add('markdown','''
## 最終test評価（設定確定後に明示実行）
データセットのtest（58枚）を評価します。experiment10の連続追跡実験とは別の評価です。
testを見て変更した場合は、そのtestは独立した最終評価ではなくなります。
''')
add('code','''
RUN_FINAL_TEST = False
FINAL_SELECTION = ''  # exports/sまたはm/日時/final_selection.json
if RUN_FINAL_TEST:
    if not FINAL_SELECTION: raise ValueError('確定したfinal_selection.jsonを指定してください')
    subprocess.run([PYTHON,str(WORK/'scripts/train_yolov5.py'),'evaluate','--size',MODEL_SIZE,
                    '--dataset',str(DATASET),'--run',SELECTED_RUN,'--split','test','--final-selection',FINAL_SELECTION],check=True)
    frozen = Path(FINAL_SELECTION).parent
    tests = sorted(Path(SELECTED_RUN).glob('test_evaluation_*'))
    if tests:
        shutil.copytree(tests[-1],frozen/tests[-1].name)
    print('test評価はrunフォルダとエクスポート先に保存されました。')
''')
add('markdown','''
## Windowsへ渡すファイル
exports内の対象フォルダを丸ごとダウンロードし、Windowsの `weights/finetuned/s/` または `m/` へ配置します。
`best.pt` と `best.pt.metadata.json` をセットで配置してください。classは0: ballです。
`final_selection.json` のimgsz・confidence・NMSを `configs/experiment.json` に反映し、`selection_split` を `val` にします。
比較実験のB〜Eでは同じ重みと共通検出設定を使います。sの重みをmとして使用することはできません。
学習run・定期チェックポイント・ログ・曲線・test評価はDriveに残ります。重みだけを残して他の記録を削除しないでください。
''')

notebook=dict(cells=cells,metadata=dict(colab=dict(name='train_yolov5_colab.ipynb',provenance=[]),
              kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells):
    if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell_{i}','exec')
(ROOT/'colab').mkdir(exist_ok=True)
(ROOT/'colab/train_yolov5_colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(ROOT/'colab_upload.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('src','scripts','configs','tests'):
        for path in sorted((ROOT/folder).glob('*')):
            if path.is_file() and path.suffix in ('.py','.ps1','.json'):
                z.write(path,'basketball_tracking_midterm/'+path.relative_to(ROOT).as_posix())
    for name in ('requirements.txt','constraints-common.txt','README.md','THIRD_PARTY.md'):
        z.write(ROOT/name,'basketball_tracking_midterm/'+name)
(ROOT/'colab_upload.sha256').write_text(sha256(ROOT/'colab_upload.zip')+'  colab_upload.zip\n')
print('Created notebook and code-only colab_upload.zip')
