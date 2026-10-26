"""Pinned classical YOLOv5 training, true resume, split evaluation and Windows export."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, YOLO_COMMIT, check_repo, environment, offline_environment, save_json, sha256
from src.data import validate_dataset
from src.detector import trusted_checkpoint_loading
from src.compat import enable_legacy_font_size


class Tee:
    def __init__(self,terminal,file):self.terminal,self.file=terminal,file
    def write(self,text):self.terminal.write(text);self.file.write(text);self.file.flush()
    def flush(self):self.terminal.flush();self.file.flush()
    def isatty(self):return False


@contextmanager
def transcript(path):
    with Path(path).open('a',encoding='utf-8') as f:
        with redirect_stdout(Tee(sys.stdout,f)),redirect_stderr(Tee(sys.stderr,f)):
            yield


def load_training_modules(repo):
    check_repo(repo);offline_environment();sys.path.insert(0,str(repo))
    enable_legacy_font_size()
    import torch
    if torch.__version__.split('+')[0]!='2.5.1':
        raise RuntimeError('Training requires pinned torch 2.5.1; run the installation cell and use its venv Python')
    import train
    from utils.callbacks import Callbacks
    return torch,train,Callbacks


def checkpoint_callback(run_dir):
    def on_save(last,epoch,*args):
        # on_model_save fires before v7.0 strips optimizer from final last.pt/best.pt.
        temp=run_dir/'weights/resume_latest.tmp'
        shutil.copy2(last,temp)
        temp.replace(run_dir/'weights/resume_latest.pt')
        save_json(run_dir/'resume_checkpoint.json',dict(epoch_zero_based=epoch,checkpoint='weights/resume_latest.pt',
                  includes_optimizer=True,saved_utc=datetime.now(timezone.utc).isoformat()))
    return on_save


def configure_resume_save_period(run_dir, checkpoint_options, save_period):
    """v7 main reloads opt.yaml on resume, overriding the caller's options."""
    import yaml
    options_path=run_dir/'opt.yaml'
    options=(yaml.safe_load(options_path.read_text(encoding='utf-8'))
             if options_path.is_file() else checkpoint_options)
    if not isinstance(options,dict):
        raise ValueError('Resume requires opt.yaml or checkpoint opt settings')
    options=dict(options)
    previous=options.get('save_period',-1)
    options['save_period']=save_period
    if options_path.is_file() and previous!=save_period:
        backup=run_dir/'opt.before_storage_change.yaml'
        if not backup.exists():shutil.copy2(options_path,backup)
    options_path.write_text(yaml.safe_dump(options,sort_keys=False),encoding='utf-8')
    save_json(run_dir/'resume_storage_policy.json',dict(previous_save_period=previous,
        save_period=save_period,updated_utc=datetime.now(timezone.utc).isoformat(),
        note='Only periodic epoch-file retention changed; existing checkpoints are not deleted.'))


def train_model(a):
    import yaml
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    size=a.size
    cfg['model_size']=size
    repo=ROOT/'vendor/yolov5'
    torch,train,Callbacks=load_training_modules(repo)
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU runtime is required for training')
    if cfg['imgsz']%32 or cfg['batch']<1 or cfg['epochs']<1:raise ValueError('Invalid image size / batch / epochs')
    cfg.setdefault('save_period',-1)
    if type(cfg['save_period']) is not int or (cfg['save_period']!=-1 and cfg['save_period']<1):
        raise ValueError('save_period must be -1 (no epoch archives) or a positive integer')
    if a.mode=='train':
        if a.run.exists():raise FileExistsError('Choose a new run directory; use resume only for an interrupted run')
        a.run.mkdir(parents=True)
        report,data=validate_dataset(a.dataset,a.run/'data_audit',cfg['allow_background'])
        initial=ROOT/f'weights/pretrained/yolov5{size}.pt'
        if not initial.is_file():raise FileNotFoundError('Run setup_assets.py --sizes '+size)
        provenance=json.loads(initial.with_suffix('.pt.metadata.json').read_text(encoding='utf-8'))
        if provenance['model_size']!=size or provenance['sha256']!=sha256(initial):raise ValueError('Initial model differs')
        metadata=dict(model_size=size,config=cfg,initial_weights=provenance,yolov5_commit=YOLO_COMMIT,
            environment=environment(),dataset_inventory_sha256=sha256(a.run/'data_audit/dataset_inventory.csv'),
            dataset=report,class_names={0:'ball'},selection_split='val',
            selection_rule='YOLOv5 v7.0 validation fitness; no test used in training/early stopping')
        metadata['compatibility_adaptations']=['Pillow 10 getsize plotting shim; model/training mathematics unchanged']
        metadata['wrapper_source_sha256']={str(p.relative_to(ROOT)):sha256(p) for folder in ('src','scripts') for p in (ROOT/folder).glob('*.py')}
        save_json(a.run/'training_metadata.json',metadata)
        (a.run/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
        opt=train.parse_opt(known=True)
        opt.weights=str(initial);opt.data=str(data.resolve());opt.cfg=str(repo/f'models/yolov5{size}.yaml')
        opt.imgsz=cfg['imgsz'];opt.epochs=cfg['epochs'];opt.batch_size=cfg['batch'];opt.seed=cfg['seed']
        opt.patience=cfg['patience'];opt.workers=cfg['workers'];opt.save_period=cfg['save_period']
        opt.optimizer=cfg['optimizer'];opt.cache='ram' if cfg['cache'] else None
        opt.project=str(a.run.parent);opt.name=a.run.name;opt.exist_ok=True;opt.device='0'
        opt.hyp=str(repo/'data/hyps/hyp.scratch-low.yaml')
        opt.resume=False;opt.evolve=False
        opt.weights=str(initial)
    else:
        metadata=json.loads((a.run/'training_metadata.json').read_text(encoding='utf-8'))
        if metadata['model_size']!=size or metadata['yolov5_commit']!=YOLO_COMMIT:raise ValueError('Resume model/source mismatch')
        validate_dataset(a.dataset,a.run/'resume_data_audit',metadata['config']['allow_background'])
        if sha256(a.run/'resume_data_audit/dataset_inventory.csv')!=metadata['dataset_inventory_sha256']:
            raise ValueError('Resume dataset differs from original run')
        # Restore absolute data.yaml in the persisted run to the newly mounted/local dataset.
        original_data=a.run/'data_audit/data.yaml'
        original_data.write_text(yaml.safe_dump(dict(path=str(a.dataset.resolve()),train='images/train',val='images/val',
                                                    test='images/test',nc=1,names={0:'ball'})),encoding='utf-8')
        checkpoint=a.checkpoint or a.run/'weights/resume_latest.pt'
        if checkpoint.resolve().parent!=(a.run/'weights').resolve():
            raise ValueError('Resume checkpoint must belong to this run/weights directory')
        with trusted_checkpoint_loading():ckpt=torch.load(checkpoint,map_location='cpu')
        if ckpt.get('optimizer') is None or ckpt.get('epoch',-1)<0:
            raise ValueError('This weight has no optimizer/epoch state; it is not a resume checkpoint')
        if ckpt['epoch']+1>=metadata['config']['epochs']:
            raise ValueError('Requested training already completed; using weights for another run is additional training, not resume')
        configure_resume_save_period(a.run,ckpt.get('opt'),cfg['save_period'])
        del ckpt
        opt=train.parse_opt(known=True);opt.resume=str(checkpoint.resolve());opt.device='0'
    callbacks=Callbacks()
    callbacks.register_action('on_model_save',name='persist_optimizer_checkpoint',callback=checkpoint_callback(a.run))
    with transcript(a.run/f'{a.mode}.log'),trusted_checkpoint_loading():
        train.main(opt,callbacks=callbacks)
    save_json(a.run/'completion.json',dict(status='completed',mode=a.mode,model_size=size,
                                           best_sha256=sha256(a.run/'weights/best.pt')))


def evaluate_model(a):
    metadata=json.loads((a.run/'training_metadata.json').read_text(encoding='utf-8'))
    if a.size!=metadata['model_size']:raise ValueError('Evaluation model size differs from training run')
    repo=ROOT/'vendor/yolov5';load_training_modules(repo)
    import val
    weight=a.run/'weights/best.pt'
    if a.split=='test':
        if not a.final_selection:raise ValueError('Final test requires --final-selection exported/final_selection.json')
        selection=json.loads(a.final_selection.read_text(encoding='utf-8'))
        if selection['weights_sha256']!=sha256(weight) or selection['selection_split']!='val':
            raise ValueError('Final test model/settings were not frozen using val')
    target=a.run/f'{a.split}_evaluation_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    target.mkdir()
    _,data=validate_dataset(a.dataset,target/'data_audit',metadata['config']['allow_background'])
    if sha256(target/'data_audit/dataset_inventory.csv')!=metadata['dataset_inventory_sha256']:
        raise ValueError('Evaluation dataset differs from training inventory')
    with transcript(target/'evaluation.log'),trusted_checkpoint_loading():
        result,maps,timing=val.run(data=str(data),weights=str(weight),batch_size=metadata['config']['batch'],
            imgsz=metadata['config']['imgsz'],conf_thres=.001,iou_thres=.6,task=a.split,device='0',
            workers=metadata['config']['workers'],project=str(target.parent),name=target.name,exist_ok=True,
            save_txt=True,save_conf=True,plots=True,half=False)
    save_json(target/'metrics.json',dict(split=a.split,result_names=['precision','recall','mAP50','mAP50_95','box_loss','obj_loss','cls_loss'],
        results=list(map(float,result)),per_class_map=list(map(float,maps)),stage_ms=list(map(float,timing)),
        weights_sha256=sha256(weight),yolov5_commit=YOLO_COMMIT,confidence_floor=.001,
        note='PR/AP evaluation uses candidate confidence .001. Logged P/R can use v5 best-F1 curve threshold; not the deployment operating point.'))
    print('Evaluation saved:',target)


def export_model(a):
    metadata=json.loads((a.run/'training_metadata.json').read_text(encoding='utf-8'))
    if a.size!=metadata['model_size']:raise ValueError('Export model size mismatch')
    if not a.val_evidence or not a.val_evidence.is_file():raise ValueError('Specify --val-evidence pointing to this run val_evaluation.../metrics.json')
    evidence=json.loads(a.val_evidence.read_text(encoding='utf-8'))
    weight=a.run/'weights/best.pt'
    if evidence['split']!='val' or evidence['weights_sha256']!=sha256(weight):raise ValueError('Val evidence does not match selected best.pt')
    if not 0<a.confidence<=1 or not 0<a.nms_iou<=1:raise ValueError('Invalid operating thresholds')
    if not a.selection_note:raise ValueError('Describe how the operating settings were checked on val')
    a.export_dir.mkdir(parents=True,exist_ok=False)
    shutil.copy2(weight,a.export_dir/'best.pt')
    info=dict(role='finetuned',model_size=a.size,yolov5_commit=YOLO_COMMIT,sha256=sha256(weight),class_id=0,class_name='ball',
              training_run=str(a.run.resolve()),training_metadata_sha256=sha256(a.run/'training_metadata.json'))
    save_json(a.export_dir/'best.pt.metadata.json',info)
    selection=dict(weights_sha256=sha256(weight),model_size=a.size,selection_split='val',selection_note=a.selection_note,
        operating_settings=dict(imgsz=metadata['config']['imgsz'],confidence=a.confidence,nms_iou=a.nms_iou,half=False),
        validation_evidence_sha256=sha256(a.val_evidence),validation_evidence=str(a.val_evidence),
        fixed_utc=datetime.now(timezone.utc).isoformat())
    save_json(a.export_dir/'final_selection.json',selection)
    shutil.copy2(a.val_evidence,a.export_dir/'val_metrics.json')
    for name in ('training_metadata.json','environment.txt'):
        shutil.copy2(a.run/name,a.export_dir/name)
    print('Windows export:',a.export_dir)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','resume','evaluate','export'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/training.json')
    p.add_argument('--size',choices=['s','m'],required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--final-selection',type=Path)
    p.add_argument('--export-dir',type=Path)
    p.add_argument('--val-evidence',type=Path)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--nms-iou',type=float,default=.45)
    p.add_argument('--selection-note',default='')
    a=p.parse_args()
    a.dataset=a.dataset.resolve();a.run=a.run.resolve()
    if a.mode in ('train','resume'):train_model(a)
    elif a.mode=='evaluate':evaluate_model(a)
    else:
        if a.export_dir is None:p.error('--export-dir is required')
        export_model(a)


if __name__=='__main__':main()
