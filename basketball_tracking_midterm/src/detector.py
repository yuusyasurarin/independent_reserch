from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

from .common import YOLO_COMMIT, check_repo, offline_environment, resolve, sha256


@contextmanager
def trusted_checkpoint_loading():
    """v7.0 checkpoints pickle model objects; only explicit trusted local assets are allowed."""
    import torch
    original=torch.load
    def load(*args,**kwargs):
        kwargs.setdefault('weights_only',False)
        return original(*args,**kwargs)
    torch.load=load
    try:
        yield
    finally:
        torch.load=original


def verify_weight(cfg,size,condition):
    role='pretrained' if condition=='A' else 'finetuned'
    path=resolve(cfg['models'][size][role])
    if not path.is_file():
        raise FileNotFoundError(f'{condition}/{size} unavailable: {path}. Train YOLOv5{size} in Colab and copy the export; no pretrained substitution is permitted.')
    metadata=path.with_suffix('.pt.metadata.json')
    if not metadata.is_file():
        raise FileNotFoundError(f'Model provenance required: {metadata}; use setup_assets.py (A) or the Colab export (B-E)')
    info=json.loads(metadata.read_text(encoding='utf-8'))
    if (info['role'],info['model_size'],info['yolov5_commit'],info['sha256']) != (role,size,YOLO_COMMIT,sha256(path)):
        raise ValueError('Model role / size / source commit / checksum mismatch')
    class_id=cfg['models'][size][role+'_class']
    if info['class_id']!=class_id:
        raise ValueError('Configured class ID differs from model export')
    if role=='finetuned':
        selection_path=path.parent/'final_selection.json'
        if not selection_path.is_file():raise FileNotFoundError('Copy final_selection.json together with best.pt')
        selection=json.loads(selection_path.read_text(encoding='utf-8'))
        if selection['weights_sha256']!=info['sha256'] or selection['selection_split']!='val' or cfg['selection_split']!='val':
            raise ValueError('Use operating settings frozen on val, and set selection_split=val')
        for key in ('imgsz','confidence','nms_iou','half'):
            if cfg[key]!=selection['operating_settings'][key]:raise ValueError(f'Config {key} differs from exported val-selected setting')
    return path,info,class_id


class Detector:
    def __init__(self,cfg,size,condition):
        offline_environment()
        self.path,self.metadata,self.class_id=verify_weight(cfg,size,condition)
        repo=resolve(cfg['yolov5_repo'])
        self.source=check_repo(repo)
        sys.path.insert(0,str(repo))
        import torch
        import numpy as np
        from models.experimental import attempt_load
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        self.torch,self.np=torch,np
        self.letterbox,self.nms,self.scale_boxes=letterbox,non_max_suppression,scale_boxes
        self.device=torch.device(cfg['device'])
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; explicitly choose device=cpu to measure CPU performance')
        with trusted_checkpoint_loading():
            self.model=attempt_load(str(self.path),device=self.device,inplace=True,fuse=True)
        names=self.model.names
        chosen=names[self.class_id]
        expected='sports ball' if condition=='A' else 'ball'
        if chosen!=expected or (condition!='A' and len(names)!=1):
            raise ValueError(f'Unexpected classes: {names}; expected {expected}')
        architecture=self.model.yaml
        expected_multipliers={'s':(.33,.5),'m':(.67,.75)}[size]
        if (architecture['depth_multiple'],architecture['width_multiple'])!=expected_multipliers:
            raise ValueError('Checkpoint architecture does not match selected YOLOv5 size')
        self.half=bool(cfg['half'])
        if self.half and self.device.type!='cuda':
            raise ValueError('FP16 is configured for CUDA only; explicitly disable it for CPU')
        self.model.half() if self.half else self.model.float()
        self.model.eval()
        self.cfg=cfg
        self.stride=int(self.model.stride.max())
        self.warmup_calls=0

    def sync(self):
        if self.device.type=='cuda':
            self.torch.cuda.synchronize(self.device)

    def detect(self,frame):
        t=self.torch
        self.sync()
        start=time.perf_counter()
        img=self.letterbox(frame,new_shape=(self.cfg['imgsz'],self.cfg['imgsz']),auto=False,stride=self.stride)[0]
        img=self.np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))
        tensor=t.from_numpy(img).to(self.device)
        tensor=tensor.half() if self.half else tensor.float()
        tensor=tensor.unsqueeze(0)/255
        self.sync()
        pre=time.perf_counter()
        with t.inference_mode():
            raw=self.model(tensor,augment=False)[0]
        self.sync()
        inference=time.perf_counter()
        result=self.nms(raw,conf_thres=self.cfg['candidate_conf'],iou_thres=self.cfg['nms_iou'],
                        classes=[self.class_id],agnostic=False,max_det=300)[0]
        if len(result):
            result[:,:4]=self.scale_boxes(tensor.shape[2:],result[:,:4],frame.shape[:2])
        values=result.detach().cpu().tolist()
        self.sync()
        after=time.perf_counter()
        candidates=[dict(bbox=list(map(float,r[:4])),confidence=float(r[4]),class_id=int(r[5])) for r in values]
        return candidates,dict(preprocess_ms=(pre-start)*1000,inference_ms=(inference-pre)*1000,
                               postprocess_ms=(after-inference)*1000)

    def warmup(self,width,height):
        # Warm-up contains no tracker state and is excluded from timed video measurements.
        for _ in range(3):
            self.detect(self.np.zeros((height,width,3),dtype=self.np.uint8))
            self.warmup_calls+=1
