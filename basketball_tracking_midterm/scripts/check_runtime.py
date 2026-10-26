"""Preflight for offline venue use; no downloads and no training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,check_repo,environment,load_config,resolve,save_json
from src.detector import verify_weight


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/experiment.json'))
    a=p.parse_args();cfg=load_config(a.config)
    report=dict(environment=environment(),source_ready=False,conditions={})
    try:
        check_repo(resolve(cfg['yolov5_repo']));report['source_ready']=True
    except Exception as e:report['source_error']=str(e)
    for size in ('s','m'):
        for condition in 'ABCDE':
            try:
                weight,metadata,class_id=verify_weight(cfg,size,condition)
                report['conditions'][size+'/'+condition]=dict(ready=True,weight=str(weight),class_id=class_id)
            except (FileNotFoundError,ValueError) as e:
                report['conditions'][size+'/'+condition]=dict(ready=False,reason=str(e))
    save_json(ROOT/'data/runtime_preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
