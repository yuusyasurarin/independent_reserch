"""Online preparation only. Sequential experiment scripts never download assets."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, YOLO_COMMIT, save_json, sha256, check_repo


def download(url, destination):
    request=urllib.request.Request(url,headers={'User-Agent':'basketball-midterm-research'})
    with urllib.request.urlopen(request,timeout=120) as response, Path(destination).open('wb') as f:
        shutil.copyfileobj(response,f)


def setup(sizes):
    vendor=ROOT/'vendor/yolov5'
    if vendor.exists():
        check_repo(vendor)
    else:
        vendor.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=vendor.parent) as d:
            archive=Path(d)/'source.zip'
            url=f'https://codeload.github.com/ultralytics/yolov5/zip/{YOLO_COMMIT}'
            download(url,archive)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not (Path(d)/name).resolve().is_relative_to(Path(d).resolve()):
                        raise ValueError('Unsafe archive member')
                z.extractall(d)
            source=Path(d)/f'yolov5-{YOLO_COMMIT}'
            save_json(source/'midterm_source.json',dict(commit=YOLO_COMMIT,source_url=url,
                archive_sha256=sha256(archive),license='GPL-3.0',
                files={str(p.relative_to(source)).replace('\\','/'):sha256(p) for p in source.rglob('*')
                       if p.is_file() and (p.suffix in ('.py','.yaml') or p.name=='LICENSE')}))
            shutil.move(str(source),str(vendor))
    for size in sizes:
        path=ROOT/f'weights/pretrained/yolov5{size}.pt'
        metadata=path.with_suffix('.pt.metadata.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if not metadata.is_file() or json.loads(metadata.read_text())['sha256']!=sha256(path):
                raise ValueError(f'Existing unverified weight: {path}; use a different destination or verify manually')
        else:
            url=f'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{size}.pt'
            temporary=path.with_suffix('.download')
            download(url,temporary)
            temporary.replace(path)
            save_json(metadata,dict(role='pretrained',model_size=size,source_url=url,sha256=sha256(path),
                                    yolov5_commit=YOLO_COMMIT,class_id=32,class_name='sports ball'))
        print(path)
    print('Pinned YOLOv5 source ready:',vendor)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',nargs='*',choices=['s','m'],default=[])
    setup(p.parse_args().sizes)
