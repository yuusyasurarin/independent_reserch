"""Intrinsic calibration of one fisheye camera from chessboard views (OpenCV fisheye model).

Record the chessboard with the SAME camera, resolution and lens setting used for
the games; tilt it and cover the whole image including the edges (20-40 views).
Output JSON holds K, D (k1..k4), image size and per-view reprojection error.
"""
import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_json


def views(args):
    import cv2
    if args.video:
        cap=cv2.VideoCapture(str(args.video));i=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if i%args.every==0:yield f'{args.video}#{i}',frame
            i+=1
        cap.release()
    else:
        for path in sorted(glob.glob(args.images)):
            frame=cv2.imread(path)
            if frame is not None:yield path,frame


def main(args):
    import cv2
    import numpy as np
    cols,rows=args.pattern
    objp=np.zeros((cols*rows,1,3),np.float64)
    objp[:,0,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*args.square
    object_points=[];image_points=[];used=[];size=None
    for name,frame in views(args):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if size is None:size=gray.shape[::-1]
        elif gray.shape[::-1]!=size:raise ValueError(f'Image size differs: {name}')
        ok,corners=cv2.findChessboardCornersSB(gray,(cols,rows),cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
        if not ok:
            ok,corners=cv2.findChessboardCorners(gray,(cols,rows),cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,1e-4))
        if not ok:
            print('no board:',name);continue
        object_points.append(objp);image_points.append(corners.reshape(-1,1,2).astype(np.float64));used.append(name)
        print('board found:',name,flush=True)
    if len(used)<args.min_views:raise RuntimeError(f'Only {len(used)} usable views; need at least {args.min_views}')
    K=np.eye(3);D=np.zeros((4,1))
    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC|cv2.fisheye.CALIB_FIX_SKEW
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,1e-6)
    try:
        rms,K,D,rvecs,tvecs=cv2.fisheye.calibrate(object_points,image_points,size,K,D,flags=flags|cv2.fisheye.CALIB_CHECK_COND,criteria=criteria)
    except cv2.error as error:
        print('CHECK_COND failed, retrying without it:',str(error).splitlines()[0])
        rms,K,D,rvecs,tvecs=cv2.fisheye.calibrate(object_points,image_points,size,K,D,flags=flags,criteria=criteria)
    per_view=[]
    for obj,img,r,t in zip(object_points,image_points,rvecs,tvecs):
        proj,_=cv2.fisheye.projectPoints(obj,r,t,K,D)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj-img)**2,axis=2)))))
    result=dict(model='opencv_fisheye',K=K.tolist(),D=D.reshape(-1).tolist(),image_size=list(size),rms_px=float(rms),
                views=len(used),per_view_rms_px=per_view,sources=used,pattern=dict(cols=cols,rows=rows,square_m=args.square),
                note='Calibrate again whenever the lens, focus, zoom or capture resolution changes.')
    save_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),rms_px=result['rms_px'],views=len(used),K=result['K'],D=result['D']),indent=1))
    if args.preview:
        name,frame=next(views(args))
        new_K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,size,np.eye(3),balance=.5)
        m1,m2=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),new_K,size,cv2.CV_16SC2)
        cv2.imwrite(str(args.preview),cv2.remap(frame,m1,m2,cv2.INTER_LINEAR))
        print('undistorted preview:',args.preview)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--images',help='glob of chessboard images, e.g. "calib/cam1/*.jpg"')
    src.add_argument('--video',type=Path,help='chessboard video; frames every --every are used')
    p.add_argument('--every',type=int,default=15)
    p.add_argument('--pattern',type=int,nargs=2,default=[9,6],metavar=('COLS','ROWS'),help='inner corners per row and column')
    p.add_argument('--square',type=float,default=.025,help='square edge length in metres')
    p.add_argument('--min-views',type=int,default=10)
    p.add_argument('--output',type=Path,required=True,help='e.g. configs/cameras/cam1_fisheye.json')
    p.add_argument('--preview',type=Path,help='write one undistorted frame for a visual check')
    a=p.parse_args()
    if a.every<1 or a.square<=0 or min(a.pattern)<3 or a.min_views<3:p.error('Invalid settings')
    main(a)
