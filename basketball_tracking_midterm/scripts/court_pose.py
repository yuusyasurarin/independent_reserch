"""Camera pose in court coordinates from clicked court landmarks (extrinsic calibration).

click : show one frame, click the landmarks listed in the court JSON in order,
        save their pixel positions. Keys: left click = set, s = skip, u = undo, q = finish.
solve : undistort the clicked points with the fisheye intrinsics, run solvePnP,
        report reprojection error and save R, t and the camera position.

Both cameras are solved independently against the same court file, which also
fixes the hoop positions used by score_detector.py.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_json
from src.stereo import load_camera, project, undistort


def grab(args):
    import cv2
    if args.image:
        frame=cv2.imread(str(args.image))
        if frame is None:raise FileNotFoundError(args.image)
        return frame
    cap=cv2.VideoCapture(str(args.video));cap.set(cv2.CAP_PROP_POS_FRAMES,args.frame)
    ok,frame=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot read frame {args.frame} of {args.video}')
    return frame


def click(args):
    import cv2
    court=json.loads(args.court.read_text(encoding='utf-8'))
    frame=grab(args);h,w=frame.shape[:2];scale=min(1.,args.max_display/max(h,w))
    names=[l['name'] for l in court['landmarks']];points={};state=dict(index=0,cursor=None)
    def render():
        canvas=cv2.resize(frame,None,fx=scale,fy=scale) if scale!=1 else frame.copy()
        for name,(x,y) in points.items():
            cv2.circle(canvas,(round(x*scale),round(y*scale)),4,(0,220,0),-1)
            cv2.putText(canvas,name,(round(x*scale)+5,round(y*scale)-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,0),1)
        label=names[state['index']] if state['index']<len(names) else 'done: press q'
        cv2.putText(canvas,f'click: {label}   (s skip, u undo, q finish)',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
        cv2.putText(canvas,f'click: {label}   (s skip, u undo, q finish)',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.imshow('court landmarks',canvas)
    def on_mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and state['index']<len(names):
            points[names[state['index']]]=[x/scale,y/scale];state['index']+=1;render()
    cv2.namedWindow('court landmarks');cv2.setMouseCallback('court landmarks',on_mouse);render()
    while True:
        key=cv2.waitKey(30)&0xFF
        if key==ord('q'):break
        if key==ord('s') and state['index']<len(names):state['index']+=1;render()
        if key==ord('u') and state['index']>0:
            state['index']-=1;points.pop(names[state['index']],None);render()
    cv2.destroyAllWindows()
    save_json(args.output,dict(image_size=[w,h],source=str(args.image or args.video),frame=args.frame,court=str(args.court),points=points))
    print(f'{len(points)} landmarks saved to {args.output}')


def solve(args):
    import cv2
    import numpy as np
    court=json.loads(args.court.read_text(encoding='utf-8'))
    clicked=json.loads(args.points.read_text(encoding='utf-8'))['points']
    cam=load_camera(args.calibration)
    xyz={l['name']:l['xyz'] for l in court['landmarks']}
    names=[n for n in clicked if n in xyz]
    if len(names)<6:raise ValueError(f'Need at least 6 clicked landmarks present in the court file; have {len(names)}')
    world=np.array([xyz[n] for n in names],float);pixels=np.array([clicked[n] for n in names],float)
    normalised=undistort(pixels,cam)
    ok,rvec,tvec=cv2.solvePnP(world.reshape(-1,1,3),normalised.reshape(-1,1,2),np.eye(3),None,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:raise RuntimeError('solvePnP failed')
    rvec,tvec=cv2.solvePnPRefineLM(world.reshape(-1,1,3),normalised.reshape(-1,1,2),np.eye(3),None,rvec,tvec)
    R,_=cv2.Rodrigues(rvec);cam.update(R=R,t=tvec.reshape(3))
    reproj=project(world,cam);errors=np.linalg.norm(reproj-pixels,axis=1)
    position=(-R.T@tvec.reshape(3)).tolist()
    result=dict(R=R.tolist(),t=tvec.reshape(3).tolist(),rvec=rvec.reshape(3).tolist(),camera_position_world=position,
                rms_px=float(np.sqrt(np.mean(errors**2))),max_px=float(errors.max()),
                per_landmark_px={n:float(e) for n,e in zip(names,errors)},landmarks_used=len(names),
                calibration=str(args.calibration),points=str(args.points),court=str(args.court),
                convention='X_cam = R @ X_world + t (OpenCV); world = court frame of the court file')
    save_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),camera_position_world=position,rms_px=result['rms_px'],max_px=result['max_px']),indent=1))
    if result['rms_px']>args.warn_px:
        print(f'WARNING: reprojection RMS {result["rms_px"]:.1f}px > {args.warn_px}px; re-check clicked points, the court measurements and the intrinsic calibration')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('click');g=c.add_mutually_exclusive_group(required=True)
    g.add_argument('--image',type=Path);g.add_argument('--video',type=Path)
    c.add_argument('--frame',type=int,default=0);c.add_argument('--court',type=Path,required=True)
    c.add_argument('--max-display',type=int,default=1600);c.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('solve');s.add_argument('--calibration',type=Path,required=True);s.add_argument('--points',type=Path,required=True)
    s.add_argument('--court',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--warn-px',type=float,default=5.)
    a=p.parse_args()
    click(a) if a.command=='click' else solve(a)
