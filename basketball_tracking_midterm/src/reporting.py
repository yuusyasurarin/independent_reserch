from pathlib import Path
from .common import save_csv


def plots(output,per_frame,pr_curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if per_frame:
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot([r['timestamp_s'] for r in per_frame],[r['iou'] for r in per_frame])
        axes[0].axhline(.3,linestyle=':',color='orange');axes[0].axhline(.5,linestyle='--',color='red')
        axes[0].set(ylabel='IoU',ylim=(-.03,1.03))
        axes[1].plot([r['timestamp_s'] for r in per_frame],[r['cle_px'] for r in per_frame])
        axes[1].set(ylabel='Center error (px)',xlabel='Original video time (s)')
        fig.tight_layout();fig.savefig(output/'tracking_metrics.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots()
    if pr_curve:
        ax.step([r['recall'] for r in pr_curve],[r['precision'] for r in pr_curve],where='post')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Raw YOLO candidates; scheduled frames only')
    fig.tight_layout();fig.savefig(output/'detector_pr.png',dpi=150);plt.close(fig)


def save_failures(video,gt,predictions,per_frame,output,limit,fps):
    import cv2
    output=Path(output)/'failures'
    candidates=sorted((r for r in per_frame if r['eligible'] and r['output'] and r['iou']<.3),key=lambda r:r['iou'])
    chosen=[]
    for row in candidates:
        if all(abs(row['frame_id']-x['frame_id'])>=fps for x in chosen):chosen.append(row)
        if len(chosen)>=limit:break
    if not chosen or limit<=0:return
    output.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(video));index={r['frame_id']:r for r in predictions}
    try:
        for row in chosen:
            i=row['frame_id'];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
            if not ok:raise ValueError('Could not read failure frame')
            for box,color in ((gt[i]['bbox'],(0,220,0)),(index[i]['bbox'],(0,140,255))):
                if box:cv2.rectangle(frame,tuple(round(v) for v in box[:2]),tuple(round(v) for v in box[2:]),color,2)
            cv2.putText(frame,f'frame={i} IoU={row["iou"]:.3f} manual review required',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            cv2.imencode('.jpg',frame)[1].tofile(str(output/f'frame_{i:06d}.jpg'))
        save_csv(output/'examples.csv',chosen)
    finally:cap.release()
