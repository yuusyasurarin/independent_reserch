"""Actual-video KCF/LK diagnostics, with an explicitly separate teaching KCF.

OpenCV does not expose its native KCF response. The response images here are
computed with a one-channel grayscale Gaussian-kernel correlation filter, not
claimed to be E's internal Gray+ColorNames/PCA response or confidence.
LK arrays are captured from the actual AdaptiveBallPolicy call without changing
its outputs. Ground truth is used only to select and explain examples afterward.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, center, iou, resolve, save_json
from src.data import load_gt
from src.adaptive_tracking import AdaptiveBallPolicy
from scripts.research_g import read_jsonl


def gray_kcf(previous,current,box):
    import cv2
    import numpy as np
    w,h=box[2]-box[0],box[3]-box[1]
    shape=(max(32,round(w*2.5)),max(32,round(h*2.5)))
    xy=center(box)
    window=np.outer(np.hanning(shape[1]),np.hanning(shape[0]))
    def features(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return (cv2.getRectSubPix(gray,shape,xy).astype(float)/255-.5)*window
    x,z=features(previous),features(current)
    def kernel(a,b):
        correlation=np.fft.ifft2(np.fft.fft2(b)*np.conj(np.fft.fft2(a))).real
        squared=np.maximum(0,(np.sum(a*a)+np.sum(b*b)-2*correlation)/a.size)
        return np.exp(-squared/.2**2)
    ys=np.arange(shape[1]);ys=np.minimum(ys,shape[1]-ys)
    xs=np.arange(shape[0]);xs=np.minimum(xs,shape[0]-xs)
    label=np.exp(-(ys[:,None]**2+xs[None,:]**2)/(2*(math.sqrt(w*h)*.125)**2))
    alpha=np.fft.fft2(label)/(np.fft.fft2(kernel(x,x))+1e-4)
    response=np.fft.fftshift(np.fft.ifft2(alpha*np.fft.fft2(kernel(x,z))).real)
    py,px=np.unravel_index(np.argmax(response),response.shape)
    dx,dy=px-shape[0]//2,py-shape[1]//2
    predicted=[box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy]
    return dict(response=response,delta=(int(dx),int(dy)),predicted=predicted,
                peak=float(response[py,px]),shape=shape)


def collect(e_run,g_run):
    import cv2
    import numpy as np
    cv2.setNumThreads(0)
    em=json.loads((e_run/'run_config.json').read_text(encoding='utf-8'))
    gm=json.loads((g_run/'run_config.json').read_text(encoding='utf-8'))
    info=gm['video'];gt=load_gt(Path(gm['ground_truth']),info)
    ep=read_jsonl(e_run/'online.jsonl');gp=read_jsonl(g_run/'online.jsonl');logs=read_jsonl(g_run/'candidates.jsonl')
    policy=AdaptiveBallPolicy(gm['config'],info['fps'])
    candidates={k:[] for k in ('kcf_success','kcf_failure','lk_success','lk_failure')}
    counts={k:0 for k in candidates}
    def offer(key,item):
        counts[key]+=1
        if not candidates[key] or abs(item['frame_id']-421)<abs(candidates[key][0]['frame_id']-421):
            candidates[key]=[item]
    calls=[];original=cv2.calcOpticalFlowPyrLK
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        calls.append(tuple(v.copy() if v is not None else None for v in result))
        return result
    cv2.calcOpticalFlowPyrLK=capture
    cap=cv2.VideoCapture(info['path']);previous=None
    try:
        for i in range(min(1100,info['frame_count'])):
            ok,frame=cap.read();assert ok
            old=list(policy.box) if policy.box else None
            points=policy.points.copy() if policy.points is not None else None
            velocity=policy.velocity.copy();calls.clear()
            detected=policy.needs_detection(i,frame)
            assert detected==logs[i]['yolo_executed']
            if previous is not None and i>=30 and gt[i]['bbox'] and gt[i-1]['bbox']:
                # One step after a fresh KCF init: matches a single-template teaching map.
                a,b=ep[i-1],ep[i]
                if a['source']=='YOLO' and a['bbox'] and iou(a['bbox'],gt[i-1]['bbox'])>.65:
                    if (b['source']=='KCF' and iou(b['bbox'],gt[i]['bbox'])>.65) or b['kcf_success_flag'] is False:
                        k=gray_kcf(previous,frame,a['bbox'])
                        item=dict(frame_id=i,previous=previous.copy(),current=frame.copy(),
                            old=a['bbox'],truth=gt[i]['bbox'],native=b['bbox'],native_ok=b['kcf_success_flag'],**k)
                        if b['source']=='KCF' and iou(k['predicted'],gt[i]['bbox'])>.65 and np.linalg.norm(k['delta'])>=3:
                            offer('kcf_success',item)
                        if b['kcf_success_flag'] is False and iou(k['predicted'],gt[i]['bbox'])<.3:
                            offer('kcf_failure',item)
                if old and points is not None and len(points)>=4 and len(calls)==2 and iou(old,gt[i-1]['bbox'])>.65:
                    nxt,status,_=calls[0];back,reverse,_=calls[1]
                    if nxt is not None and back is not None:
                        d=max(old[2]-old[0],old[3]-old[1]);margin=max(24.,d*1.5,float(np.linalg.norm(velocity))*3)
                        offset=np.array([max(0,math.floor(old[0]-margin)),max(0,math.floor(old[1]-margin))])
                        oldp=points.reshape(-1,2);newp=nxt.reshape(-1,2)+offset
                        errors=np.linalg.norm(back.reshape(-1,2)+offset-oldp,axis=1)
                        good=(status.ravel()==1)&(reverse.ravel()==1)&np.isfinite(errors)&(errors<=1.5)
                        good&=np.isfinite(newp).all(axis=1)&(newp[:,0]>=0)&(newp[:,0]<info['width'])&(newp[:,1]>=0)&(newp[:,1]<info['height'])
                        if good.sum()>=4:
                            shift=np.median((newp-oldp)[good],axis=0)
                            good&=np.linalg.norm((newp-oldp)-shift,axis=1)<=max(2.,d*.15)
                        item=dict(frame_id=i,previous=previous.copy(),current=frame.copy(),old=old,
                            truth=gt[i]['bbox'],native=policy.flow_box,points=oldp,new_points=newp,errors=errors,
                            status=status.ravel(),reverse_status=reverse.ravel(),good=good,
                            actual_saved_source=gp[i]['source'],actual_detector_requested=detected)
                        if policy.flow_box and gp[i]['source']=='LK' and iou(policy.flow_box,gt[i]['bbox'])>.65:
                            offer('lk_success',item)
                        if policy.flow_box is None and good.sum()<4:
                            offer('lk_failure',item)
            p=policy.update(frame,i,logs[i]['candidates'],detected)
            for key in ('bbox','source','events'):assert p[key]==gp[i][key],(i,key)
            previous=frame
    finally:
        cap.release();cv2.calcOpticalFlowPyrLK=original
    selected={}
    for key,items in candidates.items():
        if not items:raise RuntimeError(f'No qualifying actual-video example: {key}')
        # Prefer the user's 14-second area; no method tuning, only post-hoc illustration.
        selected[key]=min(items,key=lambda p:abs(p['frame_id']-421))
    return selected,info,counts


def draw(selected,info,out):
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    font=FontProperties(fname='C:/Windows/Fonts/meiryo.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False})
    plt.rcParams['font.sans-serif']=[font.get_name()]
    def roi(p):
        boxes=[p['old'],p['truth']]
        cx,cy=np.mean([center(b) for b in boxes],axis=0)
        span=max(130,max(max(b[2] for b in boxes)-min(b[0] for b in boxes),max(b[3] for b in boxes)-min(b[1] for b in boxes))+70)
        x=max(0,int(cx-span/2));y=max(0,int(cy-span/2))
        return x,y,min(info['width'],x+int(span)),min(info['height'],y+int(span))
    def show(ax,frame,r,title):
        x,y,x2,y2=r
        ax.imshow(cv2.cvtColor(frame[y:y2,x:x2],cv2.COLOR_BGR2RGB),extent=(x,x2,y2,y))
        ax.set_title(title,fontproperties=font);ax.set_xticks([]);ax.set_yticks([])
    def box(ax,b,color,label=None,style='-'):
        if b is not None:ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=2,linestyle=style,label=label))
    fig,axes=plt.subplots(2,3,figsize=(14,8.4),layout='constrained')
    for row,key in enumerate(('kcf_success','kcf_failure')):
        p=selected[key];i=p['frame_id'];r=roi(p);a,b,c=axes[row]
        show(a,p['previous'],r,f"{'成功例' if row==0 else '失敗例'}：学習する前画像\nf={i-1} / {(i-1)/info['fps']:.3f}秒")
        box(a,p['old'],'#20d9ff','初期位置');a.legend(loc='lower left',prop=font,fontsize=9)
        show(b,p['current'],r,f"次画像：f={i} / {i/info['fps']:.3f}秒\nOpenCV KCF：{'追跡成功' if p['native_ok'] else '失敗を返す（枠なし）'}")
        box(b,p['truth'],'#44ff55','正解');box(b,p['native'],'#ff9b25','実際のKCF枠')
        box(b,p['predicted'],'#ff43d1','説明用KCFの最大応答位置','--')
        b.legend(loc='lower left',prop=font,fontsize=8)
        h,w=p['response'].shape
        im=c.imshow(p['response'],cmap='magma',extent=(-(w//2)-.5,w-w//2-.5,h-h//2-.5,-(h//2)-.5))
        c.scatter(*p['delta'],s=130,marker='x',color='#20d9ff',linewidths=2,label='最大応答')
        shift=np.array(center(p['truth']))-np.array(center(p['old']))
        c.scatter(*shift,s=130,marker='+',color='#44ff55',linewidths=2,label='正解の移動')
        if abs(shift[0])>w/2 or abs(shift[1])>h/2:
            c.annotate('正解は探索範囲の外',xy=shift,xytext=(.03,.86),textcoords='axes fraction',
                       fontproperties=font,fontsize=9,arrowprops=dict(arrowstyle='->',color='#168b36'),color='#168b36')
        c.set_title(f"説明用の相関応答\n最大値={p['peak']:.3f}（確率ではない）",fontproperties=font)
        c.set_xlabel('横方向のずれ [px]',fontproperties=font);c.set_ylabel('縦方向のずれ [px]',fontproperties=font)
        c.legend(loc='lower left',prop=font,fontsize=8);fig.colorbar(im,ax=c,shrink=.68)
    fig.suptitle('KCF：正しい位置に応答の山が出る例 ／ 別の位置が最大になる例',fontproperties=font,fontsize=17)
    fig.supxlabel('実動画 experiment10。右列は輝度特徴だけの説明用KCFによる再計算。OpenCV Eの内部応答そのものではありません。',fontproperties=font,fontsize=10)
    fig.savefig(out/'kcf_success_failure.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8.4),layout='constrained')
    for row,key in enumerate(('lk_success','lk_failure')):
        p=selected[key];i=p['frame_id'];r=roi(p);a,b,c=axes[row];good=p['good']
        show(a,p['previous'],r,f"{'成功例' if row==0 else '失敗例'}：前画像の特徴点\nf={i-1} / {(i-1)/info['fps']:.3f}秒")
        box(a,p['old'],'#20d9ff');a.scatter(*p['points'].T,s=20,c='#20d9ff',edgecolors='black',linewidths=.4)
        show(b,p['current'],r,f"次画像への追跡：f={i} / {i/info['fps']:.3f}秒\n有効点 {int(good.sum())}/{len(good)}、{'LK採用' if p['native'] else 'LK棄却・再検出判断へ'}")
        box(b,p['truth'],'#44ff55','正解');box(b,p['native'],'#ff9b25','LK枠')
        for old,new,valid in zip(p['points'],p['new_points'],good):
            if not np.isfinite(new).all():continue
            color='#21dfff' if valid else '#ff3f58'
            b.annotate('',xy=new,xytext=old,arrowprops=dict(arrowstyle='->',color=color,lw=1.1))
            b.scatter(*new,s=15,color=color)
        b.set_xlim(r[0],r[2]);b.set_ylim(r[3],r[1]);b.legend(loc='lower left',prop=font,fontsize=8)
        finite=np.where(np.isfinite(p['errors']),p['errors'],np.nan)
        c.bar(np.arange(len(good))+1,finite,color=np.where(good,'#1686b0','#e64055'))
        c.axhline(1.5,color='black',ls='--',label='往復誤差の上限 1.5px')
        c.set(xlabel='特徴点番号',ylabel='往復追跡のずれ [px]')
        c.set_title('往復誤差：前→次→前と追って確認',fontproperties=font)
        c.legend(prop=font,fontsize=9)
    fig.suptitle('LK：特徴点がまとまって移動する例 ／ 対応が崩れて棄却する例',fontproperties=font,fontsize=17)
    fig.supxlabel('青：有効点／赤：状態・往復誤差・移動の整合性で除外。採用には4点以上かつ半数以上。実際のLK呼出しから取得し、正解は説明にだけ使用。',fontproperties=font,fontsize=10)
    fig.savefig(out/'lk_success_failure.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/tracking_explanation_20261028')
    a=p.parse_args();out=resolve(a.output);out.mkdir(parents=True,exist_ok=True)
    root=ROOT/'results/comparison_fixed_lag_20261028'
    e_run=next(root.glob('s_E_*'));g_run=next(root.glob('s_G_*'))
    selected,info,counts=collect(e_run,g_run)
    draw(selected,info,out)
    evidence={}
    for key,item in selected.items():
        evidence[key]={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in item.items()
            if k not in ('previous','current','response')}
        if 'response' in item:
            import numpy as np
            np.save(out/f'{key}_response.npy',item['response'])
    save_json(out/'examples.json',dict(video=info,candidates=counts,selected=evidence,
        kcf_scope='Teaching grayscale Gaussian-kernel filter, not OpenCV native response; no confidence interpretation',
        lk_scope='Arrays intercepted from actual policy calls; source/events/boxes matched saved online outputs',
        selection='Post-hoc educational examples, not new accuracy measurements',
        sources=['https://github.com/joaofaro/KCFcpp/blob/master/src/kcftracker.cpp',
                 'https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/src/trackerKCF.cpp']))
    print(json.dumps(dict(output=str(out),frames={k:v['frame_id'] for k,v in selected.items()},counts=counts),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
