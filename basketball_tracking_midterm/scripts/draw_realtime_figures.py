"""Figures for the real-time fixed-delay tracking explanation (reports/2026-11-16-realtime-fixed-delay)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT

SURFACE,INK,INK2,MUTED,GRID,AXIS='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
BLUE,ORANGE,AQUA,YELLOW='#2a78d6','#eb6834','#1baf7a','#eda100'


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Yu Gothic','Meiryo','MS Gothic','sans-serif'],'axes.unicode_minus':False,
        'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
        'xtick.color':MUTED,'ytick.color':MUTED,'text.color':INK,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.8,
        'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'savefig.dpi':160})
    return plt


def box(ax,x,y,w,h,text,colour=BLUE,fill=False,size=9.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',
        linewidth=1.4,edgecolor=colour,facecolor=(colour if fill else SURFACE),alpha=1.))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=('#ffffff' if fill else INK),linespacing=1.4)


def arrow(ax,a,b,colour=INK2,text=None,dx=0.,dy=.12):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.2,color=colour,shrinkA=2,shrinkB=2))
    if text:ax.text((a[0]+b[0])/2+dx,(a[1]+b[1])/2+dy,text,ha='center',va='bottom',fontsize=8.5,color=INK2)


def pipeline(plt,out):
    fig,ax=plt.subplots(figsize=(12.8,7.2));ax.set_xlim(0,12.8);ax.set_ylim(0,7.1);ax.axis('off')
    ax.text(.3,6.75,'処理の構成：3つのスレッドと固定遅延バッファ',fontsize=13,fontweight='bold',color=INK)
    # Capture column
    box(ax,.3,4.5,2.5,.9,'カメラ（DirectShow）\nまたは動画ファイル（30fps到着を模擬）',colour=MUTED)
    box(ax,.3,2.8,2.5,1.1,'取得スレッド\n連番 seq と取得時刻 t を付与\n画像を表示バッファと処理キューへ',colour=BLUE)
    arrow(ax,(1.55,4.5),(1.55,3.9))
    # Processing column
    box(ax,3.4,.55,4.6,5.1,'',colour=GRID)
    ax.text(5.5,5.4,'処理スレッド（1本・到着順に1回ずつ）',ha='center',fontsize=10.5,fontweight='bold',color=INK)
    box(ax,3.6,4.05,3.8,1.05,'バックログ判定\n今 − t > 1.0秒 かつ 新しいフレームが待機\n→ スキップ（空の行だけを後段へ渡す）',colour=ORANGE)
    box(ax,3.6,2.9,3.8,.85,'System 1：局所 Lucas–Kanade ＋ 速度予測\n毎処理フレーム、ボール周辺だけを追跡',colour=BLUE)
    box(ax,3.6,1.8,3.8,.85,'System 2：YOLOv5s（追加学習済み, 960入力）\n失敗時・最大3フレーム間隔で再検出、上限15回/秒',colour=BLUE)
    box(ax,3.6,.7,3.8,.85,'固定遅延補正 bridge（先読み0.4秒）\n前後のYOLO位置・枠サイズを線形補間（最大0.4秒）',colour=AQUA)
    arrow(ax,(2.8,3.35),(3.4,3.35),text='キュー',dy=.08)
    arrow(ax,(5.5,4.05),(5.5,3.75));ax.text(5.6,3.86,'処理する',fontsize=8.5,color=INK2,va='center')
    ax.annotate('',xy=(7.75,1.12),xytext=(7.75,4.05),arrowprops=dict(arrowstyle='-|>',color=ORANGE,lw=1.2))
    ax.text(7.62,2.6,'スキップ：空の行',fontsize=8.5,color=ORANGE,va='center',ha='center',rotation=90)
    arrow(ax,(5.5,2.9),(5.5,2.65))
    arrow(ax,(5.5,1.8),(5.5,1.55))
    # Output column
    box(ax,9.0,2.8,3.5,1.1,'表示バッファ（DelayedOutputStore）\n画像を t + 3.0秒 まで保持し、\n確定した位置と組にして解放',colour=AQUA)
    ax.plot([2.8,3.1,3.1,10.75,10.75],[3.7,3.7,6.1,6.1,3.9],color=INK2,lw=1.2,solid_capstyle='round')
    arrow(ax,(10.75,4.05),(10.75,3.9))
    ax.text(6.9,6.17,'画像（全フレーム、間引きなし）',ha='center',va='bottom',fontsize=8.5,color=INK2)
    ax.annotate('',xy=(9.4,2.8),xytext=(8.0,1.12),arrowprops=dict(arrowstyle='-|>',color=AQUA,lw=1.2,connectionstyle='arc3,rad=-0.2'))
    ax.text(8.05,.8,'位置（seq − 12 の行を出力）',fontsize=8.5,color=AQUA)
    box(ax,9.0,1.3,1.65,.95,'表示スレッド\n(main)\nimshow＋HUD',colour=BLUE)
    box(ax,10.85,1.3,1.65,.95,'録画スレッド\nmp4＋jsonl\n溢れたら録画側のみ落とす',colour=MUTED,size=8.5)
    arrow(ax,(9.8,2.8),(9.8,2.25));ax.text(9.9,2.52,'t + 3.0秒',fontsize=8.5,color=INK2,va='center')
    arrow(ax,(11.65,2.8),(11.65,2.25))
    ax.text(.3,.2,'固定遅延3.0秒のうち、補正に使うのは0.4秒だけ。残り2.6秒が処理の遅れと揺らぎの吸収に使われる。',fontsize=9,color=INK2)
    fig.savefig(out/'pipeline.png',bbox_inches='tight');plt.close(fig)


def timeline(plt,out):
    fig,ax=plt.subplots(figsize=(11,4.3))
    rows=[('フレーム取得（時刻 t）',[(0,.03)],MUTED),
          ('処理待ち（バックログ ≤ 1.0秒）',[(.03,.97)],ORANGE),
          ('System 1 / 2 の処理（通常 10〜40ms）',[(1.0,.04)],BLUE),
          ('先読み待ち：12フレーム後の処理を待つ（0.4秒）',[(1.04,.4)],BLUE),
          ('bridge で位置確定（t + 約1.45秒）',[(1.44,.03)],AQUA),
          ('余裕（処理の揺らぎを吸収）',[(1.47,1.53)],GRID),
          ('表示（t + 3.0秒、固定）',[(3.0,.03)],AQUA)]
    for k,(label,spans,colour) in enumerate(rows):
        y=len(rows)-1-k
        ax.broken_barh(spans,(y-.3,.6),facecolors=colour,edgecolors=SURFACE,linewidth=1)
        ax.text(-.05,y,label,ha='right',va='center',fontsize=9.5,color=INK)
    ax.axvline(3.0,color=AXIS,lw=1);ax.text(3.0,len(rows)-.45,'表示期限',ha='center',fontsize=9,color=INK2)
    ax.set_yticks([]);ax.set_xlim(-.02,3.25);ax.set_ylim(-.6,len(rows)-.2)
    ax.set_xlabel('取得からの経過時間 [秒]（最悪ケース：バックログ上限に張り付いた状態）')
    ax.grid(axis='y',visible=False);ax.spines['left'].set_visible(False)
    ax.set_title('1フレームの時間線：処理が1秒遅れていても、3.0秒の表示時刻には位置が確定している',fontsize=11,loc='left',color=INK)
    fig.savefig(out/'timeline.png',bbox_inches='tight');plt.close(fig)


def session(plt,out,run):
    rows=[json.loads(l) for l in (run/'frames.jsonl').open(encoding='utf-8')]
    summary=json.loads((run/'session.json').read_text(encoding='utf-8'))
    t=[r['capture_s'] for r in rows];delay=[r['display_delay_s'] for r in rows]
    seconds=sorted({int(x) for x in t})
    processed={s:[r['processed'] for r in rows if int(r['capture_s'])==s] for s in seconds}
    rate=[100*sum(v)/len(v) for s,v in ((s,processed[s]) for s in seconds)]
    fig,(a,b)=plt.subplots(2,1,figsize=(10,5.6),sharex=True,gridspec_kw=dict(hspace=.35))
    a.plot(t,delay,color=BLUE,lw=1.6)
    a.axhline(summary['display_delay_target_s'],color=AXIS,lw=1)
    a.set_ylim(2.9,3.3);a.set_ylabel('取得→表示の遅延 [秒]')
    a.set_title(f"表示遅延：中央値 {summary['display_delay_s']['median']:.3f}秒、p95 {summary['display_delay_s']['p95']:.3f}秒、最大 {max(delay):.3f}秒（目標3.0秒）",loc='left',fontsize=10.5)
    b.bar(seconds,rate,width=.9,color=BLUE,edgecolor=SURFACE,linewidth=1)
    b.set_ylim(0,100);b.set_ylabel('処理したフレーム [%]');b.set_xlabel('取得時刻 [秒]')
    tr=summary['tracking']
    b.set_title(f"処理した割合（残りはスキップし bridge で補間）：全体 {100*(1-tr['drop_rate']):.1f}%、処理 {tr['processing_ms']['median']:.0f}ms/フレーム（中央値）",loc='left',fontsize=10.5)
    for ax in (a,b):ax.grid(axis='x',visible=False)
    fig.savefig(out/'session_battery.png',bbox_inches='tight');plt.close(fig)


def accuracy(plt,out,path):
    items=json.loads(Path(path).read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(9,.7*len(items)+1.4))
    labels=[i['label'] for i in items];values=[100*i['tsr05'] for i in items]
    y=list(range(len(items)))[::-1]
    ax.barh(y,values,height=.55,color=[i.get('colour',BLUE) for i in items],edgecolor=SURFACE,linewidth=1)
    for yy,v in zip(y,values):ax.text(v+.4,yy,f'{v:.2f}%',va='center',fontsize=9.5,color=INK)
    ax.set_yticks(y);ax.set_yticklabels(labels,fontsize=9.5,color=INK);ax.set_xlim(0,100);ax.set_xlabel('TSR@0.5 [%]（experiment10、可視1,883フレーム）')
    ax.grid(axis='y',visible=False);ax.spines['left'].set_color(AXIS)
    ax.set_title('位置精度：間引きが増えても補間で緩やかに低下する',loc='left',fontsize=11)
    fig.savefig(out/'accuracy.png',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/2026-11-16-realtime-fixed-delay')
    p.add_argument('--session-run',type=Path,help='realtime_<timestamp> folder with frames.jsonl and session.json')
    p.add_argument('--accuracy',type=Path,help='JSON list of {label, tsr05}')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt=setup()
    pipeline(plt,a.output);timeline(plt,a.output)
    if a.session_run:session(plt,a.output,a.session_run)
    if a.accuracy:accuracy(plt,a.output,a.accuracy)
    print('figures written to',a.output)
