"""Publish compact G evidence; raw videos and frame logs remain local."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_csv, save_json


def summarize(root,output):
    root=root.resolve();output=output.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    verification=read(root/'verification.json')
    assert verification['status']=='passed'
    shutil.copy2(root/'verification.json',output/'verification.json')
    cached=[]
    for sweep in ('sweep_2s','sweep_bidir','sweep_lag'):
        cached.extend(read(root/sweep/'comparison.json')['rows'])
        for run in sorted((root/sweep).iterdir()):
            if not run.is_dir():continue
            dest=output/'cached'/sweep/run.name;dest.mkdir(parents=True,exist_ok=True)
            for name in ('manifest.json','metrics.json'):shutil.copy2(run/name,dest/name)
    save_csv(output/'cached_comparison.csv',cached)
    save_json(output/'cached_comparison.json',dict(rows=cached))
    fresh=[]
    for run in sorted((root/'fresh').iterdir(),key=lambda p:p.name.rsplit('_',1)[-1]):
        if not run.is_dir():continue
        m=read(run/'run_config.json');v=read(run/'metrics.json');p=v['performance']
        dest=output/'fresh'/run.name;dest.mkdir(parents=True,exist_ok=True)
        for name in ('run_config.json','metrics.json','status.json','verification.json'):
            shutil.copy2(run/name,dest/name)
        with (run/'release.csv').open(encoding='utf-8-sig',newline='') as f:release=list(csv.DictReader(f))
        latency=[float(r['simulated_arrival_to_release_s']) for r in release if r['simulated_arrival_to_release_s']]
        row=dict(run=run.name,device=m['config']['device'],imgsz=m['config']['imgsz'],profile=m['profile'],
            method=m['method'],lag_s=m['lag_s'],paced=m['realtime_pacing'],tsr05=v['tracking']['tsr_iou_05'],
            online_tsr05=v['online_tracking']['tsr_iou_05'],yolo_calls=p['yolo_calls'],
            yolo_hz=p['realized_yolo_calls_per_video_second'],processing_fps=p['processing_fps'],wall_fps=p['wall_fps'],
            backend_mean_ms=p['backend_ms']['mean'],peak_rgb_buffer_bytes=p['peak_rgb_buffer_bytes'],
            peak_process_bytes=p['peak_process_rss_or_peak_wset_bytes'],gpu_peak_allocated_bytes=p['gpu_peak_allocated_bytes'],
            latency_median=p['simulated_release_latency_s']['median'],latency_p95=p['simulated_release_latency_s']['p95'],
            latency_max=max(latency) if latency else None,latency_first=latency[0] if latency else None,
            latency_last=latency[-1] if latency else None,
            video=str((run/'tracked.mp4').relative_to(ROOT)).replace('\\','/'))
        fresh.append(row)
    save_csv(output/'fresh_comparison.csv',fresh);save_json(output/'fresh_comparison.json',dict(rows=fresh))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ('forward','bridge','rts','beam','bidir'):
        rows=sorted([r for r in cached if r['method']==method and r['lag_s']==2],key=lambda r:r['yolo_calls'])
        axes[0].plot([r['yolo_calls'] for r in rows],[100*r['tsr05'] for r in rows],'o-',label=method)
    axes[0].axhline(81.46574615,color='black',linestyle=':',label='B reference')
    axes[0].set(xlabel='YOLO calls / 2512 frames',ylabel='TSR@0.5 (%)',title='2 s lookahead: matched cached inputs')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for profile in ('f','budget10'):
        for method in ('bridge','bidir'):
            rows=sorted([r for r in cached if r['method']==method and r['profile']==profile],key=lambda r:r['lag_s'])
            axes[1].plot([r['lag_s'] for r in rows],[100*r['tsr05'] for r in rows],'o-',label=f'{profile}/{method}')
    axes[1].set(xlabel='Lookahead (s)',ylabel='TSR@0.5 (%)',title='Short-gap correction limited to 0.4 s')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    lines=['# 条件G：固定遅延とSystem 1 / System 2の実験記録','',
        f'experiment10全2,512フレーム、可視位置を評価できる1,883フレームを共通分母としたTSR@0.5。候補再生による{len(cached)}比較と、実際にYOLOを呼び出した{len(fresh)}回の全動画実験。',
        '実験設計・解釈・再実行手順は [プロジェクトREADMEの条件G](../../README.md#10-条件g固定遅延とsystem-1--system-2) を参照。','',
        '## 遅延2秒の方式比較（検出候補を再利用）','',
        '| 検出予算 | YOLO回数 | 遅延のみ | 短区間補間 | RTS | Beam | 双方向LK |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for profile in ('f','budget10','budget5'):
        rows={r['method']:r for r in cached if r['profile']==profile and r['lag_s']==2}
        values=' | '.join(f"{100*rows[k]['tsr05']:.2f}%" for k in ('forward','bridge','rts','beam','bidir'))
        lines.append(f"| {profile} | {rows['forward']['yolo_calls']} | {values} |")
    lines+=['','キャッシュ比較の時間は後処理だけで、YOLO推論込みのFPSではありません。比較した入力が同じ方式間では、YOLO回数は完全に同じです。','',
        '![方式と遅延長の比較](comparison.png)','','## 実推論・全動画測定','',
        '| 構成 | ペーシング | 補正前TSR | 補正後TSR | YOLO回数 | 処理fps | 遅延中央値 / p95 / 最大 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in fresh:
        latency='未測定' if not r['paced'] else f"{r['latency_median']:.3f} / {r['latency_p95']:.3f} / {r['latency_max']:.3f}秒"
        lines.append(f"| {r['device']} / {r['imgsz']} / {r['profile']} / {r['method']} | {'30fps入力' if r['paced'] else '最速'} | {100*r['online_tsr05']:.2f}% | {100*r['tsr05']:.2f}% | {r['yolo_calls']} | {r['processing_fps']:.2f} | {latency} |")
    lines+=['','処理fpsは明示的な待ち時間を除いた値です。ペーシングありの測定は30fpsのファイル到着を模擬し、元フレームの到着予定時刻から動画書込み完了までを計測しています。カメラ入力・画面表示までの遅延ではありません。',
        '読み込み、LK、YOLO、補正、描画、ログ・動画保存、末尾出力を含みます。モデル初期化、3回のウォームアップ、終了後の評価は除きます。',
        '表は実行順です。過去のB/E/Fとは計測セッションと計測処理が異なるため、過去のFPSから速度向上率を算出しません。各構成1回ずつの測定であり、統計的な速度差や長時間安定性を示すものではありません。','',
        '## 動画と監査記録','',
        '動画はローカルの以下のパスです。各動画は元と同じ30fps・2,512フレームで、表示位置の時刻に対応します。冒頭に2秒の黒画面を挿入した映像ではありません。遅延の証拠は各runのrelease.csvです。','']
    lines.extend(f"- `{r['video']}`" for r in fresh)
    lines+=['','[検証記録](verification.json)、[キャッシュ比較CSV](cached_comparison.csv)、[実推論CSV](fresh_comparison.csv) を同梱。各runの設定・指標・ソースハッシュは `cached/` と `fresh/` に保存しています。',
        '動画・全フレームログ・重みは大きいためGitに含めません。正解は追跡処理に渡さず、描画と事後評価だけに使用しました。',
        '検証では全動画を最後までデコードし、保存候補だけから因果的な呼び出し判断と固定遅延出力を再現し、位置・イベント・時刻境界・予算・指標の一致を確認しました。','',
        '## 解釈の範囲','',
        '- experiment10を見て方式を選んだ開発実験です。独立した最終評価でも、汎化性能の証明でもありません。',
        '- 不可視629フレームは確認済みですが、真のボール位置は不明です。そこでの推定位置の正確さは検証できません。',
        '- 2秒先まで参照できても、bridge/bidirが補う区間は最大0.4秒です。この実装での遅延長の頭打ちを、一般の平滑化の限界とは解釈しません。',
        '- 末尾60フレームは2秒分の未来がないためfull_lookahead=false。主要TSRは末尾を含む共通1,883可視フレームで算出し、完全な先読みがある範囲の指標もmetrics.jsonに分けて保存しています。',
        '- CPUは4スレッド。640/416は追加の開発用入力解像度で、valで選ばれた設定ではありません。既存の重み・final_selection.jsonは変更していません。',
        '- 2秒のRGB表示バッファは最大61枚、168,652,800 bytesです。モデル等を含むプロセスメモリは別に記録しています。評価用ログは全動画分を保持するため、長時間配信でメモリ一定とは主張しません。',
        '- 保持選手・姿勢・3D軌道は未使用です。LKは局所追跡であり、画面全体から新規ボールを発見する検出器ではありません。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Published',len(cached),'cached trials and',len(fresh),'fresh runs to',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/condition_g')
    p.add_argument('--output',type=Path,default=ROOT/'reports/2026-10-27-condition-g')
    a=p.parse_args();summarize(a.root,a.output)
