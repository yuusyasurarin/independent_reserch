"""Summarize the matched B/D/E/G runs and render source-time-aligned comparisons."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, resolve, save_csv, save_json, sha256
from src.common import center, iou
from src.data import load_gt
from scripts.research_g import read_jsonl


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def comparison_video(runs,output,size):
    import cv2
    import numpy as np
    cv2.setNumThreads(0)
    caps=[cv2.VideoCapture(str(runs[c]/'tracked.mp4')) for c in 'BDEG']
    info=read(runs['B']/'run_config.json')['video']
    n,fps=info['frame_count'],info['fps']
    assert all(c.isOpened() for c in caps)
    output.mkdir(parents=True,exist_ok=True)
    full=output/f'{size}_B_D_E_G.mp4';clip=output/f'{size}_12_to_16_seconds.mp4'
    writers=[cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720)) for p in (full,clip)]
    assert all(w.isOpened() for w in writers)
    try:
        for i in range(n):
            panels=[]
            for condition,cap in zip('BDEG',caps):
                ok,frame=cap.read();assert ok
                panel=cv2.resize(frame,(640,360))
                panel[:38]=0
                lag=0 if condition in 'BD' else 2
                method={'B':'YOLO every frame','D':'YOLO + hold','E':'corrected KCF + reverse KCF','G':'adaptive LK + bridge'}[condition]
                cv2.putText(panel,f'{size}/{condition} | {method} | lag={lag}s',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                cv2.putText(panel,f'f={i} source={i/fps:.3f}s | source-time aligned, not release time',(8,32),cv2.FONT_HERSHEY_SIMPLEX,.39,(220,220,220),1)
                panels.append(panel)
            canvas=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
            writers[0].write(canvas)
            if 12<=i/fps<16:writers[1].write(canvas)
    finally:
        for cap in caps:cap.release()
        for w in writers:w.release()
    checked=[]
    for path,count in ((full,n),(clip,round(4*fps))):
        cap=cv2.VideoCapture(str(path));decoded=0
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==fps
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);decoded+=1
        cap.release();assert decoded==count
        checked.append(dict(path=path.relative_to(ROOT).as_posix(),frames=decoded,sha256=sha256(path)))
    return checked


def summarize(args):
    root=resolve(args.root);output=resolve(args.output)
    output.mkdir(parents=True,exist_ok=True)
    verified=read(root/'verification.json');assert verified['status']=='passed'
    runs={};rows=[];stages=[];latencies={};comparability={};diagnostics=[]
    for manifest in sorted(root.glob('*/run_config.json')):
        run=manifest.parent;m=read(manifest);metrics=read(run/'metrics.json');p=metrics['performance']
        size,condition=m['model_size'],m['condition']
        assert condition in 'BDEG' and metrics['complete_video']
        assert (size,condition) not in runs;runs[size,condition]=run
        assert read(run/'verification.json')['status']=='passed'
        signature={k:m['config'][k] for k in ('imgsz','confidence','candidate_conf','nms_iou','half','device')}
        signature.update(video=m['video']['sha256'],gt=m['gt_sha256'],weights=m['weights']['sha256'])
        sources={k.replace('\\','/'):v for k,v in m['source_sha256'].items()}
        signature['core_sources']={k:sources[k] for k in ('src/tracking.py','src/adaptive_tracking.py',
            'src/lag_tracking.py','src/detector.py','scripts/run_fixed_lag.py')}
        if size in comparability:assert signature==comparability[size]
        else:comparability[size]=signature
        with (run/'release.csv').open(encoding='utf-8-sig',newline='') as f:release=list(csv.DictReader(f))
        latency=[float(r['simulated_arrival_to_release_s']) for r in release]
        latencies[size,condition]=latency
        t=metrics['tracking'];online=metrics['online_tracking']
        row=dict(size=size,condition=condition,method=m['method'],lag_s=m['lag_s'],
            tsr05=t['tsr_iou_05'],tsr03=t['tsr_iou_03'],mean_iou=t['mean_iou'],
            output_rate=t['output_rate_visible'],cle_mean_px=t['cle_px']['mean'],cle_n=t['cle_px']['n'],
            longest_failure_s=t['longest_failure_at_05_s'],online_tsr05=online['tsr_iou_05'],
            correction_gain_pp=100*(t['tsr_iou_05']-online['tsr_iou_05']),
            yolo_calls=p['yolo_calls'],yolo_hz=p['realized_yolo_calls_per_video_second'],
            processing_fps=p['processing_fps'],wall_fps=p['wall_fps'],
            latency_median=p['simulated_release_latency_s']['median'],latency_p95=p['simulated_release_latency_s']['p95'],
            latency_max=max(latency),latency_first=latency[0],latency_last=latency[-1],
            frontend_ms=p['frontend_ms']['mean'],backend_ms=p['backend_ms']['mean'],
            display_buffer_bytes=p['peak_rgb_buffer_bytes'],process_peak_bytes=p['peak_process_rss_or_peak_wset_bytes'],
            gpu_peak_allocated_bytes=p['gpu_peak_allocated_bytes'],
            run=run.relative_to(ROOT).as_posix(),video=(run/'tracked.mp4').relative_to(ROOT).as_posix())
        rows.append(row)
        dest=output/f'{size}_{condition}';dest.mkdir(exist_ok=True)
        for name in ('run_config.json','metrics.json','status.json','verification.json'):
            shutil.copy2(run/name,dest/name)
        if condition=='E':
            old_runs=list((ROOT/'results/comparison_20261027').glob(f'{size}_E_*'));assert len(old_runs)==1
            old=old_runs[0];old_m=read(old/'run_config.json');old_v=read(old/'metrics.json')
            assert old_m['weights']['sha256']==m['weights']['sha256']
            before=read_jsonl(old/'candidates.jsonl');after=read_jsonl(run/'candidates.jsonl')
            same=sum(a['candidates']==b['candidates'] for a,b in zip(before,after))
            stages.append(dict(size=size,old_buggy_online_tsr05=old_v['tracking']['tsr_iou_05'],
                corrected_online_tsr05=online['tsr_iou_05'],corrected_bidirectional_tsr05=t['tsr_iou_05'],
                exact_candidate_frame_matches=same,frames=len(after),old_run=old.relative_to(ROOT).as_posix(),
                warning='Old E is a bug-affected historical record, not a valid method baseline. Do not compare old speed.'))
            gt=load_gt(Path(m['ground_truth']),m['video'])
            versions={'old_buggy':read_jsonl(old/'tracking.jsonl'),
                'corrected_online':read_jsonl(run/'online.jsonl'),'corrected_bidirectional':read_jsonl(run/'tracking.jsonl')}
            for label,trace in versions.items():
                for pred in trace:
                    i=pred['frame_id']
                    if not 13<=i/m['video']['fps']<=15:continue
                    box=pred['bbox'];truth=gt[i]['bbox'];xy=center(box) if box else (None,None)
                    diagnostics.append(dict(size=size,version=label,frame_id=i,time_s=i/m['video']['fps'],
                        source=pred['source'],center_x=xy[0],center_y=xy[1],
                        iou=iou(box,truth) if box and truth else (0. if truth else None),
                        bbox=json.dumps(box),gt_bbox=json.dumps(truth)))
    assert set(runs)=={(s,c) for s in ('s','m') for c in 'BDEG'}
    rows.sort(key=lambda r:(r['size']!='s','BDEG'.index(r['condition'])))
    save_csv(output/'comparison.csv',rows);save_json(output/'comparison.json',dict(rows=rows,comparable_settings= comparability))
    save_csv(output/'e_fix_and_smoothing.csv',stages)
    save_csv(output/'e_14s_diagnostic.csv',diagnostics)
    for name in ('comparison_plan.json','verification.json'):shutil.copy2(root/name,output/name)
    if (root/'hardware_observation.json').is_file():
        save_json(output/'hardware_observation.json',json.loads((root/'hardware_observation.json').read_text(encoding='utf-8-sig')))
    if (root/'power_context.json').is_file():
        save_json(output/'power_context.json',json.loads((root/'power_context.json').read_text(encoding='utf-8-sig')))
    if (root/'interruption.json').is_file():
        save_json(output/'interruption.json',json.loads((root/'interruption.json').read_text(encoding='utf-8-sig')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for size,color in (('s','#2675b7'),('m','#d27916')):
        r=[x for x in rows if x['size']==size];offset=-.18 if size=='s' else .18
        axes[0].bar([i+offset for i in range(4)],[100*x['tsr05'] for x in r],width=.36,label=f'YOLOv5{size}',color=color)
    axes[0].set(xticks=range(4),xticklabels=['B','D','E + 2s','G + 2s'],ylabel='TSR@0.5 (%)',ylim=(0,100),title='All 1883 visible frames')
    axes[0].legend()
    for size,style in (('s','-'),('m','--')):
        for condition in 'BDEG':
            a=latencies[size,condition]
            axes[1].plot([i/30 for i in range(len(a))],a,style,label=f'{size}/{condition}',linewidth=.8)
    axes[1].set(xlabel='Source time (s)',ylabel='Simulated release latency (s)',title='Includes compute backlog')
    axes[1].legend(ncol=2,fontsize=8)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    videos=[]
    if not args.no_mosaic:
        for size in ('s','m'):
            videos.extend(comparison_video({c:runs[size,c] for c in 'BDEG'},root/'comparison_videos',size))
    save_json(output/'comparison_video_verification.json',dict(videos=videos))
    lines=['# B・Dと固定遅延ありE・Gの比較（2026-10-28）','',
        'EのKCF初期化を修正したうえで、experiment10全2,512フレームをs/m各モデルで計8回、新たに推論しました。主な対象はRTX 3050 Ti Laptop GPU 4GBです。',
        '可視1,883フレームを共通分母とし、元フレームの時刻に対応する推定位置を評価しています。遅れた枠を正解と合わせるために時刻をずらす処理はしません。','',
        '## 事前に固定した条件','',
        '| 条件 | 前向き処理 | 未来情報の利用 | 設定遅延 |','|---|---|---|---:|',
        '| B | 毎フレームYOLO | なし | 0秒 |',
        '| D | 3フレームごとのYOLO＋位置保持 | なし | 0秒 |',
        '| E | 3フレームごとのYOLO＋修正済みKCF | 後のYOLO位置から逆向きKCF。前向きとの整合性を確認して融合 | 2秒 |',
        '| G | 適応的YOLO＋局所LK＋速度予測（fプロファイル） | 前後のYOLO位置・枠サイズを短区間補間（bridge） | 2秒 |','',
        '共通：学習済み同サイズの重み、imgsz=960、confidence=0.25、NMS IoU=0.45、FP32、同じ入力動画・正解、動画とログ保存あり。30fpsファイル到着を模擬し、同じrun_fixed_lag.pyで実行しました。',
        'E/Gの補正区間は最大0.4秒、先読み上限は2秒です。Eの逆向きKCFは強いYOLO検出で初期化し、初回updateをその画像上で済ませます。逆追跡が失敗した部分は前向き結果を残します。',
        '前向き枠がある場合、逆追跡が前のYOLO位置まで戻り、中心誤差がmax(6px, 前の枠の最大辺×0.75)以下なら位置とサイズを時間比率で融合します。前向き出力がない位置は成功した逆追跡で補います。長い遮蔽の真の位置を保証する処理ではありません。',
        '実行順はs/B→s/D→s/E→s/G→m/B→m/D→m/E→m/G、各1回です。キャッシュ済みYOLOによる速度測定ではありません。','',
        'm/Gの初回実行は途中で異常終了し、後の確認時にはDドライブが認識されませんでした。因果関係は未確定です。途中記録を別フォルダへ保存し、再接続後の別セッションでm/Gを完走させた結果を掲載します。前7条件とm/Gの速度差にはセッションの違いも含まれるため、厳密に同時条件での速度比較とは扱いません。詳細はinterruption.jsonです。','',
        '## 全動画の結果','',
        '| モデル | 条件 | TSR@0.5 | 補正前TSR | YOLO回数 | 処理fps | 遅延中央値 / p95 / 最大 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['size']} | {r['condition']} | {r['tsr05']*100:.2f}% | {r['online_tsr05']*100:.2f}% | {r['yolo_calls']} | {r['processing_fps']:.2f} | {r['latency_median']:.3f} / {r['latency_p95']:.3f} / {r['latency_max']:.3f}秒 |")
    lines+=['','補正前TSRは同じ実行のオンライン出力です。Eでは初期化修正済み・未来補正なしのKCF、Gでは未来補正なしのF相当の結果です。B/Dでは補正しないため一致します。',
        '処理fpsは読み込み、推論、前向き追跡、逆追跡/補間、描画・保存・ログ、末尾の出力を含み、明示的な待ち時間を除きます。モデル初期化・ウォームアップ3回・終了後の評価は含みません。',
        '遅延は元フレームの到着予定から動画書込み完了までです。計算が30fpsに追いつかなければ蓄積する遅れも含みます。設定遅延0秒は実遅延0秒の意味ではなく、2秒も厳密な表示期限の保証ではありません。カメラ・画面表示までの遅延は未測定です。','',
        '今回はユーザー確認済みのバッテリー駆動です（power_context.json）。以前のG計測より低速で、実行中のGPU状態もhardware_observation.jsonに1回記録しています。速度と遅延はこのバッテリー駆動セッションの値で、AC電源時の性能を示しません。低速化の全要因を分離した実験ではなく、過去のFPSと合算・直接比較しません。','',
        '![精度と遅延](comparison.png)','','## Eの修正と未来補正の効果','',
        '| モデル | 旧E（不具合あり・参考） | 修正済みEの前向き出力 | 逆向きKCFで補正後 |',
        '|---|---:|---:|---:|']
    for r in stages:
        lines.append(f"| {r['size']} | {r['old_buggy_online_tsr05']*100:.2f}% | {r['corrected_online_tsr05']*100:.2f}% | {r['corrected_bidirectional_tsr05']*100:.2f}% |")
    for r in stages:
        lines.append(f"\n{r['size']}: 初期化修正前後の保存YOLO候補が完全一致したフレームは{r['exact_candidate_frame_matches']}/{r['frames']}。修正済み前向きEに対する未来補正の差は{100*(r['corrected_bidirectional_tsr05']-r['corrected_online_tsr05']):+.2f}ポイントです。")
    lines+=['','OpenCV 4.10.0のKCFは最初のupdateで特徴を学習します。従来のEはinitの次の画像で初回updateを呼び、移動前のROIを移動後の画像で学習していました。検出画像上で初回updateを行うよう修正し、次フレームから探索させます。YOLOの検出枠自体はこの初回updateでは変更しません。',
        '初期化の修正は追跡処理の正しさを直すもので、全動画のTSRが必ず向上するとは限りません。動きが大きい場面で正しく追跡失敗を返す場合もあります。未来補正の効果とは分けて解釈します。',
        '移動する合成画像で、初期化直後・再初期化直後の1フレーム目から追跡できる回帰テストを追加しました。旧結果は削除・上書きせず、不具合を含む履歴として扱います。旧Eの速度や、旧Eに対する過去の改善倍率を今回の比較根拠にしません。','',
        '[14秒前後の診断CSV](e_14s_diagnostic.csv) に、旧E・修正済み前向きE・逆向き補正後Eの座標とIoUを保存しています。','',
        '## 再実行・検証','',
        'プロジェクト直下のPowerShellで実行します。出力先は新しいフォルダ名にしてください。','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/compare_fixed_lag.py --output results/comparison_fixed_lag_new',
        '.\\.venv\\Scripts\\python.exe scripts/verify_g.py --fresh-root results/comparison_fixed_lag_new --output results/comparison_fixed_lag_new/verification.json',
        '.\\.venv\\Scripts\\python.exe scripts/summarize_fixed_lag.py --root results/comparison_fixed_lag_new --output reports/comparison_fixed_lag_new',
        '```','',
        '本体は `src/tracking.py`（Eの修正）、`src/lag_tracking.py`（逆向きKCFと固定遅延補正）、`scripts/run_fixed_lag.py`（共通測定）です。設定・コマンドは各run_config.jsonとcomparison_plan.jsonに保存します。',
        '保存候補と元動画だけから前向き処理・未来補正を再実行し、全フレームの座標、判断、時刻境界、検出スケジュールを照合しました。全出力動画を最後までデコードし、指標も再計算しています。正解は描画と事後評価だけに使用します。','',
        '## 動画とデータ','',
        '各動画は同じ元時刻の映像に推定枠を重ねています。冒頭の待ち時間や計算の遅れを映像に挿入した動画ではありません。実際の模擬出力時刻はrelease.csvを参照してください。','']
    for r in rows:lines.append(f"- {r['size']}/{r['condition']}: `{r['video']}`")
    if videos:
        lines+=['','4分割比較動画は左上B・右上D・左下E・右下G。同じ元フレームで並べた位置精度の比較で、出力遅延を同期比較する動画ではありません。','']
        lines.extend(f"- `{v['path']}`" for v in videos)
    lines+=['','[比較CSV](comparison.csv)、[全検証記録](verification.json)、[E修正の比較](e_fix_and_smoothing.csv) と各条件の設定・集計値をGitに保存します。重み、全フレームログ、動画はローカル保存です。','',
        '## 解釈の制約','',
        '- experiment10で方式を設計した開発比較です。独立した汎化評価ではありません。',
        '- E/Gは未来情報を使える代わりに2秒の遅延を許す条件で、B/Dと同じ応答性ではありません。精度だけで優劣を判断しません。',
        '- Eは838回の周期検出、Gは適応検出なので回数が異なります。これはシステム全体の比較であり、同一検出候補を使ったKCF対LKだけの比較ではありません。',
        '- 629不可視フレームでは真の位置が不明です。不可視区間の出力の正しさは評価できません。',
        '- E/Gの末尾60フレームは未来が不足しfull_lookahead=false。主指標は末尾を含む同じ1,883可視フレームで算出し、完全な先読みがある範囲の指標も別に保存します。',
        '- 位置を出さない場合はTSRの失敗に含めます。CLEは位置を出した可視フレームだけの値なので、方式間で分母が異なります。',
        '- 表示用RGBバッファは2秒条件で最大61枚です。逆KCFは別に最大13枚のRGBを持ちます。プロセスメモリはモデルや全動画分の評価ログも含むため、長時間配信のメモリ一定性は未検証です。',
        '- 電力・温度・実行順を完全には統制していない各1回の測定です。速度差の統計的優位性や15分連続運転の保証はありません。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Published',len(rows),'runs to',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/comparison_fixed_lag_20261028')
    p.add_argument('--output',default='reports/2026-10-28-fixed-lag-comparison')
    p.add_argument('--no-mosaic',action='store_true')
    summarize(p.parse_args())
