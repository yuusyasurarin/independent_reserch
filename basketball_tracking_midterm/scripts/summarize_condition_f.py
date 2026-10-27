"""Publish compact, measured B/E/F evidence without copying videos or weights."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_json, save_csv, sha256
from scripts.compare_results import compare


def summarize(runs,output):
    compare(runs,output)
    rows=[];phases=[];details={}
    for run in runs:
        read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
        manifest,metrics,verification=read('run_config.json'),read('metrics.json'),read('verification.json')
        assert metrics['complete_video'] and verification['status']=='passed'
        size,condition=manifest['model_size'],manifest['condition']
        key=f'{size}_{condition}'
        if key in details:raise ValueError(f'Duplicate result: {key}')
        target=output/key;target.mkdir()
        for name in ('run_config.json','metrics.json','status.json','summary.csv','verification.json','telemetry.json'):
            shutil.copy2(run/name,target/name)
        with (run/'tracking.jsonl').open(encoding='utf-8') as f:
            predictions=[json.loads(line) for line in f]
        with (run/'per_frame.csv').open(encoding='utf-8-sig',newline='') as f:
            per_frame=list(csv.DictReader(f))
        for detected in (True,False):
            group=[r for r,p in zip(per_frame,predictions) if r['eligible']=='1' and p['yolo_executed']==detected]
            phases.append(dict(size=size,condition=condition,yolo_executed=detected,eligible_frames=len(group),
                tsr05=sum(int(r['success_05']) for r in group)/len(group) if group else None,
                missing=sum(r['output']=='0' for r in group)))
        details[key]=dict(run=run.resolve().relative_to(ROOT).as_posix(),
            video=(run/'tracked.mp4').resolve().relative_to(ROOT).as_posix(),
            sources=dict(Counter(p['source'] for p in predictions)),
            events=dict(Counter(e for p in predictions for e in p['events'])),
            source_hashes=manifest['source_sha256'])
        historical=ROOT/'reports/2026-10-27'/key/'metrics.json'
        if condition in ('B','E') and historical.is_file():
            prior=json.loads(historical.read_text(encoding='utf-8'))
            details[key]['historical_accuracy_identical']={
                section:prior[section]==metrics[section] for section in ('tracking','detection')}
        t,p=metrics['tracking'],metrics['performance']
        rows.append(dict(size=size,condition=condition,tsr05=t['tsr_iou_05'],tsr03=t['tsr_iou_03'],
            output_rate=t['output_rate_visible'],cle_mean=t['cle_px']['mean'],cle_n=t['cle_px']['n'],
            longest_failure=t['longest_failure_at_05_s'],fps=p['processing_fps'],calls=p['yolo_calls'],
            tracking_ms=p['tracking_ms_all_frames']['mean']))
    assert set(details)=={f'{s}_{c}' for s in ('s','m') for c in ('B','E','F')}
    rows.sort(key=lambda r:(r['size']=='m',r['condition']))
    save_csv(output/'detection_phase.csv',phases)
    save_json(output/'evidence.json',dict(development_only=True,
        video='experiment10; previously inspected while designing F',
        no_independent_generalization_claim=True,run_order=[r.name for r in runs],runs=details,
        f_config_sha256=sha256(ROOT/'configs/experiment_f.json')))
    lines=['# 条件Fの開発実験（experiment10）','',
        '既に結果を確認したexperiment10で方式Fを設計・検証した開発実験です。独立した最終評価ではありません。',
        's/mのF→B→Eを逐次実行しました。各条件1回であり、実行順・温度・他プロセスの影響を分離していません。','',
        '| モデル | 条件 | TSR@0.3 | TSR@0.5 | 出力率 | CLE平均px（件数） | 最長失敗秒 | 処理fps | YOLO回数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['size']} | {r['condition']} | {r['tsr03']:.2%} | {r['tsr05']:.2%} | {r['output_rate']:.2%} | {r['cle_mean']:.2f} ({r['cle_n']}) | {r['longest_failure']:.3f} | {r['fps']:.2f} | {r['calls']} |")
    lines+=['','TSRは全1,883可視フレームを共通分母とし、未出力も失敗に含めます。629不可視フレームの位置精度は評価しません。',
        'CLEは出力がある可視フレームだけの平均で、誤対象の出力も含みます。未出力を0pxとして扱いません。','']
    for size in ('s','m'):
        by={r['condition']:r for r in rows if r['size']==size}
        b,e,f=(by[c] for c in ('B','E','F'))
        lines.append(f"- **{size}/F**：TSR@0.5はB比 {(f['tsr05']-b['tsr05'])*100:+.2f}ポイント、E比 {(f['tsr05']-e['tsr05'])*100:+.2f}ポイント。処理速度はBの {f['fps']/b['fps']:.2f}倍、Eの {f['fps']/e['fps']:.2f}倍。YOLO回数はB比 {1-f['calls']/b['calls']:.2%}減。")
    lines+=['','## 方法と制約','',
        '- Fは局所Lucas–Kanade特徴点追跡、往復誤差チェック、速度予測、予測位置に基づく候補選択、予算付き適応再検出を組み合わせます。',
        '- 基本間隔3フレーム。30fps映像の任意の連続30フレームでYOLOは最大15回。実時間の1秒あたり呼び出し上限とは異なります。',
        '- 特徴点不足・急な動きでは前倒し再検出。予算不足なら延期し、予測は最大2フレーム、高信頼度YOLOなしの出力は最大0.3秒に制限します。',
        '- 低信頼度候補は既存軌跡の近傍だけで救済し、新規開始や高信頼度検出の期限延長には使いません。遠方の強い候補は2回の検出で確認して復帰します。',
        '- 人物検出、保持者の推定、3D物理モデルは未実装です。ドリブルの反転等を一定速度へ強制しないよう再検出を要求しますが、プレーの種類は分類しません。',
        '- 検出モデル・imgsz960・confidence0.25・NMS0.45・FP32は既存B/Eと同一。学習はやり直していません。F固有の閾値は工学的初期値です。',
        '- フル動画実行後のパラメータ探索はしていません。初期120フレームの動作確認で全画面LKの負荷を確認し、局所処理へ変更しました。',
        '- B/E/Fは同じ設定ファイルで新規実行。過去の2026-10-27のA〜E測定と速度を混ぜません。',
        '- YOLO回数には別途3回のウォームアップを含めません。処理fpsには読み込み・追跡・描画・保存・テレメトリを含めます。カメラ入力から表示までの遅延ではありません。',
        '- 正解ラベルは推論後の描画・評価にだけ使用。Fの判断を元動画と保存検出候補から再実行し、全フレームの出力・適応スケジュール一致を検証しました。',
        '- 小型CPU端末での実測、15分安定性、別動画への汎化、構成要素別の寄与は未検証です。','',
        '## 目視した失敗例','',
        's/Fのframe 99（3.300秒）と551（18.367秒）を保存画像とログで確認しました。',
        'いずれもLK出力が正解ボールから外れ、選手の腕付近に残っています。frame 551では正解ボールはより下にあります。',
        '往復誤差が小さくても、ボール以外の特徴点を一貫して追うことはあります。人物情報を用いる遮蔽対策は今後の課題です。','',
        '## 保存場所','',
        '生ログ・動画はローカルの以下にあります。大容量動画と重みはGitへ含めません。','']
    for key,item in sorted(details.items()):
        lines.append(f"- {key}: `{item['video']}`")
    lines+=['','比較値: `comparison.csv` / `comparison.json`。検出実行有無別の精度: `detection_phase.csv`。',
        '各条件フォルダに設定・メトリクス・検証結果、`evidence.json` にソースハッシュとイベント集計を保存しています。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.runs,args.output)
