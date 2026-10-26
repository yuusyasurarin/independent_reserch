"""Detection PR and tracking metrics use separate predictions and denominators."""
from __future__ import annotations

import math
from .common import center, iou, stats


def match_detections(predictions,truths,threshold=.5):
    """Confidence-ordered, one-to-one matching. Wrong object => FP plus unmatched GT FN."""
    matched=set(); flags=[]
    for pred in sorted(predictions,key=lambda p:p['confidence'],reverse=True):
        available=[(iou(pred['bbox'],box),i) for i,box in enumerate(truths) if i not in matched]
        best=max(available,default=(0,None))
        tp=best[1] is not None and best[0]>=threshold
        if tp: matched.add(best[1])
        flags.append((pred['confidence'],int(tp)))
    return flags,len(truths)-len(matched)


def detection_metrics(gt,logs,confidence=.25):
    # D/E raw detector accuracy is conditional on scheduled frames, NOT full-timeline tracking.
    flags=[]; total_gt=0; eligible_count=0; operating_tp=operating_fp=operating_fn=0
    for log in logs:
        row=gt[log['frame_id']]
        if not log['yolo_executed'] or row['status'] not in ('visible','absent'):
            continue
        eligible_count+=1
        boxes=[row['bbox']] if row['status']=='visible' else []
        total_gt+=len(boxes)
        matched,_=match_detections(log['candidates'],boxes)
        flags.extend(matched)
        operating,fn=match_detections([x for x in log['candidates'] if x['confidence']>=confidence],boxes)
        operating_tp+=sum(tp for _,tp in operating)
        operating_fp+=sum(1-tp for _,tp in operating)
        operating_fn+=fn
    flags.sort(key=lambda x:x[0],reverse=True)
    curve=[];tp=fp=0
    for index,(score,correct) in enumerate(flags):
        tp+=correct;fp+=1-correct
        # Group ties: PR coordinates must not depend on arbitrary tied-score ordering across frames.
        if index+1==len(flags) or flags[index+1][0]!=score:
            curve.append(dict(confidence=score,precision=tp/(tp+fp),recall=tp/total_gt if total_gt else 0,tp=tp,fp=fp))
    ap=None
    if total_gt:
        ap=sum(max((r['precision'] for r in curve if r['recall']>=level/100),default=0.) for level in range(101))/101
    precision=operating_tp/(operating_tp+operating_fp) if operating_tp+operating_fp else (0. if total_gt else None)
    recall=operating_tp/total_gt if total_gt else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0. if total_gt else None)
    return dict(precision=precision,recall=recall,f1=f1,ap50=ap,tp=operating_tp,fp=operating_fp,fn=operating_fn,
                confidence=confidence,iou_threshold=.5,eligible_detection_frames=eligible_count,gt_boxes=total_gt,
                yolo_executed_frames=sum(r['yolo_executed'] for r in logs),
                scope='Scheduled YOLO frames only; visible/confirmed-absent GT only. Unknown positions excluded.',
                ap_definition='101-point interpolated precision envelope, one ball class, IoU 0.5; candidates above configured floor after NMS.'),curve


def tracking_metrics(gt,predictions,fps):
    rows=[];overlaps=[];errors=[];runs=[];missing_runs=[];run=missing_run=0;previous=None
    for pred in predictions:
        truth=gt[pred['frame_id']]
        eligible=truth['status']=='visible'
        consecutive=(previous is not None and pred['frame_id']==previous['frame_id']+1
                     and truth['segment_id']==previous['segment_id'])
        if not consecutive:
            if run:runs.append(run);run=0
            if missing_run:missing_runs.append(missing_run);missing_run=0
        overlap=iou(truth['bbox'],pred['bbox']) if eligible else None
        error=math.dist(center(truth['bbox']),center(pred['bbox'])) if eligible and pred['bbox'] else None
        if eligible:
            overlaps.append(overlap)
            if error is not None:errors.append(error)
        if eligible and overlap<.5:run+=1
        elif run:runs.append(run);run=0
        if eligible and pred['bbox'] is None:missing_run+=1
        elif missing_run:missing_runs.append(missing_run);missing_run=0
        rows.append(dict(frame_id=pred['frame_id'],timestamp_s=pred['timestamp_s'],gt_status=truth['status'],
                         eligible=int(eligible),output=int(pred['bbox'] is not None),iou=overlap,cle_px=error,
                         success_03=int(overlap>=.3) if eligible else None,success_05=int(overlap>=.5) if eligible else None,
                         state=pred['state'],source=pred['source'],track_id=pred['track_id']))
        previous=truth
    if run:runs.append(run)
    if missing_run:missing_runs.append(missing_run)
    n=len(overlaps)
    summary=dict(evaluated_position_frames=n,total_frames=len(predictions),
        tsr_iou_03=sum(v>=.3 for v in overlaps)/n if n else None,
        tsr_iou_05=sum(v>=.5 for v in overlaps)/n if n else None,
        mean_iou=sum(overlaps)/n if n else None,
        output_rate_visible=sum(r['output'] for r in rows if r['eligible'])/n if n else None,
        output_rate_all=sum(r['output'] for r in rows)/len(rows) if rows else None,
        cle_px=stats(errors),
        missing_outputs=sum(not r['output'] for r in rows if r['eligible']),
        longest_failure_at_05_s=max(runs,default=0)/fps if n else None,
        longest_no_output_s=max(missing_runs,default=0)/fps if n else None,
        failure_runs_ge3=sum(v>=3 for v in runs),
        unknown_position_frames=sum(r['gt_status']=='unknown_position' for r in rows),
        unannotated_frames=sum(r['gt_status']=='unannotated' for r in rows),
        confirmed_absent_frames=sum(r['gt_status']=='absent' for r in rows),
        outputs_unknown_position=sum(r['output'] for r in rows if r['gt_status']=='unknown_position'),
        outputs_confirmed_absent=sum(r['output'] for r in rows if r['gt_status']=='absent'),
        limitations=[
            'CLE uses every visible-GT frame with output, including incorrect outputs; missing outputs excluded and reported separately.',
            'TSR denominator is all visible-GT frames; missing output is failure and IoU=0.',
            'Runs are observed consecutive visible-GT frames only, broken at unknown/unannotated/segment boundaries; not true occlusion recovery time.',
            'Without annotated shot boundaries, runs may span scene edits. Sparse labels cannot establish full continuous lost duration.',
            'Unknown-position outputs cannot be judged as FP, unlike outputs on confirmed-absent frames.',
            'Failure images show low IoU; semantic wrong-object identity requires visual review.'
        ])
    return summary,rows
