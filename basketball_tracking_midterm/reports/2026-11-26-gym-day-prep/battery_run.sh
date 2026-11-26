#!/bin/bash
# 2026-11-26 battery-only speed/accuracy of model A and B (experiment10 paced at 30 fps)
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe; O=results/battery_20261126; V="../編集済み元データ/experiment10_編集.mp4"
B="--roi-imgsz 320 --max-interval 1 --max-yolo-hz 30"
run(){ name=$1; shift; echo "### $name $*"; $PY scripts/realtime_tracking.py --video "$V" --no-display --record $O/$name "$@" 2>&1 | grep -E "Running|tracking:|processing_ms|Saved"; }
run A_960
run B_960 $B
run A_704 --imgsz 704
run B_704 --imgsz 704 $B
echo DONE
