#!/bin/bash
# 2026-11-26 gym-day models 1/2 on experiment11/12 (paced at 30 fps) for shot-judgement evaluation
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe; O=results/shot_models_20261126
for e in 11 12; do
  V="../編集済み元データ/experiment${e}_編集.mp4"
  echo "### exp$e model1"; $PY scripts/realtime_tracking.py --video "$V" --no-display --record $O/exp${e}_model1 --imgsz 704 2>&1 | grep -E "Running|tracking:"
  echo "### exp$e model2"; $PY scripts/realtime_tracking.py --video "$V" --no-display --record $O/exp${e}_model2 --roi-imgsz 320 2>&1 | grep -E "Running|tracking:"
done
echo DONE
