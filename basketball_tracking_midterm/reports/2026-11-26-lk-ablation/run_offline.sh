#!/bin/bash
# 2026-11-26 LK ablation on experiment10 (file benchmark, unpaced)
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe; O=results/nolk_20261126/offline
for a in "--imgsz 704 --frontend nolk" "--imgsz 704 --frontend nolk --max-interval 2" "--imgsz 704 --frontend nolk --max-interval 1 --max-yolo-hz 30" \
         "--roi-imgsz 320 --frontend nolk" "--roi-imgsz 320 --frontend nolk --max-interval 2"; do
  echo "### $a"; $PY scripts/run_fixed_lag.py --condition G $a --output $O --no-video 2>&1 | grep -E "Saved|Error|Traceback"
done
echo DONE
