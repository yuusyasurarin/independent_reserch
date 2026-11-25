#!/bin/bash
# 2026-11-25 five-run comparison on experiment10 (README section 6)
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=results/alt_20261125
power(){ powershell -NoProfile -Command "\$b=Get-CimInstance Win32_Battery; '{0} status={1} charge={2}%' -f (Get-Date -Format o),\$b.BatteryStatus,\$b.EstimatedChargeRemaining; nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader" >> $OUT/power_log.txt; }
run(){ echo "### $*" >> $OUT/power_log.txt; power; $PY scripts/run_fixed_lag.py --condition G "$@" --output $OUT --no-video; power; }
run
run --frontend template
run --roi-imgsz 320
run --roi-imgsz 320 --max-interval 1 --max-yolo-hz 30
run --frontend template --roi-imgsz 320 --max-interval 1 --max-yolo-hz 30
echo DONE >> $OUT/power_log.txt
