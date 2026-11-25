#!/bin/bash
# 2026-11-25 ROI lock-on recovery variants (on G+ROI, cap 30/s unless noted)
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=results/alt_20261125
power(){ powershell -NoProfile -Command "\$b=Get-CimInstance Win32_Battery; '{0} status={1} charge={2}%' -f (Get-Date -Format o),\$b.BatteryStatus,\$b.EstimatedChargeRemaining; nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader" >> $OUT/power_log.txt; }
run(){ echo "### $*" >> $OUT/power_log.txt; power; $PY scripts/run_fixed_lag.py --condition G "$@" --output $OUT --no-video; power; }
R30="--roi-imgsz 320 --max-interval 1 --max-yolo-hz 30"
run $R30 --keep-challenger-on-roi
run $R30 --keep-challenger-on-roi --instant-switch-conf 0.5
run $R30 --roi-escalate
run $R30 --keep-challenger-on-roi --instant-switch-conf 0.5 --roi-escalate
run $R30 --keep-challenger-on-roi --instant-switch-conf 0.5 --roi-escalate --roi-full-interval-s 0.33
run --roi-imgsz 320 --roi-full-interval-s 0.2
run --instant-switch-conf 0.5
echo "### benchmark_stages" >> $OUT/power_log.txt; power
$PY scripts/benchmark_stages.py --yolov5 vendor/yolov5 --weights weights/finetuned/s/best.pt --device cuda:0 --imgsz 960 704 320 --video "../編集済み元データ/experiment10_編集.mp4" --frames 300 --yolo-frames 200 --out $OUT/gpu_benchmark_ac.json
power
echo DONE >> $OUT/power_log.txt
