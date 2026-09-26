"""Evaluate single-ball CSVs against manually annotated, zero-based video frames."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        result = {}
        for row in csv.DictReader(f):
            index = int(row["frame_index"])
            if index < 0 or index in result:
                raise ValueError(f"Invalid or duplicate frame_index in {path}: {index}")
            result[index] = row
    if not result:
        raise ValueError(f"Empty CSV: {path}")
    return result


def point(row):
    if row is None or row.get("center_x", "") == "" or row.get("center_y", "") == "":
        return None
    values = float(row["center_x"]), float(row["center_y"])
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite or blank")
    return values


def flag(row, key):
    if row.get(key) not in ("0", "1"):
        raise ValueError(f"{key} must be 0 or 1")
    return row[key] == "1"


def evaluate(gt, predictions, threshold):
    groups = {key: [] for key in ("all", "normal", "occluded", "near_ring")}
    correct, successes = {}, {}
    for index, truth in sorted(gt.items()):
        occluded, near = flag(truth, "occluded"), flag(truth, "near_ring")
        visible = flag(truth, "visible")
        target = point(truth)
        if visible and target is None:
            raise ValueError(f"Visible frame {index} needs coordinates")
        prediction = predictions.get(index)
        output = point(prediction)
        observed = prediction is not None and prediction.get("observed") == "1"
        error = math.dist(target, output) if target is not None and output is not None else None
        success = error is not None and error <= threshold
        correct[index] = bool(success and observed)
        successes[index] = success if target is not None else None
        sample = dict(annotated=target is not None, output=output is not None,
                      error=error, success=success, observed_success=success and observed)
        groups["all"].append(sample)
        if occluded:
            groups["occluded"].append(sample)
        if near:
            groups["near_ring"].append(sample)
        if not occluded and not near:
            groups["normal"].append(sample)
    result = {}
    for name, samples in groups.items():
        labeled = [s for s in samples if s["annotated"]]
        errors = [s["error"] for s in labeled if s["error"] is not None]
        count = len(labeled)
        result[name] = dict(frames=len(samples), annotated_frames=count,
                            output_coverage=sum(s["output"] for s in samples)/len(samples) if samples else None,
                            success_rate=sum(s["success"] for s in labeled)/count if count else None,
                            observed_success_rate=sum(s["observed_success"] for s in labeled)/count if count else None,
                            mean_error_px=statistics.mean(errors) if errors else None,
                            rmse_px=math.sqrt(statistics.mean(e*e for e in errors)) if errors else None)
    # Count only contiguous annotated runs; unlabeled frames must not bridge a failure event.
    failures, longest, run, previous, failed_before = 0, 0, 0, None, False
    for index, success in successes.items():
        contiguous = previous is not None and index == previous+1
        if success is None:
            run, failed_before = 0, False
        elif success:
            run = run+1 if contiguous else 1
            longest = max(longest, run)
            failed_before = False
        else:
            failures += int(not contiguous or not failed_before)
            run, failed_before = 0, True
        previous = index
    # Recovery starts on first visible frame after a labeled occluded frame.
    # Missing annotation frames, new occlusion, or end-of-clip censor the event.
    recovery, censored = [], 0
    indices = sorted(gt)
    for index in indices:
        if index-1 not in gt or not flag(gt[index-1], "occluded") or flag(gt[index], "occluded") or not flag(gt[index], "visible"):
            continue
        cursor = index
        while cursor in gt and flag(gt[cursor], "visible") and not flag(gt[cursor], "occluded"):
            if correct[cursor]:
                recovery.append(cursor-index)
                break
            cursor += 1
        else:
            censored += 1
    rows = list(predictions.values())
    times = [float(row["processing_ms"]) for row in rows]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("processing_ms must be finite and nonnegative")
    result["runtime"] = dict(processed_frames=len(rows), missing_prediction_rows=len(set(gt)-set(predictions)),
                             detector_calls=sum(int(r["detector_called"]) for r in rows),
                             processing_fps=1000*len(times)/sum(times) if sum(times) else None,
                             switches=sum(int(r["switched"]) for r in rows),
                             failure_events=failures, longest_success_run_frames=longest,
                             recovery_frames=recovery, recovery_censored=censored,
                             mean_recovery_frames=statistics.mean(recovery) if recovery else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold-px", type=float, default=20.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.threshold_px) or args.threshold_px <= 0:
        parser.error("--threshold-px must be positive and finite")
    gt = load_rows(args.ground_truth)
    report = dict(threshold_px=args.threshold_px, ground_truth=str(args.ground_truth.resolve()), methods={})
    for path in args.predictions:
        report["methods"][str(path)] = evaluate(gt, load_rows(path), args.threshold_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
