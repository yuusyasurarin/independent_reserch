"""Extract reproducible, temporally separated CVAT image splits (no labels)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "eval", "test")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan_frames(count, fps, sample_fps, ratios, guard):
    """Half-open chronological intervals; guard seconds on both sides of cuts."""
    duration = count / fps
    cuts = [0, duration * ratios[0], duration * sum(ratios[:2]), duration]
    result, intervals = {}, []
    for i, split in enumerate(SPLITS):
        start = cuts[i] + (guard if i else 0)
        end = cuts[i + 1] - (guard if i < 2 else 0)
        if end - start < 1 / sample_fps:
            raise ValueError(f"Interval too short for {split}; reduce guard or adjust ratios")
        intervals.append(dict(split=split, start_s=start, end_s=end))
        for n in range(math.ceil((end - start) * sample_fps)):
            t = start + (n + 0.5) / sample_fps
            frame = round(t * fps)
            if frame < count and start <= frame / fps < end:
                result[frame] = split
    return result, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "編集済み元データ")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/cvat_experiment11_12")
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--guard-seconds", type=float, default=2)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if not (math.isfinite(args.sample_fps) and args.sample_fps > 0
            and math.isfinite(args.guard_seconds) and args.guard_seconds >= 0
            and all(math.isfinite(r) and r > 0 for r in args.ratios)
            and math.isclose(sum(args.ratios), 1) and 1 <= args.jpeg_quality <= 100):
        parser.error("Invalid sampling, guard, ratios or JPEG quality")
    sources = []
    for name in ("experiment11", "experiment12"):
        matches = sorted(args.source_dir.glob(name + "_*.mp4"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name}_*.mp4; found {len(matches)}")
        path = matches[0]
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open {path}")
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count <= 0 or args.sample_fps > fps:
                raise ValueError(f"Invalid metadata/sampling for {path}")
            frames, intervals = plan_frames(count, fps, args.sample_fps, args.ratios, args.guard_seconds)
            sources.append(dict(name=name, path=str(path.resolve()), sha256=sha256(path),
                                fps=fps, frame_count=count, duration_s=count/fps,
                                width=int(cap.get(3)), height=int(cap.get(4)),
                                intervals=intervals, frames=frames))
        finally:
            cap.release()
    if sources[0]["sha256"] == sources[1]["sha256"]:
        raise ValueError("Duplicate source videos")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an annotated dataset.
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    (output / "cvat_upload").mkdir()
    rows, duplicates, seen = [], [], set()
    for source in sources:
        cap = cv2.VideoCapture(source["path"])
        targets = source["frames"]
        try:
            for index in range(max(targets) + 1):
                if not cap.grab():
                    raise RuntimeError(f"Decode stopped early: {source['name']} frame {index}")
                if index not in targets:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot decode frame {index}")
                raw_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                if raw_hash in seen:
                    duplicates.append(dict(source=source["name"], frame_index=index, split=targets[index]))
                    continue
                seen.add(raw_hash)
                split = targets[index]
                relative = Path("images") / split / f"{source['name']}_f{index:07d}.jpg"
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                encoded.tofile(str(output / relative))  # Windows Unicode paths.
                rows.append(dict(image=relative.as_posix(), split=split, source=source["name"],
                                 frame_index=index, timestamp_s=round(index/source["fps"], 6),
                                 width=frame.shape[1], height=frame.shape[0],
                                 pixel_sha256=raw_hash, image_sha256=sha256(output / relative),
                                 annotation_status="unlabeled"))
        finally:
            cap.release()
        print(f"Extracted {source['name']}", flush=True)
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    if not all(counts.values()):
        raise RuntimeError("Empty split after duplicate removal")
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        with zipfile.ZipFile(output / "cvat_upload" / f"{split}.zip", "w", zipfile.ZIP_STORED) as archive:
            for row in selected:
                archive.write(output / row["image"], Path(row["image"]).name)
        # Representative images for manual QA; these previews are not training images.
        indices = sorted(random.Random(42).sample(range(len(selected)), min(12, len(selected))))
        canvas = np.full((3 * 210, 4 * 320, 3), 245, dtype=np.uint8)
        for slot, index in enumerate(indices):
            row = selected[index]
            frame = cv2.imdecode(np.fromfile(str(output / row["image"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            thumb = cv2.resize(frame, (320, 180))
            y, x = (slot // 4) * 210, (slot % 4) * 320
            canvas[y:y+180, x:x+320] = thumb
            cv2.putText(canvas, Path(row["image"]).stem, (x+5, y+198), cv2.FONT_HERSHEY_SIMPLEX, .43, (0,0,0), 1)
        cv2.imencode(".jpg", canvas)[1].tofile(str(output / f"preview_{split}.jpg"))
    metadata = dict(status="complete_unlabeled", strategy="chronological_per_video_with_guard",
                    sample_fps=args.sample_fps, ratios=args.ratios, guard_seconds=args.guard_seconds,
                    jpeg_quality=args.jpeg_quality, counts=counts,
                    sources=[{k: v for k, v in s.items() if k != "frames"} for s in sources],
                    skipped_exact_duplicates=duplicates, opencv_version=cv2.__version__)
    (output / "preparation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data.yaml.template").write_text(
        '# Set class names to the exact CVAT export order, then rename to data.yaml.\n'
        + 'path: ' + json.dumps(output.as_posix(), ensure_ascii=False) + '\n'
        + 'train: images/train\nval: images/eval\ntest: images/test\n'
        + '# Example only: names: {0: ball, 1: player, 2: ring}\nnames: {}\n', encoding="utf-8")
    print(json.dumps(counts), flush=True)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
