"""Build a verified YOLO detection dataset from this project's CVAT exports.

Uses only the Python standard library. Original ZIPs are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

SPLITS = {"train": "train", "eval": "val", "test": "test"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entries(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate archive members: {archive.filename}")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError(f"Unsafe archive member: {name}")
    return names


def parse_label(text, name):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Expected class 0 detection box: {name}: {line}")
        x, y, w, h = map(float, parts[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Non-finite box: {name}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {name}")
        # Six-decimal CVAT exports can differ at the boundary by rounding.
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"Box outside image: {name}")
        boxes.append((x, y, w, h))
    if len(boxes) != len(set(boxes)):
        raise ValueError(f"Duplicate boxes: {name}")
    return boxes


def build(source, output, missing_policy):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    prepared = source / "cvat_experiment11_12"
    manifest = {}
    with (prepared / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["split"], Path(row["image"]).name)
            if key in manifest:
                raise ValueError(f"Duplicate manifest row: {key}")
            manifest[key] = row
    rows, payloads, missing, seen_hashes, seen_stems = [], [], [], {}, set()
    report = {
        "status": "verified",
        "class_names": {"0": "ball"},
        "missing_label_policy": missing_policy,
        "background_confirmation": (
            "User confirmed all 161 missing-label images were reviewed and contain no ball (2026-10-09)."
            if missing_policy == "background" else None
        ),
        "split_mapping": SPLITS,
        "splits": {}, "source_archives": {},
        "split_strategy": "Original chronological split, no random resplit; see provenance/preparation.json",
    }
    for original, target in SPLITS.items():
        annotations = source / f"ball_only_アノテーション_{original}.zip"
        images = prepared / "cvat_upload" / f"{original}.zip"
        report["source_archives"][original] = {
            "annotations": annotations.name, "annotations_sha256": digest(annotations.read_bytes()),
            "images": f"cvat_upload/{original}.zip", "images_sha256": digest(images.read_bytes()),
        }
        with zipfile.ZipFile(annotations) as az, zipfile.ZipFile(images) as iz:
            anames, inames = entries(az), entries(iz)
            # These CVAT Ultralytics exports contain exactly this one class.
            yaml_text = az.read("data.yaml").decode("utf-8-sig").replace("\r\n", "\n")
            if not re.search(r"(?m)^names:\s*\n\s+0:\s*ball\s*\n", yaml_text):
                raise ValueError("Expected CVAT names: {0: ball}")
            if re.search(r"(?m)^\s+[1-9][0-9]*:", yaml_text):
                raise ValueError("Unexpected additional classes")
            listed = [PurePosixPath(n.strip()).name for n in az.read("train.txt").decode("utf-8-sig").splitlines() if n.strip()]
            image_names = [n for n in inames if not n.endswith("/")]
            if any(PurePosixPath(n).suffix.lower() != ".jpg" for n in image_names):
                raise ValueError("Unexpected non-JPEG image ZIP member")
            by_name = {PurePosixPath(n).name: n for n in image_names}
            expected = {name for split, name in manifest if split == original}
            if len(by_name) != len(image_names) or len(listed) != len(set(listed)):
                raise ValueError("Duplicate image name")
            if set(listed) != set(by_name) or set(by_name) != expected:
                raise ValueError(f"Image list / ZIP / original manifest mismatch: {original}")
            label_map = {}
            for name in anames:
                if name.startswith("labels/") and name.endswith(".txt"):
                    stem = PurePosixPath(name).stem
                    if stem in label_map:
                        raise ValueError(f"Duplicate label stem: {stem}")
                    label_map[stem] = name
            if set(label_map) - {Path(n).stem for n in by_name}:
                raise ValueError(f"Labels without images: {original}")
            counts = dict(source_images=len(by_name), images=0, positive_images=0,
                          background_images=0, boxes=0, missing_label_files=0, excluded=0)
            for name in sorted(by_name):
                stem = Path(name).stem
                if stem in seen_stems:
                    raise ValueError(f"Cross-split duplicate name: {name}")
                seen_stems.add(stem)
                data = iz.read(by_name[name])
                hash_value = digest(data)
                if hash_value in seen_hashes:
                    raise ValueError(f"Duplicate image bytes: {name}, {seen_hashes[hash_value]}")
                seen_hashes[hash_value] = name
                source_row = manifest[(original, name)]
                if hash_value != source_row["image_sha256"]:
                    raise ValueError(f"Original image hash mismatch: {name}")
                is_missing = stem not in label_map
                if is_missing:
                    counts["missing_label_files"] += 1
                    missing.append(dict(split=target, image=name, action=missing_policy))
                    if missing_policy == "error":
                        raise ValueError(f"Missing label: {name}. Explicit policy required.")
                    if missing_policy == "exclude":
                        counts["excluded"] += 1
                        continue
                    label = ""
                else:
                    label = az.read(label_map[stem]).decode("utf-8-sig")
                boxes = parse_label(label, name)
                label_data = ("\n".join("0 " + " ".join(f"{v:.6f}" for v in box) for box in boxes)
                              + ("\n" if boxes else "")).encode()
                image_rel, label_rel = f"images/{target}/{name}", f"labels/{target}/{stem}.txt"
                payloads.extend([(image_rel, data), (label_rel, label_data)])
                rows.append(dict(split=target, image=image_rel, label=label_rel,
                                 source=source_row["source"], frame_index=source_row["frame_index"],
                                 timestamp_s=source_row["timestamp_s"], width=source_row["width"],
                                 height=source_row["height"], boxes=len(boxes),
                                 annotation_status="confirmed_background" if is_missing else "exported_label",
                                 image_sha256=hash_value, label_sha256=digest(label_data)))
                counts["images"] += 1
                counts["boxes"] += len(boxes)
                counts["positive_images" if boxes else "background_images"] += 1
            if not counts["positive_images"]:
                raise ValueError(f"No positive images in {target}")
            report["splits"][target] = counts

    output.mkdir(parents=True)
    for relative, data in payloads:
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (output / "data.yaml").write_text(
        "# Colab rewrites path to its absolute extraction directory.\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n",
        encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "missing_labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "action"])
        writer.writeheader()
        writer.writerows(missing)
    (output / "provenance").mkdir()
    for name in ("preparation.json", "verification.json", "manifest.csv"):
        shutil.copy2(prepared / name, output / "provenance" / name)
    report["total_images"] = len(rows)
    report["total_boxes"] = sum(r["boxes"] for r in rows)
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify(dataset):
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_files, seen_hashes = set(), set()
    for row in rows:
        for key in ("image", "label"):
            rel = PurePosixPath(row[key])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe manifest path")
            p = dataset / row[key]
            expected_files.add(p.resolve())
            if digest(p.read_bytes()) != row[key + "_sha256"]:
                raise ValueError(f"Hash mismatch: {p}")
        boxes = parse_label((dataset / row["label"]).read_text(encoding="utf-8"), row["label"])
        if len(boxes) != int(row["boxes"]):
            raise ValueError("Box count mismatch")
        if row["image_sha256"] in seen_hashes:
            raise ValueError("Duplicate image")
        seen_hashes.add(row["image_sha256"])
    actual = {p.resolve() for kind in ("images", "labels") for p in (dataset / kind).rglob("*")
              if p.is_file() and p.suffix != ".cache"}
    if expected_files != actual or len(expected_files) != 2 * len(rows):
        raise ValueError("Unexpected, missing, or duplicate dataset files")
    print(f"Verified {len(rows)} image/label pairs and SHA-256 hashes.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent / "datasets")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--missing-labels", choices=("error", "exclude", "background"), default="error",
                        help="Use background only after confirming every missing label is a reviewed negative.")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        report = build(args.source, args.output, args.missing_labels)
        verify(args.output)
        print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
