"""Decode all images, make label previews, and package files for Colab (requires Pillow)."""
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile

from PIL import Image, ImageDraw
from prepare_dataset import verify, parse_label

ROOT = Path(__file__).resolve().parent


def main():
    dataset = ROOT / "dataset"
    rows = verify(dataset)
    sizes, box_widths, box_heights = set(), [], []
    for row in rows:
        with Image.open(dataset / row["image"]) as im:
            im.load()
            sizes.add(im.size)
            if im.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Wrong dimensions: {row['image']}")
            for x, y, w, h in parse_label((dataset / row["label"]).read_text(), row["label"]):
                box_widths.append(w * im.width)
                box_heights.append(h * im.height)
    qa = dict(decoded_images=len(rows), dimensions=sorted(sizes),
              box_width_pixels=dict(min=min(box_widths), max=max(box_widths)),
              box_height_pixels=dict(min=min(box_heights), max=max(box_heights)))
    (ROOT / "image_quality_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    # Only train/val previews: keep final test outside the tuning workflow.
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        positives = [r for r in subset if int(r["boxes"]) > 0]
        backgrounds = [r for r in subset if int(r["boxes"]) == 0]
        rng = random.Random(42)
        chosen = rng.sample(positives, min(9, len(positives))) + rng.sample(backgrounds, min(3, len(backgrounds)))
        canvas = Image.new("RGB", (3 * 480, 4 * 298), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(chosen):
            with Image.open(dataset / row["image"]) as src:
                thumb = src.convert("RGB")
            painter = ImageDraw.Draw(thumb)
            for x, y, w, h in parse_label((dataset / row["label"]).read_text(), row["label"]):
                painter.rectangle(((x-w/2)*thumb.width, (y-h/2)*thumb.height,
                                   (x+w/2)*thumb.width, (y+h/2)*thumb.height),
                                  outline=(0, 255, 80), width=4)
            thumb = thumb.resize((480, 270))
            x0, y0 = i % 3 * 480, i // 3 * 298
            canvas.paste(thumb, (x0, y0))
            draw.text((x0+4, y0+275), f"{Path(row['image']).stem} | ball={row['boxes']}", fill="black")
        canvas.save(ROOT / f"preview_{split}.jpg", quality=92)
    archive = ROOT / "ball_yolo_colab.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and p.suffix not in (".zip", ".pyc", ".cache")
                   and "__pycache__" not in p.parts
                   and p.name not in ("bundle_sha256.txt", "data.colab.yaml"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            z.write(p, "ball_yolo_colab/" + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        if len(z.namelist()) != len(files):
            raise ValueError("ZIP member count mismatch")
    with archive.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    (ROOT / "bundle_sha256.txt").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          files=len(files), sha256=checksum, image_qa=qa), indent=2))


if __name__ == "__main__":
    main()
