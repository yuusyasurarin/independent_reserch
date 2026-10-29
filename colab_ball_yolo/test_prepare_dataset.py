"""Meaningful failure-path tests; no model download or training."""
import csv
import tempfile
import unittest
from pathlib import Path
import zipfile

from prepare_dataset import digest, entries, parse_label, verify


class PreparationTests(unittest.TestCase):
    def test_background_and_multiple_boxes(self):
        self.assertEqual(parse_label("", "empty"), [])
        self.assertEqual(len(parse_label("0 0.5 0.5 0.1 0.1\n0 0.1 0.1 0.1 0.1", "boxes")), 2)

    def test_bad_boxes_fail(self):
        for text in ("32 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1",
                     "0 0.5 0.5 -0.1 0.1", "0 0.99 0.5 0.2 0.1",
                     "0 0.5 0.5 0.1 0.1 0.4",
                     "0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_label(text, "invalid")

    def test_boundary_rounding_allowed(self):
        self.assertEqual(len(parse_label("0 0.05 0.5 0.100001 0.1", "rounding")), 1)

    def test_archive_traversal_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "bad.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("../escape.txt", "x")
            with zipfile.ZipFile(p) as z, self.assertRaises(ValueError):
                entries(z)

    def test_tampered_and_extra_files_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "images/train/a.jpg"
            label = root / "labels/train/a.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"hash-fixture")
            label.write_bytes(b"")
            row = dict(image="images/train/a.jpg", label="labels/train/a.txt", boxes=0,
                       image_sha256=digest(image.read_bytes()), label_sha256=digest(b""))
            with (root / "manifest.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            verify(root)
            label.write_text("0 0.5 0.5 0.1 0.1")
            with self.assertRaises(ValueError):
                verify(root)
            label.write_bytes(b"")
            (label.parent / "unexpected.txt").write_text("")
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == "__main__":
    unittest.main()
