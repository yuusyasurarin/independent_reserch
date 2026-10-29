"""Colab training/evaluation entry point. Training is explicit, never run during preparation."""
from __future__ import annotations
import argparse
import csv
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

from prepare_dataset import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val", "test"))
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--project", type=Path, default=Path("/content/drive/MyDrive/ball_yolo/runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--final-test", action="store_true",
                        help="Acknowledge model/settings were fixed using val before accessing test.")
    args = parser.parse_args()
    if args.mode == "test" and not args.final_test:
        parser.error("Use --final-test only after selecting model/settings on val.")
    if args.mode != "train" and args.resume:
        parser.error("--resume is only valid for training.")
    if args.mode != "train" and args.model == "yolo26n.pt":
        parser.error("Evaluation requires --model pointing to a trained single-class best.pt.")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs/batch must be positive; imgsz must be a positive multiple of 32.")

    import torch
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. In Colab select Runtime > Change runtime type > GPU.")
    dataset = args.dataset.resolve()
    verify(dataset)
    # Absolute paths avoid Ultralytics' global datasets_dir and Colab cwd ambiguity.
    config = dict(path=str(dataset), train="images/train", val="images/val",
                  test="images/test", names={0: "ball"})
    data_yaml = dataset / "data.colab.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = args.name or f"ball_{args.mode}_{stamp}"
    if (project / name).exists() and not args.resume:
        raise FileExistsError(f"Use a new run name: {project / name}")
    if args.mode == "train":
        model = YOLO(str(args.resume) if args.resume else args.model)
        if args.resume:
            model.train(resume=True, data=str(data_yaml), device=args.device)
        else:
            model.train(
                data=str(data_yaml), project=str(project), name=name, exist_ok=False,
                epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=2, seed=args.seed, deterministic=True,
                optimizer="AdamW", lr0=args.lr0, lrf=0.01, weight_decay=0.0005,
                cos_lr=True, warmup_epochs=3, mosaic=0.5, close_mosaic=10,
                scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0,
                mixup=0.0, degrees=0.0, cache=False, amp=True, save=True,
                save_period=10, plots=True,
            )
        run_dir = Path(model.trainer.save_dir)
        result = {"best": str(run_dir / "weights/best.pt"), "last": str(run_dir / "weights/last.pt")}
    else:
        model = YOLO(args.model)
        if model.names != {0: "ball"}:
            raise ValueError(f"Expected single-class ball model, got {model.names}")
        metrics = model.val(data=str(data_yaml), split=args.mode, imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=2,
                            project=str(project), name=name, plots=True,
                            conf=0.001, iou=0.7)
        run_dir = Path(metrics.save_dir)
        result = {k: float(v) for k, v in metrics.results_dict.items()}
        result["speed_ms_per_image"] = metrics.speed
    metadata = dict(mode=args.mode, arguments={k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, python=platform.python_version(),
                    torch=torch.__version__, ultralytics=ultralytics.__version__,
                    cuda=torch.version.cuda, result=result)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (run_dir / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
