"""ResNet-18 three-class blob classifier; session-safe manifest, validation-only model selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent/'.torch'))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from experiment_detectors import patch_transform
from prepare_experiment_data import validate_patches
from adaptive_basketball_tracker import file_hash


CLASSES = ['background', 'ball', 'player']


class Patches(Dataset):
    def __init__(self, rows, size, augment=False):
        self.rows, self.transform = rows, patch_transform(size, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            return self.transform(image.convert('RGB')), CLASSES.index(row['label'])


def assess(model, loader, device):
    confusion = np.zeros((3, 3), dtype=int)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(1).cpu().numpy()
            np.add.at(confusion, (labels.numpy(), predicted), 1)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, truth, predictions = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        per_class[name] = dict(support=truth, precision=tp/predictions if predictions else 0.,
                               recall=tp/truth if truth else None,
                               f1=2*tp/(truth+predictions) if truth else None)
    valid = [v['f1'] for v in per_class.values() if v['f1'] is not None]
    return dict(classes=CLASSES, confusion_gt_rows_pred_columns=confusion.tolist(), per_class=per_class,
                accuracy=float(confusion.trace()/confusion.sum()) if confusion.sum() else None,
                macro_f1=float(np.mean(valid)) if valid else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--image-size', type=int, default=107)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--evaluate', type=Path, help='Evaluate an existing checkpoint on test only; no training')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.image_size) < 1 or not np.isfinite(a.lr) or a.lr <= 0:
        p.error('Counts and learning rate must be positive')
    rows = validate_patches(a.manifest)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(a.evaluate, map_location='cpu', weights_only=True) if a.evaluate else None
    if checkpoint and checkpoint['classes'] != CLASSES:
        raise ValueError('Checkpoint class mismatch')
    size = checkpoint['image_size'] if checkpoint else a.image_size
    model = resnet18(weights=ResNet18_Weights.DEFAULT if a.pretrained and not checkpoint else None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    manifest_sha256=file_hash(a.manifest), classes=CLASSES, image_size=size,
                    torch=torch.__version__, device=device,
                    sessions={s: sorted({r['session_id'] for r in rows if r['split'] == s}) for s in ('train', 'val', 'test')})
    (a.output/'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def loader(split, training=False):
        return DataLoader(Patches([r for r in rows if r['split'] == split], size, training),
                          batch_size=a.batch_size, shuffle=training, num_workers=0)

    if checkpoint:
        report = assess(model, loader('test'), device)
        (a.output/'test_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    counts = [sum(r['split'] == 'train' and r['label'] == c for r in rows) for c in CLASSES]
    if not all(counts) or any(not any(r['split'] == 'val' and r['label'] == c for r in rows) for c in CLASSES):
        raise ValueError('Train and validation must contain every class')
    weights = torch.tensor([sum(counts)/(3*n) for n in counts], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best, history = -1., []
    training, validation = loader('train', True), loader('val')
    for epoch in range(a.epochs):
        model.train()
        total = 0.
        for images, labels in training:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(images)
        report = assess(model, validation, device)
        history.append(dict(epoch=epoch+1, train_loss=total/len(training.dataset), validation=report))
        if report['macro_f1'] > best:
            best = report['macro_f1']
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            classes=CLASSES, image_size=size, architecture='resnet18',
                            seed=a.seed, epoch=epoch+1, validation_macro_f1=best), a.output/'best.pt')
        print(f'Epoch {epoch+1}: validation macro-F1={report["macro_f1"]:.4f}')
    (a.output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('Training complete. Test data was not used. Freeze settings, then run --evaluate.')


if __name__ == '__main__':
    main()
