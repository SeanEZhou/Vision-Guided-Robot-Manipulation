"""Fine-tune Mask R-CNN on the simulator segmentation dataset (scripts/make_seg_dataset.py).

Validation after each epoch on held-out scenes: for every labeled object, the best-overlapping
prediction's mask IoU, and whether it is detected (IoU >= 0.5) with the correct class.

Usage: python scripts/train_segmenter.py [--data data/seg] [--epochs 10] [--out models/maskrcnn_seg.pt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from segmenter import CLASSES, MASK_THRESHOLD, SCORE_THRESHOLD, build_model


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=False):
        self.files = sorted(Path(root).glob("*_rgb.png"))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rgb = cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(self.files[i]).replace("_rgb.png", "_label.png"), cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if self.augment:  # photometric only: the camera geometry is fixed in deployment
            img = (img * np.random.uniform(0.7, 1.3) + np.random.uniform(-0.08, 0.08)).clamp(0, 1)
        masks, labels, boxes = [], [], []
        for c in range(1, len(CLASSES)):
            m = label == c
            if m.sum() == 0:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)
            labels.append(c)
        target = dict(
            boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.tensor(labels, dtype=torch.int64),
            masks=torch.from_numpy(np.array(masks, dtype=np.uint8)).reshape(-1, *label.shape),
        )
        return img, target, label


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, correct, total = [], 0, 0
    per_class = {c: [0, 0] for c in CLASSES[1:]}
    for imgs, _, labels in loader:
        outs = model([i.to(device) for i in imgs])
        for out, label in zip(outs, labels):
            keep = out["scores"] >= SCORE_THRESHOLD
            pm = (out["masks"][keep, 0] > MASK_THRESHOLD).cpu().numpy()
            pl = out["labels"][keep].cpu().numpy()
            for c in range(1, len(CLASSES)):
                tm = label == c
                if tm.sum() == 0:
                    continue
                total += 1
                per_class[CLASSES[c]][1] += 1
                if len(pm) == 0:
                    ious.append(0.0)
                    continue
                inter = (pm & tm).reshape(len(pm), -1).sum(1)
                union = (pm | tm).reshape(len(pm), -1).sum(1)
                j = int(np.argmax(inter / np.maximum(union, 1)))
                iou = float(inter[j] / max(union[j], 1))
                ious.append(iou)
                if iou >= 0.5 and pl[j] == c:
                    correct += 1
                    per_class[CLASSES[c]][0] += 1
    return dict(objects=total, detected_correct_class=correct, rate=correct / max(total, 1),
                mask_iou_median=float(np.median(ious)), mask_iou_mean=float(np.mean(ious)),
                per_class={k: f"{a}/{b}" for k, (a, b) in per_class.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/seg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out", default="models/maskrcnn_seg.pt")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = SegDataset(Path(args.data) / "train", augment=True)
    val = SegDataset(Path(args.data) / "val")
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4,
                                               collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=4, num_workers=4, collate_fn=collate)
    print(f"train {len(train)} images, val {len(val)} images, device {device}", flush=True)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    warmup = min(500, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for imgs, targets, _ in train_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        metrics = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch + 1, train_loss=total / len(train_loader), **metrics))
        print(f"epoch {epoch + 1}: loss {total / len(train_loader):.3f}  val detected+class "
              f"{metrics['detected_correct_class']}/{metrics['objects']}  mask IoU median {metrics['mask_iou_median']:.3f}  "
              f"per class {metrics['per_class']}  ({time.time() - t0:.0f} s)", flush=True)
        if metrics["rate"] + metrics["mask_iou_mean"] > best:
            best = metrics["rate"] + metrics["mask_iou_mean"]
            torch.save(model.state_dict(), args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(history, indent=2))
    print(f"saved best model to {args.out}")


if __name__ == "__main__":
    main()
