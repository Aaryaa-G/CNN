"""
GPU-ready training script for the event-based maritime ship detector.

Usage (after running build_dataset.py once):
    python train.py --data-dir built --epochs 100 --batch-size 16 --device cuda --amp

Runs fine on CPU too (just slower) -- --device auto (default) picks cuda if
available. Saves the best (by val mean-best-IoU) and the latest checkpoint
to --out-dir every epoch.
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import EventManifestDataset, SequenceGroupedBatchSampler, collate_same_res, STRIDE
from model import CenterNetLite
from losses import centernet_loss
from centernet_utils import decode_boxes
from eval_utils import match_frame


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_validation(model, loader, device, score_thresh, iou_thresh, max_batches=None):
    model.eval()
    tp = fp = fn = 0
    iou_sum = 0.0
    n_gt = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch["image"].to(device)
            heatmap, wh, offset = model(images)
            for i in range(images.shape[0]):
                preds = decode_boxes(heatmap[i], wh[i], offset[i], STRIDE, score_thresh=score_thresh)
                gt_boxes = batch["boxes"][i]
                t, f, n, s = match_frame(preds, gt_boxes, iou_thresh)
                tp += t
                fp += f
                fn += n
                iou_sum += s
                n_gt += len(gt_boxes)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mean_iou = iou_sum / n_gt if n_gt > 0 else 0.0
    return {"precision": precision, "recall": recall, "mean_best_iou": mean_iou, "n_gt": n_gt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="built")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=10.0, help="max gradient norm (0 disables)")
    parser.add_argument("--wh-weight", type=float, default=0.1, help="loss weight on the width/height head")
    parser.add_argument("--vflip", action="store_true", help="augment: random vertical flip")
    parser.add_argument("--rotate-deg", type=float, default=0.0,
                         help="augment: random rotation in [-deg, deg] (0 disables)")
    parser.add_argument("--gain-jitter", type=float, default=0.0,
                         help="augment: random multiplicative gain in [1-x, 1+x] (0 disables)")
    parser.add_argument("--gamma-jitter", type=float, default=0.0,
                         help="augment: random gamma in [1-x, 1+x] (0 disables)")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--max-train-batches", type=int, default=None,
                         help="cap batches/epoch (smoke-testing on CPU)")
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    train_ds = EventManifestDataset(args.data_dir, split="train", val_frac=args.val_frac,
                                     vflip=args.vflip, rotate_deg=args.rotate_deg,
                                     gain_jitter=args.gain_jitter, gamma_jitter=args.gamma_jitter)
    val_ds = EventManifestDataset(args.data_dir, split="val", val_frac=args.val_frac, augment=False)
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_sampler=SequenceGroupedBatchSampler(train_ds, args.batch_size, shuffle=True),
                               collate_fn=collate_same_res, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_sampler=SequenceGroupedBatchSampler(val_ds, args.batch_size, shuffle=False),
                             collate_fn=collate_same_res, num_workers=args.workers)

    model = CenterNetLite(in_ch=2, width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "hm": 0.0, "wh": 0.0, "off": 0.0}
        n_batches = 0
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            gt_hm = batch["heatmap"].to(device, non_blocking=True)
            gt_wh = batch["wh"].to(device, non_blocking=True)
            gt_off = batch["offset"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred_hm, pred_wh, pred_off = model(images)
                loss, parts = centernet_loss(pred_hm, pred_wh, pred_off, gt_hm, gt_wh, gt_off, mask,
                                              wh_weight=args.wh_weight)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running["total"] += loss.item()
            running["hm"] += parts["hm"]
            running["wh"] += parts["wh"]
            running["off"] += parts["off"]
            n_batches += 1

        scheduler.step()
        n_batches = max(n_batches, 1)
        dt = time.time() - t0
        print(f"Epoch {epoch}: loss={running['total']/n_batches:.4f} "
              f"(hm={running['hm']/n_batches:.4f} wh={running['wh']/n_batches:.4f} off={running['off']/n_batches:.4f}) "
              f"[{n_batches} batches, {dt:.1f}s]")

        metrics = run_validation(model, val_loader, device, args.score_thresh, args.iou_thresh,
                                  max_batches=args.max_val_batches)
        print(f"  val: mean_best_iou={metrics['mean_best_iou']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} (n_gt={metrics['n_gt']})")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if metrics["mean_best_iou"] > best_iou:
            best_iou = metrics["mean_best_iou"]
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best (mean_best_iou={best_iou:.4f}), saved {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
