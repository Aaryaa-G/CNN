"""
Train ShipBoxNet on the Ship/045 event voxel grid -> 3-ship bounding boxes.

With only 50 labeled bins total, a single train/val split throws away most
of the signal and gives a noisy estimate of performance. Instead we use
K-FOLD CROSS-VALIDATION: every bin is used for validation exactly once,
across K models trained on the remaining bins. This is the standard way to
get a trustworthy performance estimate (and a usable final model, via
ensembling or refitting on all data) when N is this small.

Usage:
    python3 train.py --epochs 60 --k 5
"""
import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShipEventDataset
from model import ShipBoxNet
from box_utils import box_iou, giou_loss

# NOTE: uses the NATIVE-resolution voxel grid rebuilt by build_voxel_grid.py,
# not the provided 128x128 grid -- see README_pipeline.md for why.
# Defaults assume you run this from the same folder where build_voxel_grid.py
# and prepare_data.py wrote their outputs; override with --voxel-path /
# --boxes-path if you put them elsewhere.
VOXEL_PATH = "voxel_grid_native.npy"
BOXES_PATH = "boxes_50bins.npy"


def combined_loss(pred_norm, target_norm):
    """SmoothL1 on normalized coords (fast, stable early on) + GIoU (better
    for actual localization quality). pred/target: (B, num_objects, 4) in [0,1].
    """
    l1 = nn.functional.smooth_l1_loss(pred_norm, target_norm)
    giou = giou_loss(pred_norm, target_norm)
    return l1 + giou, l1.item(), giou.item()


def evaluate(model, loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for voxels, boxes in loader:
            voxels, boxes = voxels.to(device), boxes.to(device)
            pred = model(voxels)
            iou = box_iou(pred, boxes)  # (B, num_objects)
            ious.append(iou.cpu().numpy())
    return np.concatenate(ious, axis=0) if ious else np.zeros((0, 3))


def compute_init_boxes(boxes_path, train_idx, native_w, native_h, num_objects=3):
    """Mean (cx,cy,w,h) per object over the TRAINING fold only (native px ->
    normalized [0,1]), used to bias-initialize the model's output layer so
    it starts near the right answer instead of a random oversized guess.
    """
    all_boxes = np.load(boxes_path)  # (N, num_objects, 4) in native [x1,y1,x2,y2]
    train_boxes = all_boxes[train_idx]
    init = np.zeros((num_objects, 4))
    for oid in range(num_objects):
        x1, y1, x2, y2 = (train_boxes[:, oid, i] for i in range(4))
        init[oid] = [
            ((x1 + x2) / 2 / native_w).mean(),
            ((y1 + y2) / 2 / native_h).mean(),
            ((x2 - x1) / native_w).mean(),
            ((y2 - y1) / native_h).mean(),
        ]
    return init


def train_one_fold(voxel_path, boxes_path, train_idx, val_idx, epochs, lr, device, seed):
    torch.manual_seed(seed)
    train_ds = ShipEventDataset(voxel_path, boxes_path, indices=train_idx, augment=True, seed=seed)
    val_ds = ShipEventDataset(voxel_path, boxes_path, indices=val_idx, augment=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    init_boxes = compute_init_boxes(boxes_path, train_idx, native_w=1345, native_h=451)
    model = ShipBoxNet(num_objects=3, init_boxes_cxcywh=init_boxes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_iou = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for voxels, boxes in train_loader:
            voxels, boxes = voxels.to(device), boxes.to(device)
            pred = model(voxels)
            loss, _, _ = combined_loss(pred, boxes)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_ious = evaluate(model, val_loader, device)
        mean_val_iou = val_ious.mean() if val_ious.size else 0.0
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--k", type=int, default=5, help="number of CV folds")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voxel-path", default=VOXEL_PATH,
                         help="Path to voxel_grid_native.npy (from build_voxel_grid.py)")
    parser.add_argument("--boxes-path", default=BOXES_PATH,
                         help="Path to boxes_50bins.npy (from prepare_data.py)")
    parser.add_argument("--out-dir", default=".",
                         help="Where to save model_fold*.pt and cv_ious.npy")
    args = parser.parse_args()

    for p in (args.voxel_path, args.boxes_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find: {p}\n"
                f"Run build_voxel_grid.py and prepare_data.py first, or pass "
                f"--voxel-path / --boxes-path pointing at their outputs."
            )
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    n_samples = np.load(args.boxes_path).shape[0]  # 50
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, args.k)

    fold_ious = []
    per_object_ious = []
    for fold_i in range(args.k):
        val_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(args.k) if j != fold_i])

        model, best_val_iou = train_one_fold(
            args.voxel_path, args.boxes_path, train_idx, val_idx,
            args.epochs, args.lr, device, seed=args.seed + fold_i
        )
        print(f"[fold {fold_i}] n_train={len(train_idx)} n_val={len(val_idx)} "
              f"best_mean_IoU={best_val_iou:.4f}")

        # per-object breakdown on this fold's validation bins
        val_ds = ShipEventDataset(args.voxel_path, args.boxes_path, indices=val_idx, augment=False)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        ious = evaluate(model, val_loader, device)  # (n_val, 3)
        per_object_ious.append(ious)
        fold_ious.append(best_val_iou)

        torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_fold{fold_i}.pt"))

    all_ious = np.concatenate(per_object_ious, axis=0)  # (50, 3)
    print("\n=== Cross-validated results (every bin held out exactly once) ===")
    print(f"Mean IoU across all bins/objects: {all_ious.mean():.4f}")
    for oid in range(all_ious.shape[1]):
        print(f"  Object {oid + 1} mean IoU: {all_ious[:, oid].mean():.4f}")
    print(f"Per-fold mean IoU: {[round(f, 4) for f in fold_ious]}")

    np.save(os.path.join(args.out_dir, "cv_ious.npy"), all_ious)


if __name__ == "__main__":
    main()
