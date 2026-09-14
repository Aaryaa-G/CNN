"""
Dataset for Ship/045 event voxel grids -> 3-object bounding boxes.

Input:  voxel grid (50, 2, 451, 1345), float32, normalized [0,1] per (bin,channel)
        -- built at NATIVE VISO resolution by build_voxel_grid.py, because the
        provided 128x128 grid crushes ~22x11 px ships down to ~3x5 px (unusable
        for IoU-based regression).
Labels: boxes_50bins.npy (50, 3, 4) in [x1,y1,x2,y2], native pixel coords

Because we only have 50 samples total, this dataset supports:
  - k-fold index splitting (done outside, see train.py)
  - light augmentation (horizontal flip, small translation, additive noise)
    with correct box transforms.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

NATIVE_W, NATIVE_H = 1345, 451


class ShipEventDataset(Dataset):
    def __init__(self, voxel_path, boxes_path, indices=None, augment=False, seed=0):
        self.voxels = np.load(voxel_path).astype(np.float32)   # (50, 2, 128, 128)
        self.boxes = np.load(boxes_path).astype(np.float32)    # (50, 3, 4)
        assert self.voxels.shape[0] == self.boxes.shape[0]

        self.indices = list(range(self.voxels.shape[0])) if indices is None else list(indices)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def _augment(self, voxel, boxes):
        # voxel: (2, NATIVE_H, NATIVE_W) ; boxes: (3,4) in [x1,y1,x2,y2]
        boxes = boxes.copy()

        # Random horizontal flip
        if self.rng.random() < 0.5:
            voxel = voxel[:, :, ::-1].copy()
            x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0] = NATIVE_W - x2
            boxes[:, 2] = NATIVE_W - x1

        # Random vertical flip
        if self.rng.random() < 0.5:
            voxel = voxel[:, ::-1, :].copy()
            y1, y2 = boxes[:, 1].copy(), boxes[:, 3].copy()
            boxes[:, 1] = NATIVE_H - y2
            boxes[:, 3] = NATIVE_H - y1

        # Small random integer translation (shared across channels + all boxes)
        max_shift = 30  # pixels, scaled up vs. the old 128-grid version since native is ~10x larger
        dx = int(self.rng.integers(-max_shift, max_shift + 1))
        dy = int(self.rng.integers(-max_shift // 3, max_shift // 3 + 1))  # H is much smaller than W
        if dx != 0 or dy != 0:
            voxel = np.roll(voxel, shift=(dy, dx), axis=(1, 2))
            # zero-out wrapped-around border region introduced by roll
            if dy > 0:
                voxel[:, :dy, :] = 0
            elif dy < 0:
                voxel[:, dy:, :] = 0
            if dx > 0:
                voxel[:, :, :dx] = 0
            elif dx < 0:
                voxel[:, :, dx:] = 0
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, NATIVE_W)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, NATIVE_H)

        # Mild additive noise / random event dropout (helps regularize on 50 samples)
        if self.rng.random() < 0.3:
            drop_mask = self.rng.random(voxel.shape) < 0.05
            voxel = voxel * (~drop_mask)

        return voxel.astype(np.float32), boxes.astype(np.float32)

    def __getitem__(self, i):
        idx = self.indices[i]
        voxel = self.voxels[idx].copy()   # (2, NATIVE_H, NATIVE_W)
        boxes = self.boxes[idx].copy()    # (3,4)

        if self.augment:
            voxel, boxes = self._augment(voxel, boxes)

        # normalize box coords to [0,1] independently per axis for stable
        # regression targets (x by width, y by height -- NOT the same scalar,
        # since the frame is rectangular, not square).
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= NATIVE_W
        boxes_norm[:, [1, 3]] /= NATIVE_H

        return torch.from_numpy(voxel), torch.from_numpy(boxes_norm)
