"""PyTorch Dataset over the built event-frame dataset (see build_dataset.py).

Each sequence has its own resolution, so samples are grouped by sequence and
batched with SequenceGroupedSampler (train.py) rather than padded to a
common canvas -- this avoids wasting GPU memory on empty padding for the
smaller sequence.
"""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from centernet_utils import build_targets

STRIDE = 4


def _rotate_grid_and_boxes(grid, boxes, angle_deg, h, w):
    """Rotate a (2,H,W) event grid about the image center by angle_deg and
    remap each box to the axis-aligned box enclosing its rotated corners
    (exact box shape isn't preserved under rotation, but this is standard
    practice for detection augmentation without per-object masks)."""
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = np.stack([
        cv2.warpAffine(grid[c], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        for c in range(grid.shape[0])
    ]).astype(grid.dtype)

    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        rot = corners @ M.T
        nx1, ny1 = max(0.0, rot[:, 0].min()), max(0.0, rot[:, 1].min())
        nx2, ny2 = min(float(w), rot[:, 0].max()), min(float(h), rot[:, 1].max())
        if nx2 > nx1 and ny2 > ny1:
            new_boxes.append([nx1, ny1, nx2, ny2])
    return rotated, new_boxes


class EventManifestDataset(Dataset):
    def __init__(self, data_dir, split="train", val_frac=0.15, augment=None,
                 vflip=False, rotate_deg=0.0, gain_jitter=0.0, gamma_jitter=0.0):
        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)

        self.data_dir = data_dir
        self.entries = []  # each: dict with seq npy path (lazy-opened), index, boxes, out size
        self._memmaps = {}
        self.seq_info = {}

        for seq in manifest["sequences"]:
            n = seq["n_samples"]
            val_count = max(1, int(round(n * val_frac)))
            # chronological holdout: last val_count samples of each sequence are val
            split_idx = n - val_count
            self.seq_info[seq["name"]] = seq
            for s in seq["samples"]:
                is_val = s["index_in_sequence"] >= split_idx
                if (split == "val") != is_val:
                    continue
                self.entries.append(s)

        self.augment = augment if augment is not None else (split == "train")
        self.vflip = vflip
        self.rotate_deg = rotate_deg
        self.gain_jitter = gain_jitter
        self.gamma_jitter = gamma_jitter

    def _get_memmap(self, seq_name):
        if seq_name not in self._memmaps:
            seq = self.seq_info[seq_name]
            path = os.path.join(self.data_dir, seq["npy_path"])
            self._memmaps[seq_name] = np.load(path, mmap_mode="r")
        return self._memmaps[seq_name]

    def __len__(self):
        return len(self.entries)

    def resolution_key(self, idx):
        seq = self.seq_info[self.entries[idx]["sequence"]]
        return (seq["padded_h"], seq["padded_w"])

    def __getitem__(self, idx):
        entry = self.entries[idx]
        seq = self.seq_info[entry["sequence"]]
        grid = np.array(self._get_memmap(entry["sequence"])[entry["index_in_sequence"]])  # (2,H,W)
        boxes = [list(b) for b in entry["boxes"]]

        h, w = seq["padded_h"], seq["padded_w"]
        if self.augment:
            if random.random() < 0.5:
                grid = grid[:, :, ::-1].copy()
                boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]
            if self.vflip and random.random() < 0.5:
                grid = grid[:, ::-1, :].copy()
                boxes = [[x1, h - y2, x2, h - y1] for x1, y1, x2, y2 in boxes]
            if self.rotate_deg > 0 and random.random() < 0.5:
                angle = random.uniform(-self.rotate_deg, self.rotate_deg)
                grid, boxes = _rotate_grid_and_boxes(grid, boxes, angle, h, w)
            if self.gain_jitter > 0:
                gain = 1.0 + random.uniform(-self.gain_jitter, self.gain_jitter)
                grid = np.clip(grid * gain, 0.0, 1.0)
            if self.gamma_jitter > 0:
                gamma = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
                grid = np.power(np.clip(grid, 0.0, 1.0), gamma)
            grid = grid.astype(np.float32)

        out_h, out_w = seq["padded_h"] // STRIDE, seq["padded_w"] // STRIDE
        heatmap, wh, offset, mask = build_targets(boxes, out_h, out_w, STRIDE)

        return {
            "image": torch.from_numpy(grid),
            "heatmap": torch.from_numpy(heatmap),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "mask": torch.from_numpy(mask),
            "boxes": boxes,
            "sequence": entry["sequence"],
            "frame_id": entry["frame_id"],
        }


def collate_same_res(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "wh": torch.stack([b["wh"] for b in batch]),
        "offset": torch.stack([b["offset"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "sequence": [b["sequence"] for b in batch],
        "frame_id": [b["frame_id"] for b in batch],
    }
    return out


class SequenceGroupedBatchSampler(Sampler):
    """Yields batches whose samples all come from the same underlying
    sequence resolution, so no padding is ever needed at collate time."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = {}
        for i in range(len(dataset)):
            groups.setdefault(dataset.resolution_key(i), []).append(i)
        self.groups = list(groups.values())

    def __iter__(self):
        batches = []
        for idxs in self.groups:
            idxs = list(idxs)
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i:i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum((len(idxs) + self.batch_size - 1) // self.batch_size for idxs in self.groups)
