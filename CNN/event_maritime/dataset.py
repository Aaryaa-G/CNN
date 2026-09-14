"""PyTorch Dataset over the built event-frame dataset (see build_dataset.py).

Each sequence has its own resolution, so samples are grouped by sequence and
batched with SequenceGroupedSampler (train.py) rather than padded to a
common canvas -- this avoids wasting GPU memory on empty padding for the
smaller sequence.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from centernet_utils import build_targets

STRIDE = 4


class EventManifestDataset(Dataset):
    def __init__(self, data_dir, split="train", val_frac=0.15, augment=None):
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

        if self.augment and random.random() < 0.5:
            grid = grid[:, :, ::-1].copy()
            w = seq["padded_w"]
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

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
