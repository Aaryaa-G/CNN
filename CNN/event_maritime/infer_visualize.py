"""Run a trained checkpoint on a handful of validation event-frames and save
side-by-side PNGs (event frame with predicted vs. ground-truth boxes drawn)
-- a quick visual sanity check / demo for the maritime-surveillance use case.

Usage:
    python infer_visualize.py --data-dir built --checkpoint checkpoints/best.pt --n 8 --out-dir preview
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import EventManifestDataset, STRIDE
from model import CenterNetLite
from centernet_utils import decode_boxes


def event_frame_to_rgb(grid):
    off, on = grid[0], grid[1]
    rgb = np.zeros((*off.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (off * 255).clip(0, 255).astype(np.uint8)  # OFF -> red
    rgb[..., 2] = (on * 255).clip(0, 255).astype(np.uint8)   # ON -> blue
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="built")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out-dir", default="preview")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CenterNetLite(in_ch=2, width=ckpt["args"]["width"], depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = EventManifestDataset(args.data_dir, split="val", augment=False)
    n = min(args.n, len(val_ds))
    idxs = np.linspace(0, len(val_ds) - 1, n).astype(int)

    with torch.no_grad():
        for k, idx in enumerate(idxs):
            sample = val_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            heatmap, wh, offset = model(image)
            preds = decode_boxes(heatmap[0], wh[0], offset[0], STRIDE, score_thresh=args.score_thresh)

            img = event_frame_to_rgb(sample["image"].numpy())
            draw = ImageDraw.Draw(img)
            for x1, y1, x2, y2 in sample["boxes"]:
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            for x1, y1, x2, y2, score in preds:
                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=1)

            out_path = os.path.join(args.out_dir, f"{sample['sequence']}_frame{sample['frame_id']:04d}.png")
            img.save(out_path)
            print(f"Saved {out_path} ({len(sample['boxes'])} gt, {len(preds)} pred boxes)")


if __name__ == "__main__":
    main()
