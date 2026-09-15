"""
Render the original VISO video frame side-by-side with its corresponding
event frame (both with the same ground-truth box drawn, plus the trained
model's predicted box if --checkpoint is given) -- lets you directly see
where the ship actually is in the real footage versus what the event
representation and detector see, instead of judging the event frame alone.

Usage:
    python compare_real_events.py --data-dir built_dvs \
        --viso-root "../Neuromorphic Camera/VISO" --sequence ship_045 \
        --checkpoint checkpoints/best.pt --out preview/ship_045_compare.mp4 --device cuda
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from model import CenterNetLite
from centernet_utils import decode_boxes
from dataset import STRIDE


def event_frame_to_rgb(grid, gamma=0.45):
    """gamma < 1 boosts dim pixels for human visibility (display only) -- see
    the same helper in track_sequence.py for why this is needed."""
    off, on = grid[0], grid[1]
    off_disp = np.power(np.clip(off, 0, 1), gamma)
    on_disp = np.power(np.clip(on, 0, 1), gamma)
    rgb = np.zeros((*off.shape, 3), dtype=np.uint8)
    rgb[..., 2] = (off_disp * 255).astype(np.uint8)  # OFF -> red (BGR)
    rgb[..., 0] = (on_disp * 255).astype(np.uint8)   # ON -> blue (BGR)
    return rgb


def draw_box(img, box, color, label=None, thickness=1):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="built_dvs")
    parser.add_argument("--viso-root", default=os.path.join("..", "Neuromorphic Camera", "VISO"))
    parser.add_argument("--sequence", required=True, help="e.g. ship_045")
    parser.add_argument("--checkpoint", default=None, help="optional; overlays predicted boxes too")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.45,
                         help="<1 brightens dim event pixels for visibility (display only)")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    seq = next(s for s in manifest["sequences"] if s["name"] == args.sequence)
    grid = np.load(os.path.join(args.data_dir, seq["npy_path"]), mmap_mode="r")

    category, seq_id = args.sequence.split("_", 1)
    img_dir = os.path.join(args.viso_root, "mot", category, seq_id, "img")

    model = None
    device = torch.device(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = CenterNetLite(in_ch=2, width=ckpt["args"]["width"], depth=ckpt["args"]["depth"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    out_path = args.out or os.path.join("preview", f"{args.sequence}_compare.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pad_h, pad_w = seq["padded_h"], seq["padded_w"]
    writer = None

    with torch.no_grad():
        for sample in seq["samples"]:
            frame_id = sample["frame_id"]
            real = cv2.imread(os.path.join(img_dir, f"{frame_id:06d}.jpg"))
            if real is None:
                print(f"warning: missing frame {frame_id:06d}.jpg, skipping")
                continue
            real_panel = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            rh, rw = real.shape[:2]
            real_panel[:rh, :rw] = real

            frame = np.array(grid[sample["index_in_sequence"]])
            event_panel = event_frame_to_rgb(frame, gamma=args.gamma)

            preds = []
            if model is not None:
                image = torch.from_numpy(frame).unsqueeze(0).to(device)
                heatmap, wh, offset = model(image)
                preds = decode_boxes(heatmap[0], wh[0], offset[0], STRIDE, score_thresh=args.score_thresh)

            for panel in (real_panel, event_panel):
                for box in sample["boxes"]:
                    draw_box(panel, box, (0, 255, 0))  # ground truth, green
                for x1, y1, x2, y2, score in preds:
                    draw_box(panel, (x1, y1, x2, y2), (0, 0, 255), f"{score:.2f}")  # prediction, red

            cv2.putText(real_panel, "REAL", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(event_panel, "EVENTS", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            canvas = np.hstack([real_panel, event_panel])
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                          (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)

    if writer is not None:
        writer.release()
    print(f"Saved {out_path} ({len(seq['samples'])} frames)")


if __name__ == "__main__":
    main()
