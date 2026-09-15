"""
Run a trained detector across an ENTIRE sequence and link the per-frame
detections into persistent tracks (simple greedy IoU/centroid matching, not
a learned tracker), so ships are followed across frames rather than just
detected independently -- renders an .mp4 with each track's trail drawn
behind its current box.

Usage:
    python track_sequence.py --data-dir built_dvs --checkpoint checkpoints/best.pt \
        --sequence ship_045 --out preview/ship_045_tracked.mp4 --device cuda
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from model import CenterNetLite
from centernet_utils import decode_boxes
from eval_utils import box_iou
from dataset import STRIDE


def event_frame_to_rgb(grid, gamma=0.45):
    """gamma < 1 boosts dim pixels for human visibility -- normalize_channel()
    scales each frame by its own max, so with DVS-Voltmeter's heavy dark-current
    noise most real ship-edge pixels are small fractions of that max and render
    almost black under a linear 0-255 mapping. Display-only; doesn't touch the
    underlying data used for training."""
    off, on = grid[0], grid[1]
    off_disp = np.power(np.clip(off, 0, 1), gamma)
    on_disp = np.power(np.clip(on, 0, 1), gamma)
    rgb = np.zeros((*off.shape, 3), dtype=np.uint8)
    rgb[..., 2] = (off_disp * 255).astype(np.uint8)  # OFF -> red (BGR: R at idx 2)
    rgb[..., 0] = (on_disp * 255).astype(np.uint8)   # ON -> blue (BGR: B at idx 0)
    return rgb


class GreedyTracker:
    """Minimal IoU-then-centroid-distance greedy tracker. Good enough to draw
    trails on a small-object, low-frame-count sequence; swap in SORT/ByteTrack
    if you need something more robust for real deployment."""

    def __init__(self, iou_thresh=0.1, max_dist=40, max_missed=5):
        self.iou_thresh = iou_thresh
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks = {}  # id -> {"box":..., "trail":[centers], "missed":0}
        self.next_id = 0

    @staticmethod
    def center(box):
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def step(self, detections):
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = list(range(len(detections)))
        matches = []

        for tid in list(unmatched_tracks):
            best_j, best_score = -1, -1.0
            for j in unmatched_dets:
                iou = box_iou(self.tracks[tid]["box"][:4], detections[j][:4])
                if iou > best_score:
                    best_score, best_j = iou, j
            if best_j >= 0 and best_score >= self.iou_thresh:
                matches.append((tid, best_j))
                unmatched_tracks.discard(tid)
                unmatched_dets.remove(best_j)

        for tid in list(unmatched_tracks):
            tc = self.center(self.tracks[tid]["box"])
            best_j, best_d = -1, self.max_dist
            for j in unmatched_dets:
                dc = self.center(detections[j])
                d = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                matches.append((tid, best_j))
                unmatched_tracks.discard(tid)
                unmatched_dets.remove(best_j)

        for tid, j in matches:
            self.tracks[tid]["box"] = detections[j]
            self.tracks[tid]["trail"].append(self.center(detections[j]))
            self.tracks[tid]["missed"] = 0

        for tid in unmatched_tracks:
            self.tracks[tid]["missed"] += 1

        for j in unmatched_dets:
            self.tracks[self.next_id] = {"box": detections[j], "trail": [self.center(detections[j])], "missed": 0}
            self.next_id += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}
        return self.tracks


TRACK_COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 0, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="built_dvs")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--sequence", required=True, help="e.g. ship_045")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.45,
                         help="<1 brightens dim event pixels for visibility (display only)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CenterNetLite(in_ch=2, width=ckpt["args"]["width"], depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    seq = next(s for s in manifest["sequences"] if s["name"] == args.sequence)
    grid = np.load(os.path.join(args.data_dir, seq["npy_path"]), mmap_mode="r")

    out_path = args.out or os.path.join("preview", f"{args.sequence}_tracked.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                              (seq["padded_w"], seq["padded_h"]))

    tracker = GreedyTracker()

    with torch.no_grad():
        for i, sample in enumerate(seq["samples"]):
            frame = np.array(grid[sample["index_in_sequence"]])
            image = torch.from_numpy(frame).unsqueeze(0).to(device)
            heatmap, wh, offset = model(image)
            preds = decode_boxes(heatmap[0], wh[0], offset[0], STRIDE, score_thresh=args.score_thresh)

            tracks = tracker.step(preds)

            canvas = event_frame_to_rgb(frame, gamma=args.gamma)
            for tid, t in tracks.items():
                color = TRACK_COLORS[tid % len(TRACK_COLORS)]
                trail = t["trail"][-20:]
                for k in range(1, len(trail)):
                    cv2.line(canvas, tuple(map(int, trail[k - 1])), tuple(map(int, trail[k])), color, 1)
                if t["missed"] == 0:
                    x1, y1, x2, y2 = map(int, t["box"][:4])
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(canvas, f"#{tid}", (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            writer.write(canvas)

    writer.release()
    print(f"Saved {out_path} ({len(seq['samples'])} frames, {len(tracker.tracks)} live tracks at end)")


if __name__ == "__main__":
    main()
